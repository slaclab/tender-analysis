"""Quick-look reduction of ONE ``.sif`` file for a viewer.

:func:`preview` runs the same per-frame reduction as :class:`OnePot`
(:func:`~tender_analysis.analyze.reduce_frame`) and the same curvature step, then
packages what a browser/notebook viewer needs: small percentile-stretched
``uint8`` images (raw, background-subtracted, extracted events) plus the
full-resolution corrected spectrum and scalar metadata. No maths is duplicated
here -- the spectrum is exactly ``OnePot([path], bcg=...).run().spectrum()``.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import label

from .analyze import _CONN4, background_common_mode, reduce_frame, subtract_pedestal
from .common import common_mode
from .curvature import CurvatureCorrection
from .pipeline import Thresholds, _energy_from_name
from .sif_io import SifFile

__all__ = ["preview", "resolve_background", "stretch_uint8", "block_downsample"]


def resolve_background(dark, shape) -> np.ndarray:
    """Background image from ``dark``: ``None``/``0`` -> zeros (no background),
    a ``.sif`` path or :class:`SifFile` -> its frame-averaged mean (as
    ``OnePotRIXS(use_dark_as_background=True)`` and ``Measurement`` do), or an
    array used as-is."""
    if dark is None or (np.isscalar(dark) and not isinstance(dark, str) and dark == 0):
        return np.zeros(shape)
    if isinstance(dark, (str, os.PathLike)):
        dark = SifFile(os.fspath(dark))
    if isinstance(dark, SifFile):
        return dark.data.mean(axis=0)
    bcg = np.asarray(dark, dtype=float)
    if bcg.shape != tuple(shape):
        raise ValueError(f"background shape {bcg.shape} != frame shape {tuple(shape)}")
    return bcg


def block_downsample(image: np.ndarray, factors=(4, 2)) -> np.ndarray:
    """Block-MEAN ``image`` by ``(row, col)`` factors (ragged edges trimmed)."""
    fr, fc = (int(f) for f in factors)
    if fr < 1 or fc < 1:
        raise ValueError(f"downsample factors must be >= 1, got {factors}")
    h, w = image.shape
    h2, w2 = h // fr, w // fc
    img = np.asarray(image, dtype=float)[:h2 * fr, :w2 * fc]
    return img.reshape(h2, fr, w2, fc).mean(axis=(1, 3))


def stretch_uint8(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5):
    """Percentile-stretch ``image`` to ``uint8``; returns ``(img8, (lo, hi))``.

    Sparse images (extracted events are mostly zero) can have equal
    percentiles; the upper limit then falls back to the image maximum so the
    events stay visible instead of the whole frame saturating.
    """
    img = np.asarray(image, dtype=float)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype=np.uint8), (0.0, 0.0)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8), (float(lo), float(hi))
    scaled = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.round(scaled * 255).astype(np.uint8), (float(lo), float(hi))


def preview(path, dark=None, thresholds=None, curvature_t=None, downsample=(4, 2),
            frame: int | None = None, bcg_adjust: bool = True) -> dict:
    """Reduce one ``.sif`` file and return viewer-ready outputs.

    Parameters
    ----------
    path:
        The ``.sif`` file.
    dark:
        Background: ``None`` (none), a dark ``.sif`` path (frame-averaged), or
        an array. See :func:`resolve_background`.
    thresholds:
        Anything :meth:`Thresholds.from_input` accepts (``None`` = defaults).
    curvature_t:
        ``None`` fits the curvature on this file's signal exactly as
        :meth:`OnePot.run` does; an array applies fixed coefficients; ``1`` is
        the identity (no correction).
    downsample:
        ``(row, col)`` block-mean factors for the returned images.
    frame:
        0-based frame to reduce; ``None`` sums all frames (as ``OnePot``).

    Returns
    -------
    dict
        ``raw``, ``bkg_sub``, ``events`` -- ``uint8`` downsampled images (summed
        raw frames; pedestal-scaled background-subtracted frames; the cleaned
        single-photon signal before curvature correction). ``limits`` -- their
        ``(lo, hi)`` stretch limits in downsampled ADU. ``spectrum`` -- the
        full-resolution curvature-corrected column sum (length = width);
        ``row_profile`` -- the corrected signal summed over columns (length =
        height). ``I0``, ``mono`` (``nan`` if absent), ``energy`` (mono, else
        the filename token), ``exposure``, ``n_events`` (surviving photon
        grains), ``shape`` ``(n_frames, height, width)``, ``frames`` used,
        ``t`` (curvature coefficients), ``thresholds``, ``downsample``.
    """
    sif = SifFile(os.fspath(path))
    th = Thresholds.from_input(thresholds)
    bcg = resolve_background(dark, sif.shape)
    bcg_cm = background_common_mode(bcg, bcg_adjust)

    frames = range(sif.num_frames) if frame is None else [int(frame)]
    raw = np.zeros(sif.shape)
    bkg_sub = np.zeros(sif.shape)
    signal = np.zeros(sif.shape)
    n_events = 0
    for i in frames:
        image = sif.frame(i)
        raw += image
        cm = common_mode(image, refine=True)
        sub, _ = subtract_pedestal(image, bcg, bcg_cm, cm=cm)
        bkg_sub += sub
        sig, masks = reduce_frame(image, bcg, bcg_cm, th, cm=cm)
        signal += sig
        n_events += int(label(masks["grains"], structure=_CONN4)[1])

    # The same curvature step OnePot.run takes on its summed signal.
    cc = CurvatureCorrection(t=curvature_t)
    corrected, _ = cc.fit_apply(signal)
    t = cc.t
    t_out = t if (np.isscalar(t) or t is None) else np.asarray(t, dtype=float)

    out: dict = {"limits": {}}
    for key, img in (("raw", raw), ("bkg_sub", bkg_sub), ("events", signal)):
        out[key], out["limits"][key] = stretch_uint8(block_downsample(img, downsample))

    mono = sif.mono
    out.update({
        "spectrum": corrected.sum(axis=0),
        "row_profile": corrected.sum(axis=1),
        "I0": sif.I0,
        "mono": mono,
        "energy": mono if np.isfinite(mono) else _energy_from_name(sif.path),
        "exposure": sif.exposure_time,
        "n_events": n_events,
        "shape": (sif.num_frames,) + tuple(sif.shape),
        "frames": list(frames),
        "t": t_out,
        "thresholds": th.as_array(),
        "downsample": tuple(int(f) for f in downsample),
        "path": sif.path,
    })
    return out
