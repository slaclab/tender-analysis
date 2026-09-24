"""Incremental HERFD while a RIXS series is still being acquired.

:class:`HerfdAccumulator` takes one ``.sif`` at a time and can produce the
HERFD/TFY line-out at any point, matching what
``OnePotRIXS(files, ...).herfd(...)`` gives once every file is in.

How it stays faithful to :class:`~tender_analysis.pipeline.OnePotRIXS`:

- each file is reduced with :func:`~tender_analysis.analyze.reduce_frame`
  (the body of ``extract_signal``) against the same frame-averaged dark, so its
  signal plane is bit-identical to ``OnePotRIXS``'s scan-mode plane;
- ``OnePotRIXS`` fits ONE curvature on the sum of all scan-point signals and
  applies it to every per-file plane. So the curvature is not known until the
  series is complete: :meth:`result` refits it on the accumulated sum and
  re-applies it to every stored plane. The planes are kept sparse (a reduced
  frame is almost all zeros), so nothing is re-read from disk;
- incident energy is ``mono`` from the SIF comment, else the ``dddd.dd``
  filename token; I0 is the comment's ``I0`` (``0``/missing -> 1), and the
  central pixel is auto-fitted with the same ``_fit_central_pixel``;
- stored files are ordered like ``OnePotRIXS`` orders them (natural sort of the
  path), so the auto-fit and the energy sort see the same column order
  regardless of the order ``add`` was called in.

Unlike ``OnePotRIXS`` with no dark, there is no min-projection background: it
needs every frame up front. With ``dark=None`` the first ``*_dark.sif`` passed to
:meth:`add` before any scan point is adopted as the background; otherwise no
background is subtracted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from natsort import natsort_keygen

from .analyze import background_common_mode, reduce_frame
from .common import common_mode
from .curvature import CurvatureCorrection
from .pipeline import RIXSResult, Thresholds, _energy_from_name, _fit_central_pixel
from .preview import resolve_background
from .sif_io import SifFile

__all__ = ["HerfdAccumulator"]

_natkey = natsort_keygen()


@dataclass
class _Plane:
    """One file's reduced signal, stored sparse."""

    path: str
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    energy: float
    i0: float

    def dense(self, shape) -> np.ndarray:
        img = np.zeros(shape)
        img[self.rows, self.cols] = self.vals
        return img


class HerfdAccumulator:
    """Accumulate a RIXS series file by file; read HERFD out at any time.

    Parameters
    ----------
    central_pix, n, i0_corr:
        As :meth:`OnePotRIXS.herfd` (``central_pix=None`` auto-fits on the
        accumulated map). Change later with :meth:`set_roi`.
    dark:
        Background: a dark ``.sif`` path, a ``(height, width)`` array, or
        ``None`` (see the module docstring).
    thresholds:
        Anything :meth:`Thresholds.from_input` accepts.
    curvature_t:
        ``None`` (default) refits on the accumulated signal, as ``OnePotRIXS``
        does over the full series; fixed coefficients; or ``1`` for none.
    bcg_adjust, dark_suffix:
        As ``OnePotRIXS``.
    """

    def __init__(self, central_pix=None, n: int = 7, i0_corr: bool = True,
                 dark=None, thresholds=None, curvature_t=None,
                 bcg_adjust: bool = True, dark_suffix: str = "_dark"):
        self.central_pix = central_pix
        self.n = int(n)
        self.i0_corr = i0_corr
        self.thresholds = Thresholds.from_input(thresholds)
        self.curvature_t = curvature_t
        self.bcg_adjust = bcg_adjust
        self.dark_suffix = dark_suffix
        self._dark_input = dark
        self.dark_path = os.fspath(dark) if isinstance(dark, (str, os.PathLike)) else None
        self._bcg = None
        self._bcg_cm = None
        self._shape = None
        self._planes: list[_Plane] = []
        self._sum = None
        # corrected per-file spectra cache: valid for (t, paths) it was built for
        self._cache_t = None
        self._cache: dict[str, np.ndarray] = {}

    # -- input ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._planes)

    @property
    def paths(self) -> list[str]:
        return [p.path for p in self._ordered()]

    def _is_dark(self, path: str) -> bool:
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.lower().endswith(self.dark_suffix.lower())

    def _ensure_background(self, shape) -> None:
        if self._bcg is None:
            self._shape = tuple(shape)
            self._bcg = resolve_background(self._dark_input, self._shape)
            self._bcg_cm = background_common_mode(self._bcg, self.bcg_adjust)
            self._sum = np.zeros(self._shape)

    def add(self, path) -> dict | None:
        """Reduce one file into the accumulator.

        Returns ``{"path", "energy", "I0", "n_frames"}`` for a scan point, or
        ``None`` when the file was a dark (adopted as background if none was
        set and no scan point has been added yet, otherwise ignored) or a
        duplicate of a path already added.
        """
        path = os.fspath(path)
        if self._is_dark(path):
            if self._dark_input is None and self._bcg is None:
                self._dark_input = self.dark_path = path
            return None
        if any(p.path == path for p in self._planes):
            return None

        sif = SifFile(path)
        self._ensure_background(sif.shape)
        if tuple(sif.shape) != self._shape:
            raise ValueError(f"{path}: frame shape {sif.shape} != {self._shape}")

        plane = np.zeros(self._shape)
        for i in range(sif.num_frames):
            image = sif.frame(i)
            frame, _ = reduce_frame(image, self._bcg, self._bcg_cm, self.thresholds,
                                    cm=common_mode(image, refine=True))
            plane += frame
        self._sum += plane

        rows, cols = np.nonzero(plane)
        energy = sif.mono
        if np.isnan(energy):
            energy = _energy_from_name(sif.path)
        self._planes.append(_Plane(path=path, rows=rows.astype(np.int32),
                                   cols=cols.astype(np.int32), vals=plane[rows, cols],
                                   energy=float(energy), i0=float(sif.I0)))
        return {"path": path, "energy": float(energy), "I0": float(sif.I0),
                "n_frames": sif.num_frames}

    def set_roi(self, central_pix=None, n: int | None = None) -> None:
        """Change the HERFD band; the next :meth:`result` re-slices, no re-read."""
        self.central_pix = central_pix
        if n is not None:
            self.n = int(n)

    # -- output -----------------------------------------------------------

    def _ordered(self) -> list[_Plane]:
        return sorted(self._planes, key=lambda p: _natkey(p.path))

    def curvature(self) -> CurvatureCorrection:
        """The curvature correction for the data accumulated so far."""
        if not self._planes:
            raise ValueError("no scan points added yet")
        cc = CurvatureCorrection(t=self.curvature_t)
        cc.fit(self._sum)
        return cc

    def rixs_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(map, E, I0)`` in file order: map is ``(pixel, file)``, NOT I0-divided."""
        cc = self.curvature()
        t_key = None if cc._identity else np.asarray(cc.t, dtype=float).tobytes()
        if t_key != self._cache_t:
            self._cache, self._cache_t = {}, t_key
        planes = self._ordered()
        cols = []
        for p in planes:
            spec = self._cache.get(p.path)
            if spec is None:
                spec = cc.apply(p.dense(self._shape))[0].sum(axis=0)
                self._cache[p.path] = spec
            cols.append(spec)
        E = np.array([p.energy for p in planes], dtype=float)
        i0 = np.array([p.i0 for p in planes], dtype=float)
        return np.array(cols).T, E, i0

    def result(self) -> RIXSResult:
        """HERFD/TFY line-out from everything added so far (see ``OnePotRIXS.herfd``)."""
        rixs, E, i0 = self.rixs_map()
        if self.i0_corr:
            i0 = np.where((i0 == 0) | np.isnan(i0), 1.0, i0)
            rixs = rixs / i0[np.newaxis, :]

        central_pix = self.central_pix
        if central_pix is None:
            central_pix = _fit_central_pixel(rixs)

        rng = np.arange(self.n) - self.n // 2
        band = np.clip(central_pix + rng, 0, rixs.shape[0] - 1)
        HERFD = rixs[band, :].sum(axis=0)
        TFY = rixs.sum(axis=0)

        order = np.argsort(E)
        meta = {
            "pipeline": type(self).__name__,
            "n_files": len(self._planes),
            "source_files": [os.path.basename(p) for p in self.paths],
            "thresholds [bcg_cutoff, low, xray, hi]":
                np.array2string(self.thresholds.as_array(), precision=1),
            "background": (os.path.basename(self.dark_path) if self.dark_path
                           else "supplied array" if self._dark_input is not None
                           else "none"),
            "bcg_adjust": self.bcg_adjust,
            "herfd_band_width_px": self.n,
            "central_pix": int(central_pix),
            "i0_corrected": self.i0_corr,
        }
        return RIXSResult(E=E[order], HERFD=HERFD[order], TFY=TFY[order],
                          central_pix=int(central_pix), rixs_map=rixs[:, order],
                          meta=meta)
