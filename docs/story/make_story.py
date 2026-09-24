#!/usr/bin/env python
"""Build a self-contained HTML "data story" for Tender X-ray (BL 6-2a) data.

Every figure is computed from real ``.sif`` data with ``tender_analysis`` and
inlined as a base64 PNG, so the output is ONE file with no external resources:
no web fonts, no CDN, no scripts.

    python docs/story/make_story.py                       # bundled data only
    python docs/story/make_story.py --real /path/to/story_data --out story.html

``--bundled`` is a RIXS series directory (default ``data/Na2SO4``, series _01).
``--real`` is optional; it may hold any of these subdirectories, and chapters
that need a missing one are skipped with a one-line note:

    Na2SO4_pellet/   a second RIXS series of the same sample (for averaging)
    Ag2S/ AgNO3/ P12S/   Ag L3-valence XES at a fixed incident energy
    elastic_BN/      elastic scans for the pixel -> emission-energy calibration
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import gc
import glob
import html
import inspect
import io
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from scipy.ndimage import label  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import tender_analysis as ta  # noqa: E402
from tender_analysis import average, calibration, export, live, pipeline, plotting  # noqa: E402
from tender_analysis.analyze import (_CONN4, background_common_mode, reduce_frame,  # noqa: E402
                                     subtract_pedestal)
from tender_analysis.common import NBINS, adu_histogram, common_mode  # noqa: E402
from tender_analysis.dataset import parse_sif_name  # noqa: E402
from tender_analysis.preview import preview  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

INK, INK_2, SERIES = plotting.INK, plotting.INK_2, plotting.SERIES
LINESTYLES = ["-", "--", "-.", ":"]
DPI = 80
ROI_N = 7          # the portal's (and HerfdAccumulator's) default band width
# larch pre_edge settings for the S K-edge scans (chapter 10 shows why the
# defaults are not used): linear post-edge from E0+25 eV to the end of the scan.
NORM = {"norm1": 25.0, "nnorm": 1}

matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- figure utils

def fig(w, h, **kw):
    return Figure(figsize=(w, h), layout="constrained", **kw)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def png(f, dpi=DPI) -> str:
    buf = io.BytesIO()
    f.savefig(buf, format="png", dpi=dpi, facecolor="white")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img_tag(f, alt, caption=None, dpi=DPI) -> str:
    data = png(f, dpi=dpi)
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'
            f"{cap}</figure>")


def stretch(img, lo=1.0, hi=99.7):
    a, b = np.percentile(img, [lo, hi])
    return float(a), float(max(b, a + 1e-9))


def gauss_fwhm_fit(x, y):
    """Gaussian (the pipeline's own model) fit; returns popt or None."""
    peak = int(np.argmax(y))
    p0 = [y[peak] - np.median(y), float(x[peak]), 3.0, float(np.median(y))]
    try:
        popt, _ = curve_fit(pipeline._gaussian, x, y, p0=p0, maxfev=10000)
        return popt
    except (RuntimeError, ValueError):
        return None


def smooth(y, k=9):
    return np.convolve(y, np.ones(k) / k, mode="same")


def fwhm(x, y):
    """Full width at half maximum of a single-peaked curve (linear interp)."""
    y = np.asarray(y, float)
    i = int(np.nanargmax(y))
    half = y[i] / 2
    lo = i
    while lo > 0 and y[lo] > half:
        lo -= 1
    hi = i
    while hi < y.size - 1 and y[hi] > half:
        hi += 1
    xl = np.interp(half, [y[lo], y[lo + 1]], [x[lo], x[lo + 1]]) if lo < i else x[lo]
    xh = np.interp(half, [y[hi], y[hi - 1]], [x[hi], x[hi - 1]]) if hi > i else x[hi]
    return abs(xh - xl)


def src(obj) -> str:
    """``path:line name`` for a library object (for the sources line)."""
    try:
        path = inspect.getsourcefile(obj)
        line = inspect.getsourcelines(obj)[1]
        rel = os.path.relpath(path, REPO)
        return f"{rel}:{line} {obj.__qualname__}"
    except (TypeError, OSError):
        return getattr(obj, "__qualname__", str(obj))


def git_rev(path) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- chapters

class Story:
    def __init__(self):
        self.chapters: list[dict] = []
        self.sources: list[str] = []

    def add(self, title, paras, figs=(), box=None, skipped=None):
        self.chapters.append(dict(title=title, paras=list(paras), figs=list(figs),
                                  box=box, skipped=skipped))

    def cite(self, *objs):
        for o in objs:
            s = src(o) if not isinstance(o, str) else o
            if s not in self.sources:
                self.sources.append(s)


def box(setting, function, watch):
    return dict(setting=setting, function=function, watch=watch)


def fmt(x, nd=1):
    return f"{x:,.{nd}f}"


# ---------------------------------------------------------------- data helpers

def energy_of(path):
    rec = parse_sif_name(path)
    return rec.energy if rec is not None else pipeline._energy_from_name(path)


def rixs_measurement(directory):
    idx = ta.index_beamtime(directory)
    rixs = idx.by_kind("RIXS")
    return (rixs[0] if rixs else None), idx


def dark_has_signal(dark_img, limit=5.0):
    """(lit, excess): does a 'dark' contain X-ray signal?

    A true dark is flat at the pedestal. The largest 15-column running mean of
    (dark - its pedestal) above ``limit`` ADU per pixel means the shutter was
    open: the dark holds the emission line."""
    prof = (dark_img - common_mode(dark_img)).mean(axis=0)
    excess = float(np.max(smooth(prof, 15)))
    return excess > limit, excess


def elastic_banana(edir, step=5):
    """Measure the sideways bend of the lowest-energy elastic line.

    Sums the photon events of every ``step``-th frame, finds the line's column
    centroid in 32-row bands, fits a parabola column(row), and compares the
    row-summed spectrum plain, with the library's curvature correction, and with
    each row shifted sideways by the fit (an illustration, not a library call).
    """
    paths = sorted(glob.glob(os.path.join(edir, "*.sif")))
    recs = [parse_sif_name(p) for p in paths]
    data = sorted((r for r in recs if r and r.is_elastic and not r.is_dark),
                  key=lambda r: r.energy or 0)
    darks = {round(r.energy, 2): r.path for r in recs if r and r.is_dark and r.energy}
    if not data:
        return None
    rec = data[0]
    sif = ta.SifFile(rec.path)
    dk = (ta.SifFile(darks[round(rec.energy, 2)]).data.mean(axis=0)
          if round(rec.energy, 2) in darks else np.zeros(sif.shape))
    dcm = background_common_mode(dk)
    th = ta.Thresholds.from_input(None)
    img = np.zeros(sif.shape)
    for i in range(0, sif.num_frames, step):
        img += reduce_frame(sif.frame(i), dk, dcm, th)[0]
    del sif
    gc.collect()
    colsum = img.sum(axis=0)
    pk = int(np.argmax(smooth(colsum, 5)))
    w0, w1 = max(0, pk - 60), min(2048, pk + 60)
    x = np.arange(2048)
    band_r, band_c = [], []
    for r0 in range(0, 512, 32):
        sp = img[r0:r0 + 32, w0:w1].sum(axis=0)
        if sp.sum() > 0:
            band_r.append(r0 + 15.5)
            band_c.append(float((sp * x[w0:w1]).sum() / sp.sum()))
    coef = np.polyfit(band_r, band_c, 2)
    rows = np.arange(512)
    fit_cols = np.polyval(coef, rows)
    shift = fit_cols - fit_cols[256]
    shifted = np.empty_like(img)
    for r in rows:
        shifted[r] = np.interp(x + shift[r], x, img[r], left=0, right=0)
    lib = ta.CurvatureCorrection().fit_apply(img)[0]
    out = dict(path=rec.path, step=step, img=img, window=(max(0, pk - 90), min(2048, pk + 90)),
               band_r=band_r, band_c=band_c, fit_cols=fit_cols,
               travel=float(fit_cols.max() - fit_cols.min()),
               spec_plain=colsum, spec_lib=lib.sum(axis=0), spec_shift=shifted.sum(axis=0))
    for k in ("plain", "lib", "shift"):
        out["fwhm_" + k] = fwhm(x.astype(float), smooth(out["spec_" + k], 3))
    out["fwhm_band"] = float(np.median([fwhm(x.astype(float), smooth(img[r0:r0 + 64].sum(axis=0), 3))
                                        for r0 in range(0, 512, 64)]))
    return out


def dirs_with_sif(path):
    return bool(path) and os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*.sif")))


# ================================================================ main build

def build(bundled, real, out):
    t_start = time.time()
    S = Story()
    th = ta.Thresholds.from_input(None)
    facts = {}

    # ------------------------------------------------------------ bundled RIXS
    m1, idx1 = rixs_measurement(bundled)
    if m1 is None:
        raise SystemExit(f"no RIXS measurement found in {bundled}")
    dark_path = m1.dark_paths[0]
    dark = ta.SifFile(dark_path)
    dark_img = dark.data.mean(axis=0)
    dark_lit, dark_excess = dark_has_signal(dark_img)
    # Min-projection background from the series itself (portal "dark = none").
    minproj = ta.compute_background([ta.SifFile(p) for p in m1.data_paths])
    # A "dark" that contains the emission line would be subtracted from every
    # image; use the frames' own background instead (see chapter 3).
    bcg = minproj if dark_lit else dark_img
    bcg_cm = background_common_mode(bcg)
    log(f"bundled: {m1!r}; dark lit={dark_lit} (excess {dark_excess:.1f} ADU)")

    # One pass over the series: pedestal, I0, scan-wide histograms, live HERFD.
    acc = live.HerfdAccumulator(dark=bcg, n=ROI_N)
    hist = {k: np.zeros(NBINS, dtype=np.int64)
            for k in ("xray", "bkg_free", "binned", "raw", "background")}
    ped, i0s, energies, cosmic_px, grains_kept, grains_all = [], [], [], 0, 0, 0
    for p in m1.data_paths:
        sif = ta.SifFile(p)
        im = sif.frame(0)
        cm = common_mode(im)
        ped.append(cm)
        i0s.append(sif.I0)
        energies.append(sif.mono)
        _, masks = reduce_frame(im, bcg, bcg_cm, th, cm=cm, hist=hist)
        cosmic_px += int(masks["cosmic"].sum())
        grains_kept += int(label(masks["grains"], structure=_CONN4)[1])
        grains_all += int(label(masks["events"], structure=_CONN4)[1])
        acc.add(p)
    ped, i0s, energies = map(np.asarray, (ped, i0s, energies))
    r_live = acc.result()
    cc_scan = acc.curvature()
    t_scan = np.asarray(cc_scan.t, dtype=float)
    log("bundled series accumulated")

    # The batch pipeline, as the portal runs it (Measurement -> OnePotRIXS).
    op = m1.pipeline(use_dark_as_background=not dark_lit)
    xres = op.run()                               # XESResult, scan mode
    batch_map = xres.scan_data                    # (pixel, file), not I0-divided
    live_map, _, _ = acc.rixs_map()
    batch_live_diff = float(np.max(np.abs(batch_map - live_map)) / np.max(batch_map))
    signal_sum = xres.signal.sum(axis=0)
    corr_sum = xres.corr_signal
    del xres, op
    gc.collect()

    r1 = r_live
    cp = r1.central_pix
    wl_i = int(np.argmax(r1.HERFD))
    E_wl = float(r1.E[wl_i])
    by_e = {round(energy_of(p), 2): p for p in m1.data_paths}
    wl_path = by_e[round(E_wl, 2)]
    pre_path = by_e[min(by_e)]
    wl_sif = ta.SifFile(wl_path)
    wl_img = wl_sif.frame(0)
    wl_cm = common_mode(wl_img)
    sub, bcg_adj = subtract_pedestal(wl_img, bcg, bcg_cm, cm=wl_cm)
    sig, masks = reduce_frame(wl_img, bcg, bcg_cm, th, cm=wl_cm)

    # ================================================================ 1 raw frame
    lo, hi = stretch(wl_img, 0.5, 99.8)
    f = fig(10, 5.6)
    a0, a1 = f.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    im = a0.imshow(wl_img, cmap="gray", aspect="auto", vmin=lo, vmax=hi,
                   interpolation="antialiased",
                   extent=(-0.5, wl_img.shape[1] - 0.5, wl_img.shape[0] - 0.5, -0.5))
    cb = f.colorbar(im, ax=a0, pad=0.01)
    cb.set_label("ADU")
    a0.set_ylabel("detector row\n(spatial)")
    a0.set_title(f"One raw frame: {os.path.basename(wl_path)}", color=INK)
    colsum = wl_img.sum(axis=0)
    a1.plot(colsum / 1e3, color=SERIES[0], lw=1)
    a1.axhline(512 * wl_cm / 1e3, color=INK_2, ls="--", lw=1)
    a1.annotate(f"512 rows × pedestal {wl_cm:.0f} ADU", xy=(20, 512 * wl_cm / 1e3),
                xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=9)
    a1.set_ylabel("column sum\n(thousand ADU)")
    a1.set_xlabel("detector column = dispersive pixel (emission energy)")
    style(a1)
    fig1 = img_tag(f, "raw detector frame",
                   "Top: the raw frame, grey = ADU. Bottom: summing each column of the raw "
                   "frame. The flat level is just the electronic pedestal; the photons are "
                   "the small bump on top of it.")
    meta = wl_sif.metadata
    facts["raw_bump_pct"] = 100 * (colsum.max() - np.median(colsum)) / np.median(colsum)
    metalist = (
        f"<ul class='kv'><li><b>Shape</b> {wl_sif.num_frames} frame × "
        f"{wl_sif.shape[0]} rows × {wl_sif.shape[1]} columns</li>"
        f"<li><b>Monochromator (incident) energy</b> {meta['mono']:.2f} eV</li>"
        f"<li><b>I0</b> (incident-beam monitor) {meta['I0']:,.0f} counts</li>"
        f"<li><b>I1</b> {meta['I1']:,.0f} counts</li>"
        f"<li><b>Exposure</b> {wl_sif.exposure_time:g} s per frame</li></ul>")
    S.add("1. The detector and one raw frame", [
        "The spectrometer's Andor CCD records a 512 × 2048 pixel image. The analyser "
        "crystal spreads emitted X-rays by energy along the <b>2048 columns</b>, so a "
        "column number is (after calibration, chapter 12) an emission energy. The 512 "
        "rows are the spatial direction; they are summed away later.",
        f"This frame is the scan point with the strongest signal (incident energy "
        f"{E_wl:.2f} eV, the sulfur white line). What it carries in its header:",
        metalist,
        f"A raw frame is mostly electronic offset. Even at the brightest scan point the "
        f"photons raise the column sum by only about {facts['raw_bump_pct']:.0f} % above "
        "the pedestal, which is why none of the later steps work on raw sums.",
    ], [fig1], box(
        "None: this is the input. Frames are read as (frames, 512, 2048).",
        "<code>SifFile</code> (frames via sif_parser; mono, I0, I1 parsed from the "
        "header comment).",
        "A header with no <code>mono</code> line (the energy then falls back to the "
        "filename); a file that is still being written (short or truncated); the "
        "detector clock in the header can be wrong (this one says 2009), so do not "
        "trust header dates."))
    S.cite(ta.SifFile)

    # ================================================================ 2 pedestal
    h = adu_histogram(wl_img).astype(float)
    x = np.arange(h.size)
    win = slice(int(wl_cm) - 15, int(wl_cm) + 16)
    popt = gauss_fwhm_fit(x[win].astype(float), h[win])
    sigma = abs(popt[2]) / np.sqrt(2) if popt is not None else float("nan")
    near = np.abs(wl_img - wl_cm) <= 5 * sigma
    facts["ped_sigma"] = sigma
    f = fig(10, 3.8)
    a0, a1 = f.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
    a0.bar(x[win], h[win], width=1, color=SERIES[0], alpha=0.85)
    if popt is not None:
        xx = np.linspace(x[win][0], x[win][-1], 300)
        a0.plot(xx, pipeline._gaussian(xx, *popt), color=INK, lw=1.2)
    a0.axvline(wl_cm, color=INK_2, ls="--", lw=1)
    a0.set_title(f"pedestal peak: centre {wl_cm:.1f}, σ {sigma:.1f} ADU", color=INK)
    a0.set_xlabel("pixel value (ADU)")
    a0.set_ylabel("number of pixels")
    style(a0)
    a1.step(x, np.where(h > 0, h, np.nan), where="mid", color=SERIES[0], lw=1)
    a1.set_yscale("log")
    a1.set_xlim(wl_cm - 40, wl_cm + 700)
    a1.axvline(wl_cm, color=INK_2, ls="--", lw=1)
    a1.annotate("pedestal", xy=(wl_cm, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -14), textcoords="offset points", color=INK_2, fontsize=9)
    a1.annotate("photon tail", xy=(wl_cm + 250, h[int(wl_cm) + 250] + 1), xytext=(20, 25),
                textcoords="offset points", color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    a1.set_title("same histogram, log scale, wider range", color=INK)
    a1.set_xlabel("pixel value (ADU)")
    a1.set_ylabel("number of pixels (log)")
    style(a1)
    fig2a = img_tag(f, "pedestal histogram",
                    "Histogram of every pixel of the frame in chapter 1. Left: the "
                    "pedestal peak with a gaussian fit. Right: the long tail above it is "
                    "where the photons are.")
    f = fig(10, 2.8)
    ax = f.subplots()
    order = np.argsort(energies)
    ax.plot(energies[order], ped[order], "o-", color=SERIES[0], ms=3, lw=1)
    ax.set_xlabel("incident energy (eV)")
    ax.set_ylabel("pedestal (ADU)")
    ax.set_title("pedestal of every frame in the scan", color=INK)
    style(ax)
    fig2b = img_tag(f, "pedestal over the scan",
                    f"The pedestal of each of the {len(ped)} frames. Its spread "
                    f"(max − min) is {ped.max() - ped.min():.2f} ADU.")
    S.add("2. The pedestal (common mode)", [
        f"Histogram all 1,048,576 pixels of the frame and one peak dominates: the "
        f"<b>pedestal</b>, the electronic zero of the read-out, here at {wl_cm:.1f} ADU "
        f"with a width (read-out noise) of σ ≈ {sigma:.1f} ADU. "
        f"{100 * near.mean():.1f} % of pixels are within 5σ of it, even in this bright "
        "frame. Photons appear as the tail to the right.",
        "The library calls the refined position of this peak the <b>common mode</b>. It "
        "is measured for every frame (a parabola through the tallest histogram bins) and "
        "is used to scale the dark in the next step.",
        f"Across the scan the pedestal is steady to {ped.max() - ped.min():.2f} ADU, so "
        "for this data set the scaling is a small correction.",
    ], [fig2a, fig2b], box(
        "<code>bcg_adjust</code> (default <b>on</b>) decides whether the common mode is "
        "used to scale the dark.",
        "<code>common.common_mode</code>, <code>common.adu_histogram</code>.",
        "A pedestal that jumps between frames (detector temperature, read-out mode "
        "change); a double-peaked pedestal; a histogram with no tail (no photons: "
        "shutter closed or beam lost)."))
    S.cite(common_mode, adu_histogram)

    # ================================================================ 3 dark
    ok = ~masks["events"]
    mad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))  # noqa: E731
    dcm = background_common_mode(dark_img)
    sub_dark, adj_dark = subtract_pedestal(wl_img, dark_img, dcm, cm=wl_cm)
    resid_dark = sub_dark[ok]
    resid_bg = sub[ok]
    raw_noise = mad(wl_img[ok])
    sub_noise = mad(resid_bg)
    facts.update(bcg_adj=bcg_adj, sub_noise=sub_noise, raw_noise=raw_noise,
                 dark_excess=dark_excess, dark_lit=dark_lit)
    # HERFD with each background choice, same ROI, for the consequence panel
    alt = {}
    flat_bg = np.full(dark_img.shape, dcm)
    choices = [("the paired 'dark' (default)", dark_img),
               ("frames' min-projection ('dark = none')", minproj),
               ("flat pedestal only", flat_bg)]
    for name, bgi in choices:
        a_ = live.HerfdAccumulator(dark=bgi, n=ROI_N)
        for p in m1.data_paths:
            a_.add(p)
        a_.set_roi(acc.result().central_pix, ROI_N)
        alt[name] = a_.result()
        del a_
    gc.collect()
    alt_norm = {k: export.normalize_mu(v.E, v.HERFD, NORM) for k, v in alt.items()}
    k_d, k_m, k_f = list(alt)
    loss_wl = 1 - np.max(alt[k_d].HERFD) / np.max(alt[k_m].HERFD)
    post_m = alt[k_m].E > alt[k_m].E.max() - 15
    loss_post = 1 - alt[k_d].HERFD[post_m].mean() / alt[k_m].HERFD[post_m].mean()
    wl_d, wl_m = np.max(alt_norm[k_d].flat), np.max(alt_norm[k_m].flat)
    facts.update(dark_loss_wl=loss_wl, dark_loss_post=loss_post, wl_dark=wl_d, wl_minproj=wl_m)

    f = fig(10, 8.4)
    gs = f.add_gridspec(3, 2, height_ratios=[1.1, 1, 1.15])
    a0 = f.add_subplot(gs[0, :])
    dl, dh = stretch(dark_img, 0.5, 99.5)
    im = a0.imshow(dark_img, cmap="gray", aspect="auto", vmin=dl, vmax=dh,
                   interpolation="antialiased")
    f.colorbar(im, ax=a0, pad=0.01).set_label("ADU")
    a0.set_title(f"the 'dark': {os.path.basename(dark_path)} "
                 f"({dark.num_frames} frame, {dark.exposure_time:g} s)", color=INK)
    a0.set_xlabel("detector column")
    a0.set_ylabel("row")
    a1 = f.add_subplot(gs[1, 0])
    a1.plot(dark_img.mean(axis=0) - dcm, color=SERIES[1], lw=0.8, label="the 'dark'")
    a1.plot(minproj.mean(axis=0) - background_common_mode(minproj), color=SERIES[0],
            lw=1.2, ls="--", label="min-projection of the frames")
    a1.set_xlabel("detector column")
    a1.set_ylabel("column mean −\npedestal (ADU)")
    a1.legend(frameon=False, loc="upper left", fontsize=9)
    style(a1)
    a2 = f.add_subplot(gs[1, 1])
    bins = np.arange(-60, 61)
    a2.hist(resid_dark.ravel(), bins=bins, histtype="step", color=SERIES[1], lw=1.4,
            label="frame − 'dark'")
    a2.hist(resid_bg.ravel(), bins=bins, histtype="step", color=SERIES[0], lw=1.4,
            ls="--", label="frame − min-projection")
    a2.set_yscale("log")
    a2.set_xlabel("ADU after subtraction, photon-free pixels")
    a2.set_ylabel("pixels (log)")
    a2.legend(frameon=False, fontsize=9, loc="upper right")
    style(a2)
    a3 = f.add_subplot(gs[2, :])
    for i, (k, v) in enumerate(alt_norm.items()):
        a3.plot(alt[k].E, v.flat, color=SERIES[[1, 0, 2][i]], lw=1.6, ls=LINESTYLES[i],
                label=f"{k}: white line {np.max(v.flat):.2f}")
    a3.set_xlabel("incident energy (eV)")
    a3.set_ylabel("normalised HERFD")
    a3.legend(frameon=False, fontsize=9)
    style(a3)
    fig3 = img_tag(f, "dark frame and subtraction",
                   "Top: the file named as this series' dark. Middle left: its column "
                   "profile against a background built from the frames. Middle right: "
                   "photon-free pixels after each subtraction. Bottom: the normalised "
                   "HERFD of the whole scan with each choice of background (normalised as "
                   "in chapter 10).")
    paras3 = [
        "Each series has a <b>dark</b> (<code>..._dark.sif</code>), meant to be a frame "
        "taken with the shutter closed: the pedestal plus the detector's fixed pattern. "
        "The indexer pairs it with its series by name, and by default (portal "
        "<code>dark = auto</code>) it is subtracted from every image.",
        f"Before subtracting, the background is scaled so its pedestal matches the "
        f"frame's: <i>frame − (common mode of frame ÷ common mode of background) × "
        f"background</i>. Here the factor is {bcg_adj:.5f}, a shift of "
        f"{(bcg_adj - 1) * bcg_cm:+.2f} ADU"
        + (": the min-projection background (below) is the LOWEST of many noisy "
           "readings of each pixel, so it sits a few σ under the true pedestal, and the "
           "scaling lifts it back." if dark_lit else ": a small correction for this data."),
    ]
    if dark_lit:
        paras3 += [
            f"<b>This series' 'dark' is not dark.</b> It shows the same emission band as "
            f"the data (up to {dark_excess:.0f} ADU per pixel above the pedestal, "
            f"averaged down a column), and its header gives mono = {dark.mono:g} eV and "
            f"I0 = {dark.I0:,.0f}, the same as a data frame. It was taken with the beam "
            "on the sample, at an energy above the edge.",
            0,
            f"Subtracting it removes real photons from every image. With the default "
            f"background the white line loses {100 * loss_wl:.0f} % of its counts and the "
            f"end of the scan {100 * loss_post:.0f} %. The loss is not proportional, so "
            f"the <i>shape</i> changes: the normalised white line comes out at "
            f"{wl_d:.2f} edge steps instead of {wl_m:.2f}. The background built from the "
            "frames themselves (the per-pixel minimum over the series) and a plain flat "
            "pedestal agree with each other, so <b>the rest of this story uses the "
            "min-projection background</b> for this series.",
        ]
        ag_darks = sorted(glob.glob(os.path.join(real or "", "*", "*AgL3val*_dark.sif")))
        if ag_darks:
            ag_ex = max(dark_has_signal(ta.SifFile(d).data.mean(axis=0))[1] for d in ag_darks[:4])
            paras3.append(
                f"The Ag XES darks of chapter 13 are real darks (flat to within "
                f"{ag_ex:.0f} ADU), so it is this series' acquisition, not the method, "
                "that is at fault.")
    paras3.append(
        f"Subtraction removes the offset, not the noise: photon-free pixels scatter by "
        f"σ ≈ {sub_noise:.1f} ADU afterwards against {raw_noise:.1f} ADU in the raw frame.")
    S.add("3. The dark frame and dark subtraction", paras3, [fig3], box(
        "Background: the paired dark (portal <code>dark = auto</code>, the default), or "
        "<code>dark = none</code>, which builds one from the frames (per-pixel minimum over "
        "the series). <code>bcg_adjust</code> (default on) scales it.",
        "<code>analyze.subtract_pedestal</code>, <code>background.compute_background</code>, "
        "<code>preview.resolve_background</code>; the dark is paired by "
        "<code>index_beamtime</code>.",
        "Look at the dark before trusting it: a real dark is flat. A dark with the "
        "emission line in it (as here) silently lowers and reshapes every spectrum, and "
        "nothing in the pipeline checks for it. Also: a dark with a different exposure "
        "than the data; for XES, all the darks of a measurement are averaged into one, "
        "which assumes the dark did not drift between scans."))
    S.cite(subtract_pedestal, ta.compute_background, ta.index_beamtime)

    # ================================================================ 4 thresholds
    xh = hist["xray"].astype(float)
    xs = smooth(xh, 9)
    photon_adu = int(150 + np.argmax(xs[150:1000]))
    facts["photon_adu"] = photon_adu
    curves = {"single pixel, dark-subtracted": hist["bkg_free"],
              "3×3 box sum": hist["binned"],
              "whole grain (cluster) total": hist["xray"]}
    f = plotting.adu_histogram(curves, thresholds=th, key=None)
    ax = f.axes[0]
    ax.set_xlim(0, 1200)
    ax.set_xlabel("ADU above the dark")
    ax.set_ylabel("count (log)")
    for ln, ls in zip(ax.get_lines(), LINESTYLES):
        ln.set_linestyle(ls)
    ax.legend(frameon=False, loc="upper right")
    ax.annotate(f"one S Kα photon ≈ {photon_adu} ADU", xy=(photon_adu, xh[photon_adu]),
                xytext=(40, 30), textcoords="offset points", color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    f.set_size_inches(10, 4.2)
    fig4a = img_tag(f, "threshold histogram",
                    f"Histograms summed over all {len(m1.data_paths)} frames of the scan, "
                    "with the four thresholds. The 'hi' line (2000) is off to the right.")

    # zoom panels
    c0 = int(np.clip(cp - 90, 0, 2048 - 180))
    rows = masks["events"][:, c0:c0 + 180].sum(axis=1)
    r0 = int(np.clip(np.argmax(smooth(rows.astype(float), 40)) - 30, 0, 512 - 60))
    zs = (slice(r0, r0 + 60), slice(c0, c0 + 180))
    rejected = masks["events"] & ~masks["grains"]
    f = fig(10, 5.8)
    axs = f.subplots(2, 2, sharex=True, sharey=True)
    ext = (c0 - 0.5, c0 + 179.5, r0 + 59.5, r0 - 0.5)
    vm = np.percentile(sub[zs], 99.5)
    panels = [
        (wl_img[zs], "raw", dict(vmin=np.percentile(wl_img[zs], 1),
                                 vmax=np.percentile(wl_img[zs], 99.5))),
        (sub[zs], "after dark subtraction", dict(vmin=-3 * sigma, vmax=vm)),
        (masks["events"][zs].astype(float) + rejected[zs], "event mask: kept (grey), rejected (black)",
         dict(vmin=0, vmax=2, cmap="Greys")),
        (sig[zs], "final signal: only accepted photon grains", dict(vmin=0, vmax=vm)),
    ]
    for axi, (data, ttl, kw) in zip(axs.ravel(), panels):
        axi.imshow(data, aspect="auto", interpolation="nearest", extent=ext,
                   cmap=kw.pop("cmap", "gray"), **kw)
        axi.set_title(ttl, color=INK)
        axi.tick_params(colors=INK_2, labelcolor=INK_2)
    axs[1, 0].set_xlabel("detector column")
    axs[1, 1].set_xlabel("detector column")
    axs[0, 0].set_ylabel("row")
    axs[1, 0].set_ylabel("row")
    fig4b = img_tag(f, "before and after photon detection",
                    "A 60 × 180 pixel patch of the bright frame near the ROI. After "
                    "dark subtraction the photons stand out as small clusters; the "
                    "3×3 test keeps them, the grain gate drops faint ones.")

    # sensitivity to low / xray
    band_cols = slice(cp - ROI_N // 2, cp - ROI_N // 2 + ROI_N)
    ref = sig.sum()
    lows = [40, 60, 80, 100, 130, 160, 200]
    xrays = [100, 130, 170, 200, 250, 300]
    k_low = [reduce_frame(wl_img, bcg, bcg_cm, [lv, th.xray, th.hi], cm=wl_cm)[0].sum() / ref
             for lv in lows]
    k_x = [reduce_frame(wl_img, bcg, bcg_cm, [th.low, xv, th.hi], cm=wl_cm)[0].sum() / ref
           for xv in xrays]
    pre_img = ta.SifFile(pre_path).frame(0)
    pre_cm = common_mode(pre_img)
    pre_low = [reduce_frame(pre_img, bcg, bcg_cm, [lv, th.xray, th.hi], cm=pre_cm)[0].sum()
               for lv in lows]
    f = fig(10, 3.4)
    a0, a1 = f.subplots(1, 2, sharey=True)
    a0.plot(lows, k_low, "o-", color=SERIES[0], ms=5, label=f"brightest frame ({E_wl:.1f} eV)")
    pre_ref = pre_low[lows.index(int(th.low))] or 1.0
    a0.plot(lows, np.array(pre_low) / pre_ref, "s--", color=SERIES[1], ms=5,
            label=f"weakest frame ({energy_of(pre_path):.1f} eV)")
    a0.legend(frameon=False, fontsize=9)
    a0.axvline(th.low, color=INK_2, ls="--", lw=1)
    a0.set_xlabel("low (ADU), xray fixed at 170")
    a0.set_ylabel("signal kept, relative to defaults")
    style(a0)
    a1.plot(xrays, k_x, "o-", color=SERIES[0], ms=5)
    a1.axvline(th.xray, color=INK_2, ls="--", lw=1)
    a1.axvline(photon_adu, color=INK_2, ls=":", lw=1)
    a1.annotate(" 1 photon", xy=(photon_adu, 0.95), xycoords=("data", "axes fraction"),
                color=INK_2, fontsize=9)
    a1.set_xlabel("xray (ADU), low fixed at 100")
    style(a1)
    fig4c = img_tag(f, "threshold sensitivity",
                    "How much of the bright frame's signal survives as each threshold "
                    "is moved (1.0 = the defaults).")
    n_rej = grains_all - grains_kept
    S.add("4. Thresholds: cosmic rays, photon events, grains", [
        f"After dark subtraction, a pixel with a photon sits well above the noise. One "
        f"sulfur Kα photon (about 2.3 keV) deposits roughly <b>{photon_adu} ADU</b>, "
        f"spread over a few neighbouring pixels (a 'grain'). The thresholds "
        f"<code>[bcg_cutoff, low, xray, hi]</code> = <b>[60, 100, 170, 2000]</b> pick "
        "the photons out, in three tests per frame:",
        "<ol><li><b>hi = 2000</b>: any pixel above it is a cosmic ray or a hot pixel and "
        "is zeroed.</li>"
        "<li><b>low = 100</b>: a pixel is a photon candidate if it is above low/4 = 25 ADU "
        "and the 3×3 box around it sums to more than 100 ADU. Candidates are then grown "
        "by one pixel so the whole cluster is kept.</li>"
        "<li><b>xray = 170</b>: each connected cluster is summed; clusters below 170 ADU "
        "are thrown away as noise or partial events.</li></ol>",
        0,
        f"Over the whole scan, {grains_all:,} candidate clusters were found and "
        f"{n_rej:,} ({100 * n_rej / max(grains_all, 1):.0f} %) were rejected by the "
        f"xray gate; {cosmic_px} pixel{'s' if cosmic_px != 1 else ''} {'were' if cosmic_px != 1 else 'was'} over 'hi'. What is kept is the ADU sum of "
        "the accepted clusters, not a photon count, so several photons that merge into "
        "one cluster at bright points are still counted in full.",
        "<b>bcg_cutoff = 60 is not used in a normal run.</b> It is only used by the "
        "optional two-pass (<code>evolution=True</code>) mode, which first extracts with "
        "that single threshold to fit the curvature. The portal, the live tab and the "
        "first pass all use low, xray and hi.",
        1,
        f"The bottom figure shows the defaults sit on a plateau: in the brightest "
        f"frame, moving <code>low</code> from 60 to 160 changes the kept signal by "
        f"{100 * (max(k_low[1:6]) - min(k_low[1:6])):.1f} %; in the weakest frame, "
        f"which is mostly noise, by {100 * (max(pre_low[1:6]) - min(pre_low[1:6])) / pre_ref:.0f} %, "
        "so <code>low</code> mainly decides how much noise gets in where there is little "
        "signal. Raising <code>xray</code> towards one photon's worth starts to eat real "
        "photons.",
    ], [fig4a, fig4b, fig4c], box(
        "<code>threshold</code> = [bcg_cutoff, low, xray, hi], default "
        "<b>[60, 100, 170, 2000]</b> ADU (portal field 'ADU thresholds').",
        "<code>analyze.reduce_frame</code> (one frame), <code>analyze.extract_signal</code> "
        "(many), <code>pipeline.Thresholds</code>.",
        "The thresholds are in ADU, so they depend on the photon energy: an Ag L photon "
        "(3.3 keV) gives more ADU than an S Kα photon, and the defaults are further "
        "from the photon peak. Watch the grain histogram: its single-photon peak must "
        "sit clearly above xray. Too low a <code>low</code> lets noise clusters in "
        "(a flat rise across all columns); too high an xray loses photons."))
    S.cite(reduce_frame, ta.extract_signal, ta.Thresholds)

    # ================================================================ 5 curvature
    cols = np.arange(1, 2049)
    offsets = np.round(np.polyval(t_scan, cols)).astype(int)
    spec_before = signal_sum.sum(axis=0)
    spec_after = corr_sum.sum(axis=0)
    lost = 1 - spec_after.sum() / spec_before.sum()
    acc.curvature_t = 1
    acc.set_roi(cp, ROI_N)
    r_flat_same = acc.result()
    acc.curvature_t = None
    acc.set_roi(None, ROI_N)
    dh = np.abs(r_flat_same.HERFD - r1.HERFD) / np.max(r1.HERFD)
    facts.update(curv_shift=(int(offsets.min()), int(offsets.max())), curv_lost=lost,
                 curv_herfd=float(dh.max()))
    res_ns = SimpleNamespace(signal=signal_sum, corr_signal=corr_sum, t=t_scan)
    f = plotting.curvature_before_after(res_ns)
    f.set_size_inches(10, 6)
    fig5a = img_tag(f, "curvature before and after",
                    "The library's correction on this scan: summed photon signal before "
                    "and after, and the row profile of each.")
    pct = lambda v: ("none" if v == 0 else "less than 0.001 %" if v < 1e-5  # noqa: E731
                     else f"{100 * v:.3f} %")
    paras5 = [
        "An analyser crystal does not image a perfectly straight line onto the detector; "
        "the line is bent (a 'banana'). The library's correction measures a drift of the "
        "signal <i>along the rows</i> as a function of column, by cross-correlating the "
        "row profile of each group of columns with the overall profile, fits a parabola, "
        "and shifts every column up or down by that amount. For a RIXS scan it fits one "
        "curve on the sum of all scan points and applies it to every point.",
        0,
        f"For this scan the fitted shift runs from {offsets.min()} to {offsets.max()} "
        f"rows across the detector.",
        f"<b>How much it matters for the 1D result: essentially nothing, by "
        f"construction.</b> The shift moves pixels up or down <i>within</i> a column, "
        f"and a column's sum does not care where in the column a count sits. The only "
        f"possible change is counts pushed off the top or bottom edge: "
        f"{pct(lost)} of the total here. "
        + ("The HERFD with and without the correction is identical."
           if float(dh.max()) == 0 else
           f"The HERFD with and without the correction differs by at most "
           f"{pct(float(dh.max()))} of its peak."),
    ]
    figs5 = [fig5a]
    real = real or ""
    edir = os.path.join(real, "elastic_BN")
    ban = elastic_banana(edir) if dirs_with_sif(edir) else None
    if ban is not None:
        facts.update(banana_px=ban["travel"], fwhm_plain=ban["fwhm_plain"],
                     fwhm_lib=ban["fwhm_lib"], fwhm_shift=ban["fwhm_shift"])
        f = fig(10, 6.4)
        gs = f.add_gridspec(2, 2, width_ratios=[1, 1.5])
        a0 = f.add_subplot(gs[:, 0])
        c0, c1 = ban["window"]
        vmax = np.percentile(ban["img"][:, c0:c1], 99.7)
        a0.imshow(ban["img"][:, c0:c1], cmap="gray", aspect="auto", vmin=0, vmax=vmax,
                  extent=(c0 - 0.5, c1 - 0.5, 511.5, -0.5), interpolation="antialiased")
        a0.plot(ban["fit_cols"], np.arange(512), color=SERIES[1], lw=1.5, ls="--",
                label="fitted line position")
        a0.plot(ban["band_c"], ban["band_r"], "o", color=SERIES[1], ms=5)
        a0.set_xlabel("detector column")
        a0.set_ylabel("detector row")
        a0.set_title("elastic line, photon events", color=INK)
        a0.legend(frameon=False, loc="lower left", fontsize=9, labelcolor="white")
        a1 = f.add_subplot(gs[:, 1])
        xx = np.arange(2048)
        for i, (k, lab) in enumerate([("spec_plain", "no correction"),
                                      ("spec_lib", "library correction (row shifts)"),
                                      ("spec_shift", "rows shifted sideways (not in the library)")]):
            y = ban[k]
            a1.plot(xx, y / ban["spec_plain"].max(), color=SERIES[i], lw=1.6, ls=LINESTYLES[i],
                    label=f"{lab}: FWHM {ban['fwhm_' + k.split('_')[1]]:.1f} px")
        a1.set_xlim(c0 + 20, c1 - 20)
        a1.set_xlabel("dispersive pixel")
        a1.set_ylabel("elastic spectrum (peak of 'no correction' = 1)")
        a1.legend(frameon=False, fontsize=9, loc="upper left")
        style(a1)
        figs5.append(img_tag(
            f, "elastic banana",
            f"The real banana, seen in the elastic line of chapter 12 "
            f"({os.path.basename(ban['path'])}, every {ban['step']}th frame). Left: the line "
            f"drifts sideways by {ban['travel']:.0f} columns from top to bottom. Right: the "
            "summed spectrum without correction, with the library's correction, and with "
            "each row shifted sideways by the fitted amount."))
        paras5 += [
            "<b>But the banana that matters bends the other way.</b> A sharp line shows it "
            f"best: in the elastic scans of chapter 12 the line's column drifts by "
            f"{ban['travel']:.0f} columns from the top of the detector to the bottom, a "
            "smooth curve. Summing the rows then smears the line sideways, in the "
            "energy direction.",
            1,
            f"Row by row the elastic line is {ban['fwhm_band']:.0f} columns wide; summed "
            f"over all rows it is {ban['fwhm_plain']:.0f}. The library's correction, "
            f"which only moves pixels within columns, leaves it at "
            f"{ban['fwhm_lib']:.0f}. Shifting each row <i>sideways</i> by the fitted "
            f"amount (done here for illustration; it is not a library function) "
            f"narrows it to {ban['fwhm_shift']:.0f} columns. So the correction as "
            "implemented cannot improve resolution, while a sideways one could make lines "
            f"about {ban['fwhm_plain'] / ban['fwhm_shift']:.1f}× sharper. This is worth "
            "checking against the original MATLAB, which transposed the image twice: "
            "the orientation may have been lost in the port.",
        ]
    S.add("5. Curvature (\"banana\") correction", paras5, figs5, box(
        "<code>curvature_t</code>: fitted from the data (default), fixed coefficients, "
        "or 1 for none. Not exposed in the portal form.",
        "<code>curvature.CurvatureCorrection</code> (<code>fit</code>, <code>apply</code>).",
        "The correction only moves counts within columns, so it cannot change a "
        "column-sum spectrum or a HERFD; it cannot fix a line that bends across columns. "
        "Look at a sharp (elastic) line image: if it is not vertical, the spectrum is "
        "broadened by that much."))
    S.cite(ta.CurvatureCorrection)

    # ================================================================ 6 spectrum
    pv = preview(wl_path, dark=bcg, curvature_t=t_scan, downsample=(4, 2))
    f = plotting.image_with_spectrum(pv, central_pix=cp, n=ROI_N, image="events",
                                     title=f"{os.path.basename(wl_path)}: photon events "
                                           "over their column sum")
    f.set_size_inches(10, 6)
    f.axes[1].set_ylabel("counts (ADU per column)")
    f.axes[1].set_xlabel("dispersive pixel (detector column)")
    fig6 = img_tag(f, "image over spectrum",
                   "Top: the photon-event image (downsampled 4×2 for display). Bottom: its "
                   "emission spectrum at full resolution. The x axis is shared: a feature "
                   "in the spectrum sits directly under the column that makes it. The grey "
                   "band is the ROI of chapter 7.")
    spec = pv["spectrum"]
    sp_s = smooth(spec, 15)
    facts["spec_peak_col"] = int(np.argmax(sp_s))
    facts["spec_fwhm_px"] = fwhm(np.arange(spec.size), sp_s)
    S.add("6. The emission spectrum is the column sum", [
        "Summing each cleaned column over its 512 rows gives one number per column: the "
        "<b>emission spectrum</b> of this image, intensity against dispersive pixel. Below, "
        "the image and its spectrum share the same x axis, so you can see which columns "
        "make which part of the spectrum.",
        f"At this incident energy the emission is a band about "
        f"{facts['spec_fwhm_px']:.0f} pixels wide at half height with two peaks "
        "(chapter 8 shows why that matters). The "
        "column-to-column spikiness is photon counting noise: each column has only a "
        "few hundred photons in a 1-s frame.",
    ], [fig6], box(
        "None beyond chapters 3 to 5; the spectrum follows from them.",
        "<code>preview.preview</code> (one image), <code>XESResult.spectrum</code> "
        "(a whole measurement); figure <code>plotting.image_with_spectrum</code>.",
        "Structure that is the same in every image regardless of incident energy (hot "
        "columns, a bad dark); a spectrum that runs off the edge of the detector (the "
        "spectrometer was set for a different line)."))
    S.cite(preview, plotting.image_with_spectrum)

    # ================================================================ 7 ROI
    band_sum = float(spec[band_cols].sum())
    point = band_sum / wl_sif.I0
    live_point = float(r1.HERFD[wl_i])
    ns = [1, 3, 7, 15, 31, 61]
    pts = [spec[cp - n // 2: cp - n // 2 + n].sum() / wl_sif.I0 for n in ns]
    f = fig(10, 3.6)
    a0, a1 = f.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    xx = np.arange(cp - 150, cp + 151)
    a0.plot(xx, spec[xx], color=SERIES[0], lw=1, label="spectrum (this image)")
    a0.plot(xx, smooth(spec, 15)[xx], color=INK, lw=1.4, ls="--", label="15-column running mean")
    lo_b, hi_b = cp - ROI_N // 2 - 0.5, cp - ROI_N // 2 + ROI_N - 0.5
    a0.axvspan(lo_b, hi_b, color=plotting.ROI, alpha=0.25, lw=0)
    a0.annotate(f"ROI {cp} ± {ROI_N // 2}\nsum {band_sum:,.0f} ADU", xy=(hi_b, 0.95),
                xycoords=("data", "axes fraction"), xytext=(6, 0),
                textcoords="offset points", va="top", fontsize=9, color=INK)
    a0.set_xlabel("dispersive pixel")
    a0.set_ylabel("counts (ADU)")
    a0.legend(frameon=False, loc="upper left", fontsize=9)
    style(a0)
    a1.plot(ns, np.array(pts) / np.array(ns), "o-", color=SERIES[0], ms=5)
    a1.set_xscale("log")
    a1.set_xticks(ns)
    a1.set_xticklabels([str(n) for n in ns])
    a1.set_xlabel("ROI width n (pixels)")
    a1.set_ylabel("HERFD point ÷ n\n(counts/I0 per pixel)")
    style(a1)
    fig7 = img_tag(f, "ROI band",
                   "Left: the spectrum near the ROI; the grey band is summed. Right: the "
                   "same image's point for different band widths, per pixel of width.")
    S.add("7. The ROI band gives one HERFD point", [
        f"HERFD (high-energy-resolution fluorescence detection) keeps only a narrow slice "
        f"of the emission. That slice is the <b>ROI band</b>: columns centre − n/2 to "
        f"centre + n/2. Here the centre is column {cp} (found automatically, chapter 8) "
        f"and n = {ROI_N}.",
        f"For this image: the band sums to {band_sum:,.0f} ADU; divided by I0 "
        f"({wl_sif.I0:,.0f}) that is <b>{point:.4g}</b>, which is exactly the HERFD value "
        f"the scan has at {E_wl:.2f} eV ({live_point:.4g}). One image, one point.",
        "A wider band collects more photons (less noise) but lets in more of the "
        "emission line's width, which blurs the HERFD toward an ordinary fluorescence "
        "spectrum. The right panel shows how the value per pixel of width changes as the "
        "band widens; how it changes depends on the line shape under the band, which "
        "chapter 8 shows is a doublet.",
    ], [fig7], box(
        f"ROI centre pixel (default: automatic fit) and ROI width <code>n</code> "
        f"(portal and live tab default <b>{ROI_N}</b>; note "
        "<code>OnePotRIXS.herfd</code> used directly defaults to <b>3</b>).",
        "<code>OnePotRIXS.herfd</code>, <code>live.HerfdAccumulator.result</code>.",
        "The two defaults differ (7 in the portal, 3 in the bare library call), so a "
        "notebook result and a portal result are not comparable unless n is set. A "
        "band on the wrong side of the line; a band so narrow that a single noisy "
        "column dominates."))
    S.cite(pipeline.OnePotRIXS.herfd, live.HerfdAccumulator.result)

    # ================================================================ 8 whole scan
    f = plotting.rixs_map(r1, pixels=(700, 2048))
    f.set_size_inches(10, 5.5)
    fig8a = img_tag(f, "RIXS map",
                    "The RIXS map: one row per image (incident energy, y), its emission "
                    "spectrum along x. The grey strip is the ROI band.")
    f = plotting.herfd_overlay([r1], labels=["series _01"], tfy=True)
    f.set_size_inches(10, 5)
    f.axes[0].set_ylabel("HERFD (counts / I0)")
    f.axes[1].set_ylabel("TFY (counts / I0)")
    fig8b = img_tag(f, "HERFD and TFY",
                    "Top: HERFD, the ROI band against incident energy. Bottom: TFY, the "
                    "whole detector summed.")
    # central pixel fit, replicated for display from the same profile
    ncols = r1.rixs_map.shape[1]
    prof = acc.rixs_map()[0]
    i0f = np.where((acc.rixs_map()[2] == 0), 1.0, acc.rixs_map()[2])
    prof = (prof / i0f)[:, max(0, ncols - 10):].sum(axis=1)
    xpix = np.arange(prof.size, dtype=float)
    gp = gauss_fwhm_fit(xpix, prof)
    sp9 = smooth(prof, 15)
    lo_w = max(0, cp - 150)
    pk_idx, pk_prop = find_peaks(sp9[lo_w:cp + 150], distance=15,
                                 prominence=0.03 * sp9.max())
    top2 = sorted(pk_idx[np.argsort(pk_prop["prominences"])[-2:]] + lo_w)
    p_lo, p_hi = (int(top2[0]), int(top2[-1])) if len(top2) == 2 else (cp, cp)
    ratio_lohi = float(sp9[p_lo] / sp9[p_hi])
    f = fig(10, 3.4)
    ax = f.subplots()
    ax.plot(xpix, prof, color=SERIES[0], lw=0.8, label="sum of the last 10 files in scan order")
    if gp is not None:
        ax.plot(xpix, pipeline._gaussian(xpix, *gp), color=INK, lw=1.5,
                label=f"gaussian fit: centre {gp[1]:.1f}, FWHM {2 * abs(gp[2]) * np.sqrt(np.log(2)):.0f} px")
    ax.axvline(cp, color=INK_2, ls="--", lw=1)
    for pp in (p_lo, p_hi):
        ax.axvline(pp, color=SERIES[1], ls=":", lw=1.4)
    ax.annotate(f"two peaks: {p_lo} and {p_hi}", xy=(p_hi, 0.6),
                xycoords=("data", "axes fraction"), xytext=(12, 0),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlim(700, 2048)
    ax.set_xlabel("dispersive pixel")
    ax.set_ylabel("counts / I0")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig8c = img_tag(f, "central pixel fit",
                    "How the ROI centre is chosen: a gaussian fit to the emission profile "
                    "of the last 10 files in scan order.")
    roi_try = {}
    for c_ in (p_lo, cp, p_hi):
        acc.set_roi(c_, ROI_N)
        rr = acc.result()
        roi_try[c_] = (rr, export.normalize_mu(rr.E, rr.HERFD, NORM))
    acc.set_roi(None, ROI_N)
    f = fig(10, 3.8)
    ax = f.subplots()
    for i, (c_, (rr, nn)) in enumerate(roi_try.items()):
        tag = "automatic" if c_ == cp else ("lower peak" if c_ == p_lo else "upper peak")
        ax.plot(rr.E, nn.flat, color=SERIES[i], lw=1.6, ls=LINESTYLES[i],
                label=f"ROI {c_} ± {ROI_N // 2} ({tag}): white line {np.max(nn.flat):.2f}")
    ax.set_xlim(r1.E.min(), r1.E.min() + 45)
    ax.set_xlabel("incident energy (eV)")
    ax.set_ylabel("normalised HERFD")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig8d = img_tag(f, "ROI choice", "The same scan read out with the band on each of the "
                    "two peaks and at the automatic centre between them (first 45 eV).")
    wl_by_roi = [float(np.max(nn.flat)) for _, nn in roi_try.values()]
    facts.update(p_lo=p_lo, p_hi=p_hi, wl_by_roi=wl_by_roi)
    tfy_n = r1.TFY / r1.TFY.max()
    her_n = r1.HERFD / r1.HERFD.max()
    facts.update(herfd_wl_fwhm=fwhm(r1.E, her_n), tfy_wl_fwhm=fwhm(r1.E, tfy_n))
    S.add("8. The whole scan: RIXS map, HERFD and TFY", [
        f"Doing chapters 1 to 7 for all {len(m1.data_paths)} images of the series and "
        "stacking the spectra gives the <b>RIXS map</b>: incident energy against emission "
        "pixel. Reading along the ROI band gives the <b>HERFD</b> spectrum; summing the "
        "whole detector gives <b>TFY</b> (total fluorescence yield).",
        0,
        f"HERFD is sharper: its white line is {facts['herfd_wl_fwhm']:.1f} eV wide at half "
        f"height against {facts['tfy_wl_fwhm']:.1f} eV for TFY, because the narrow band "
        "removes much of the core-hole lifetime broadening that TFY carries.",
        1,
        "The map also shows a weak diagonal streak whose emission position moves with the "
        "incident energy: resonant (Raman-like) scattering at a fixed energy loss. It "
        "crosses the ROI band just below the edge; a band placed elsewhere would pick up "
        "more or less of it.",
        f"<b>The automatic ROI centre.</b> The library sums the last 10 files in scan "
        f"order (the highest incident energies, above the edge) and fits ONE gaussian; "
        f"its centre, column {cp}, becomes the ROI centre. But the emission line has "
        f"<b>two peaks</b>, at columns {p_lo} and {p_hi} (height ratio "
        f"{ratio_lohi:.2f}): most likely Kα2 and Kα1, the 2p1/2 and 2p3/2 components "
        "(if emission energy rises with column, as it does in the calibration of chapter "
        "12, they are in the expected order, and the ratio is roughly the expected 1:2). "
        "A single gaussian over "
        "a doublet lands between the peaks, so the default band sits in the valley.",
        2,
        f"It matters: with the band on the lower peak, at the automatic centre, or on "
        f"the upper peak, the normalised white line is "
        f"{', '.join(f'{w:.2f}' for w in wl_by_roi)} edge steps. Part of that "
        "difference is the diagonal Raman streak, which crosses the lower peak right at "
        "the white line. There is no single right answer, but the choice should be "
        "made on purpose, and the same for every sample being compared.",
        3,
        f"Check: the live accumulator (portal Live tab) and the batch pipeline (portal "
        f"Processing) give the same map to {batch_live_diff:.1e} of its maximum.",
    ], [fig8a, fig8b, fig8c, fig8d], box(
        "ROI centre (<code>central_pix</code>, default automatic), ROI width "
        f"<code>n</code> (default {ROI_N}).",
        "<code>OnePotRIXS.herfd</code>, <code>pipeline._fit_central_pixel</code>, "
        "<code>live.HerfdAccumulator</code>; figures <code>plotting.rixs_map</code>, "
        "<code>plotting.herfd_overlay</code>.",
        "The auto-fit is one gaussian on the last 10 files <i>in file order</i>: on a "
        "doublet like this it picks the valley; on a scan that ends below the edge, or "
        "whose last files are weak, it picks noise. Always look at the map and the "
        "emission profile with the band drawn on them."))
    S.cite(pipeline._fit_central_pixel, live.HerfdAccumulator, plotting.rixs_map,
           plotting.herfd_overlay)

    # ================================================================ 9 I0
    acc.i0_corr = False
    acc.set_roi(cp, ROI_N)
    r_noi0 = acc.result()
    acc.i0_corr = True
    acc.set_roi(None, ROI_N)
    Es = np.sort(energies)
    I0s = i0s[np.argsort(energies)]
    post = r1.E > E_wl + 20
    f = fig(10, 5.2)
    a0, a1 = f.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    a0.plot(Es, I0s / 1e3, "o-", color=SERIES[0], ms=3, lw=1)
    a0.set_ylabel("I0 (thousand counts)")
    a0.set_title("I0 from each file's header", color=INK)
    style(a0)
    a1.plot(r_noi0.E, r_noi0.HERFD / r_noi0.HERFD[post].mean(), color=SERIES[1], lw=1.4,
            ls="--", label="HERFD, raw counts")
    a1.plot(r1.E, r1.HERFD / r1.HERFD[post].mean(), color=SERIES[0], lw=1.6,
            label="HERFD ÷ I0")
    a1.set_ylabel("HERFD, scaled to 1\n20+ eV above the peak")
    a1.set_xlabel("incident energy (eV)")
    a1.legend(frameon=False)
    style(a1)
    fig9 = img_tag(f, "I0 correction",
                   "Top: the incident-beam monitor over the scan. Bottom: HERFD without "
                   "and with the I0 division, scaled to the same post-edge level.")
    i0_var = 100 * (i0s.max() - i0s.min()) / i0s.mean()
    wl_ratio = (r1.HERFD[wl_i] / r1.HERFD[post].mean()) / (r_noi0.HERFD[wl_i] / r_noi0.HERFD[post].mean())
    S.add("9. I0 correction", [
        "The incident beam is not constant: the monochromator's throughput changes with "
        "energy and the storage-ring current decays. I0, an ion chamber upstream of the "
        "sample, records it, and each file's header carries its value.",
        f"Over this scan I0 changes by {i0_var:.0f} % (from {i0s.min():,.0f} to "
        f"{i0s.max():,.0f}), rising with energy. Dividing each image's counts by its I0 "
        f"removes that trend. The effect on the spectrum's shape is modest but not zero: "
        f"relative to the post-edge, the white line changes by "
        f"{100 * (wl_ratio - 1):+.1f} %.",
    ], [fig9], box(
        "<code>i0_corr</code> (default <b>on</b>; portal checkbox 'I0 correction').",
        "<code>OnePotRIXS.herfd(i0_corr=True)</code>; I0 read by "
        "<code>SifFile.I0</code>.",
        "A file with I0 = 0 or no I0 in its header is silently treated as I0 = 1, so one "
        "such point becomes a huge spike or dip. A beam dump mid-scan shows as a sudden "
        "I0 drop; the division corrects the counts but those points are noisier. For "
        "multi-frame XES files the header holds one I0 for the whole file."))
    S.cite(ta.SifFile.I0.fget)

    # ================================================================ 10 normalise
    norm_def = export.normalize_mu(r1.E, r1.HERFD)          # what the portal writes
    norm = export.normalize_mu(r1.E, r1.HERFD, NORM)        # used from here on
    facts.update(e0=norm.e0, step=norm.edge_step, step_default=norm_def.edge_step,
                 norm_method=norm.method)
    tail = r1.E > r1.E.max() - 25
    span_post = float(r1.E.max() - norm.e0)

    def larch_lines(ov):
        try:
            from larch import Group
            from larch.xafs import pre_edge
            g = Group(energy=r1.E.astype(float), mu=r1.HERFD.astype(float))
            pre_edge(g.energy, g.mu, group=g, **ov)
            d = g.pre_edge_details
            return np.asarray(g.pre_edge), np.asarray(g.post_edge), (d.norm1, d.norm2, d.nnorm)
        except Exception:  # noqa: BLE001 -- display-only lines
            return None, None, None

    pre_d, post_d, par_d = larch_lines({})
    pre_c, post_c, par_c = larch_lines(NORM)
    f = fig(10, 5.8)
    a0, a1 = f.subplots(2, 1, sharex=True)
    a0.plot(r1.E, r1.HERFD, color=SERIES[0], lw=1.6, label="HERFD (μ)")
    if pre_d is not None:
        a0.plot(r1.E, pre_d, color=INK_2, ls="--", lw=1, label="pre-edge line")
        a0.plot(r1.E, post_d, color=SERIES[1], ls=":", lw=1.6,
                label=f"post-edge, default (E0+{par_d[0]:g} to E0+{par_d[1]:g} eV)")
        a0.plot(r1.E, post_c, color=SERIES[2], ls="-.", lw=1.6,
                label=f"post-edge, used here (E0+{par_c[0]:g} to end)")
    a0.axvline(norm.e0, color=INK_2, lw=0.8)
    a0.annotate(f"E0 {norm.e0:.2f} eV", xy=(norm.e0, 0.9), xycoords=("data", "axes fraction"),
                xytext=(-70, 0), textcoords="offset points", fontsize=9, color=INK)
    a0.set_ylabel("counts / I0")
    a0.legend(frameon=False, loc="upper right", fontsize=9)
    style(a0)
    a1.plot(r1.E, norm_def.flat, color=SERIES[1], lw=1.4, ls=":",
            label=f"default: edge step {norm_def.edge_step:.1f}")
    a1.plot(r1.E, norm.flat, color=SERIES[2], lw=1.7, ls="-.",
            label=f"used here: edge step {norm.edge_step:.1f}")
    a1.axhline(0, color=INK_2, lw=0.6)
    a1.axhline(1, color=INK_2, lw=0.6)
    a1.set_ylabel("normalised μ (flat)")
    a1.set_xlabel("incident energy (eV)")
    a1.legend(frameon=False, loc="upper right", fontsize=9)
    style(a1)
    fig10 = img_tag(f, "normalisation",
                    "Top: HERFD with the pre-edge line and two post-edge fits (drawn with "
                    "xraylarch's pre_edge). Bottom: the normalised result for each; 0 = "
                    "pre-edge, 1 = one edge step.")
    S.add("10. Normalisation", [
        "To compare spectra measured with different counting times, sample thicknesses or "
        "concentrations, they are put on a common scale: fit a straight line to the "
        "region before the edge and a smooth curve to the region after it, subtract the "
        "first and divide by their separation at the edge, the <b>edge step</b>. "
        "Then 0 means 'before the edge' and 1 means 'one edge step'. <i>flat</i> also "
        "removes the post-edge slope.",
        0,
        f"E0 (the maximum of the first derivative) is <b>{norm.e0:.2f} eV</b>. The "
        f"normalisation is done by <b>{html.escape(norm.method)}</b>, the routine the "
        "portal uses.",
        f"<b>The default post-edge range fails on this scan.</b> The scan ends only "
        f"{span_post:.0f} eV above the edge, and the default fit (E0+{par_d[0]:g} to "
        f"E0+{par_d[1]:g} eV, a straight line) starts on the strong resonance near "
        f"2496 eV, so it slopes steeply down and meets the edge far too high: edge step "
        f"{norm_def.edge_step:.1f} instead of about {norm.edge_step:.1f}. The default "
        f"'norm' then ends at {np.nanmean(norm_def.norm[tail]):.2f} instead of 1, and "
        f"every normalised height is too small by about "
        f"{norm_def.edge_step / norm.edge_step:.1f}×. Starting the post-edge fit at "
        f"E0+{par_c[0]:g} eV puts the end of the scan at "
        f"{np.nanmean(norm.norm[tail]):.2f}; the rest of this story uses that.",
        "The honest summary: with a post-edge this short, a normalised height is only "
        "as good as the chosen range. Normalise every sample you compare with exactly "
        "the same ranges, and scan further above the edge when heights matter.",
    ], [fig10], box(
        "Pre- and post-edge ranges (larch <code>pre1, pre2, norm1, norm2, nnorm</code>) "
        f"and E0: automatic by default. Used here: <code>{NORM}</code>.",
        "<code>export.normalize_mu</code> (chemcat's xas_core, then xraylarch, then a "
        "built-in fallback; the one used is recorded), written by "
        "<code>export.write_xas_csv</code>.",
        "Check that 'norm' sits near 1 at the end of the scan; if not, the edge step is "
        "wrong and so is every normalised number. The portal's Tender CSVs use the "
        "defaults. E0 from the maximum derivative is the steepest point of the edge, not "
        "the white-line peak."))
    S.cite(export.normalize_mu, export.write_xas_csv)

    # ================================================================ 11 average
    rdir = os.path.join(real, "Na2SO4_pellet")
    if dirs_with_sif(rdir):
        m2, idx2 = rixs_measurement(rdir)
        groups = average.group_repeats(list(idx1) + list(idx2))
        key = (m1.sample, m1.emission_line)
        reps = groups.get(key, [])
        log(f"averaging: {key} -> {[m.series_index for m in reps]}")
        lit = {}
        for m in reps:
            lit[m.series_index] = dark_has_signal(
                ta.SifFile(m.dark_paths[0]).data.mean(axis=0))[1] if m.dark_paths else 0.0
        use_frames = {k: v > 5.0 for k, v in lit.items()}
        cp2_auto = m2.run(n=ROI_N, use_dark_as_background=not use_frames.get(
            m2.series_index, False)).central_pix
        runs = [m.run(n=ROI_N, central_pix=cp,
                      use_dark_as_background=not use_frames.get(m.series_index, False))
                for m in reps]
        gc.collect()
        labels = [f"series _{m.series_index:02d}" for m in reps]
        avg = average.average_series(runs, labels=labels)
        f = plotting.herfd_overlay(runs, labels=labels)
        f.set_size_inches(10, 3.6)
        for ln, ls in zip(f.axes[0].get_lines(), LINESTYLES):
            ln.set_linestyle(ls)
        f.axes[0].set_ylabel("HERFD (counts / I0)")
        fig11a = img_tag(f, "two series", "The two repeats, same ROI, same settings.")
        f = plotting.averaged(avg, label="mean of 2 series")
        f.set_size_inches(10, 4.6)
        f.axes[0].set_ylabel("HERFD (counts / I0)")
        f.axes[1].set_ylabel("series")
        fig11b = img_tag(f, "average with spread",
                         "The mean with a ±1σ band (σ over the two series); the small "
                         "panel counts how many series cover each energy.")
        # energy offset between the repeats: shift series 2 to best match series 1
        n_1 = export.normalize_mu(runs[0].E, runs[0].HERFD, NORM)
        n_2 = export.normalize_mu(runs[-1].E, runs[-1].HERFD, NORM)
        fine = np.arange(runs[0].E.min() + 5, runs[0].E.min() + 40, 0.05)
        y1 = np.interp(fine, runs[0].E, n_1.flat)
        shifts = np.arange(-3, 3.001, 0.05)
        cost = [np.mean((np.interp(fine + d, runs[-1].E, n_2.flat) - y1) ** 2) for d in shifts]
        dE = float(shifts[int(np.argmin(cost))])
        facts["series_shift_eV"] = dE
        shifted = (runs[-1].E - dE, runs[-1].HERFD, runs[-1].TFY)
        avg_al = average.average_series(runs[:-1] + [shifted], labels=labels)
        nav = average.normalize_average(avg_al, NORM)

        # counting-statistics expectation at each point
        band_adu = np.array([r.HERFD for r in runs]) * np.array(
            [np.interp(r.E, Es, I0s) for r in runs])
        photons = band_adu / photon_adu
        rel_poiss = 1 / np.sqrt(np.maximum(photons.mean(axis=0), 1)) / np.sqrt(2)
        good = r1.E > norm.e0 - 3
        rel_obs = avg.std / np.abs(avg.mean)
        rel_al = avg_al.std / np.abs(avg_al.mean)
        ratio = float(np.nanmedian(rel_obs[good] / rel_poiss[good]))
        ratio_al = float(np.nanmedian(rel_al[good] / rel_poiss[good]))
        facts.update(spread_ratio=ratio, spread_ratio_aligned=ratio_al)
        f = fig(10, 3.4)
        ax = f.subplots()
        ax.plot(avg.E, 100 * rel_obs, "o", color=SERIES[1], ms=3.5, label="observed σ / mean, as measured")
        ax.plot(avg_al.E, 100 * rel_al, "s", color=SERIES[0], ms=3.5, mfc="none",
                label=f"observed, after shifting series _02 by {-dE:+.2f} eV")
        ax.plot(avg.E, 100 * rel_poiss, color=INK, lw=1.2, ls="--",
                label="expected from photon counting alone")
        ax.set_yscale("log")
        ax.set_xlabel("incident energy (eV)")
        ax.set_ylabel("relative spread (%)")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
        fig11c = img_tag(f, "spread vs counting noise",
                         f"Spread between the repeats, point by point, against what photon "
                         f"counting alone would give (photons = band ADU ÷ {photon_adu} ADU "
                         "per photon).")
        _nr = [n_1, n_2]
        wl2 = [float(np.max(n_.flat)) for n_ in _nr]
        e0s = [float(n_.e0) for n_ in _nr]
        S.add("11. Averaging repeats", [
            f"The bundled data is series <code>_01</code>; the real-data folder holds "
            f"series <code>_02</code> of the same pellet. <code>group_repeats</code> "
            f"recognises them as repeats of one (sample, emission line) = "
            f"({html.escape(key[0])}, {key[1]}) and <code>average_series</code> puts "
            "them on one energy grid and returns the mean, the spread and the count.",
            (f"Both were reduced with the same ROI (centre {cp}; the automatic fit on "
             f"series _02 alone gives {cp2_auto}) and the same background rule. The "
             f"'dark' of series _02 is lit too, even more than _01's (up to "
             f"{max(lit.values()):.0f} ADU per pixel), so both use the min-projection "
             "background (chapter 3).") if all(use_frames.values()) else
            (f"Both were reduced with the same ROI (centre {cp}; the automatic fit on "
             f"series _02 alone gives {cp2_auto})."),
            0,
            f"<b>The two repeats are not at the same energy.</b> Their headers give "
            f"identical monochromator values, but series _02's edge and white line sit "
            f"{abs(dE):.2f} eV {'higher' if dE > 0 else 'lower'} (E0 "
            f"{e0s[0]:.2f} vs {e0s[1]:.2f} eV). Something moved between the two scans: "
            "the monochromator's true energy, or the beam position on the crystal. "
            "Averaging them as they are blurs the edge.",
            1,
            f"<b>What the spread tells you.</b> With two repeats σ is half their "
            f"difference, a rough estimate. As measured, the spread is typically "
            f"{ratio:.1f}× what photon counting alone would give (median over points "
            f"from the edge up). After shifting series _02 by {-dE:+.2f} eV it drops to "
            f"{ratio_al:.1f}×. "
            + ("So once the energy offset is removed the repeats agree to about counting "
               "noise." if ratio_al < 1.6 else
               "Even after the shift there is more than counting noise left, so something "
               "else (intensity drift, ROI, sample) also changed.")
            + f" The normalised white lines are {wl2[0]:.2f} and {wl2[1]:.2f} edge steps. "
            "Beam damage (sulfate being reduced) would show as a lower white line and new "
            "intensity a few eV below the edge in the later scan; neither is seen here.",
            2,
            f"The aligned average, normalised, has E0 = {nav['e0']:.2f} eV on series _01's "
            "energy scale.",
        ], [fig11a, fig11b, fig11c], box(
            "Which series to average (all repeats of a sample and line), the common "
            "energy grid (default: the first series), <code>ddof</code> (default 0).",
            "<code>average.group_repeats</code>, <code>average.average_series</code>, "
            "<code>average.normalize_average</code>; figure <code>plotting.averaged</code>.",
            "Repeats with different ROIs, thresholds or I0 settings are not comparable; "
            "set them the same. Energy points that only one series covers have zero "
            "spread, not zero uncertainty (see the 'n' panel). A steady trend from repeat "
            "to repeat is damage or drift, not noise, and averaging hides it."))
        S.cite(average.group_repeats, average.average_series, average.normalize_average,
               plotting.averaged)
        del runs
        gc.collect()
    else:
        S.add("11. Averaging repeats", [], skipped=f"Skipped: no second series found "
              f"(needs {html.escape(rdir or '--real DIR/Na2SO4_pellet')}).")

    # ------------------------------------------------ XES reductions (for 12/13)
    xes_dirs = [(n, os.path.join(real, n)) for n in ("Ag2S", "AgNO3", "P12S")]
    xes_dirs = [(n, d) for n, d in xes_dirs if dirs_with_sif(d)]
    xes = []
    for name, d in xes_dirs:
        idx = ta.index_beamtime(d)
        for m in idx.by_kind("XES"):
            log(f"XES: {m!r}")
            op = m.pipeline(scan=True)
            res = op.run()
            per_scan = res.scan_data.copy()      # (pixel, file)
            spec_x = res.spectrum().copy()
            i0_x = op.i0_values()
            nfr = sum(s.num_frames for s in op.sif_files)
            # grain-size check on one frame of the first file
            sfx = op.sif_files[0]
            hx = {k: np.zeros(NBINS, dtype=np.int64)
                  for k in ("xray", "bkg_free", "binned", "raw", "background")}
            bx = m._dark_background()
            reduce_frame(sfx.frame(0), bx, background_common_mode(bx), th, hist=hx)
            ph = int(150 + np.argmax(smooth(hx["xray"].astype(float), 9)[150:1500]))
            xes.append(dict(name=name, m=m, spec=spec_x, per_scan=per_scan, i0=i0_x,
                            nfr=nfr, exp=sfx.exposure_time, photon_adu=ph,
                            n_dark=len(m.dark_paths)))
            del op, res, sfx
            gc.collect()

    # ================================================================ 12 calibration
    cal = None
    if dirs_with_sif(edir):
        log("calibration: indexing elastic scans")
        paths = sorted(glob.glob(os.path.join(edir, "*.sif")))
        recs = [parse_sif_name(p) for p in paths]
        e_data = [r for r in recs if r and r.is_elastic and not r.is_dark]
        e_dark = [r for r in recs if r and r.is_elastic and r.is_dark]
        idx_e = ta.index_beamtime(edir)
        cal_note = ""
        try:
            points = calibration.index_elastic(edir)
        except ValueError as exc:
            # One dark for the whole elastic set: pair every scan with it.
            cal_note = (f"The library's pairing rule wants a dark at each elastic energy "
                        f"and refused ({html.escape(str(exc)[:120])}...). This folder "
                        f"has {len(e_dark)} dark for {len(e_data)} elastic scans, so "
                        "every scan was reduced with that one dark instead.")
            dbg = np.mean([ta.SifFile(d.path).data.mean(axis=0) for d in e_dark], axis=0)
            points = []
            for r in e_data:
                sp = ta.OnePot(r.path, bcg=dbg).run().spectrum()
                points.append(calibration.ElasticPoint(energy=float(ta.SifFile(r.path).mono),
                                                       spectrum=sp, path=r.path))
                gc.collect()
        points.sort(key=lambda p: p.energy)
        cal = calibration.ElasticCalibration.fit(points)
        facts.update(cal_m=cal.m, cal_b=cal.b, cal_rms=cal.rms)
        nfr = ta.SifFile(e_data[0].path).num_frames if e_data else 0
        # display: one elastic frame, photon events
        pv_e = preview(points[0].path, dark=e_dark[0].path if e_dark else None, frame=0,
                       curvature_t=1)
        gc.collect()
        f = fig(10, 6.4)
        gs = f.add_gridspec(3, 2, height_ratios=[1, 1.3, 1])
        a0 = f.add_subplot(gs[0, :])
        a0.imshow(pv_e["events"], cmap="gray", aspect="auto",
                  extent=(-0.5, 2047.5, 511.5, -0.5), interpolation="antialiased")
        a0.set_title(f"one frame (of {nfr}) of {os.path.basename(points[0].path)}: "
                     "photon events", color=INK)
        a0.set_ylabel("row")
        a1 = f.add_subplot(gs[1, :], sharex=a0)
        xpx = np.arange(2048, dtype=float)
        for i, p in enumerate(points):
            y = p.spectrum / p.spectrum.max()
            a1.plot(xpx, y, color=SERIES[i], lw=1, ls=LINESTYLES[i % 4],
                    label=f"{p.energy:.1f} eV → pixel {cal.centers[i]:.1f}")
            pk = int(np.argmax(p.spectrum))
            try:
                popt, _ = curve_fit(calibration._lorentzian, xpx, p.spectrum,
                                    p0=[p.spectrum.max() - np.median(p.spectrum), pk, 3.0],
                                    maxfev=10000)
                a1.plot(xpx, calibration._lorentzian(xpx, *popt) / p.spectrum.max(),
                        color=INK, lw=0.8)
            except (RuntimeError, ValueError):
                pass
        a1.set_xlabel("dispersive pixel")
        a1.set_ylabel("elastic spectrum\n(scaled to 1)")
        a1.legend(frameon=False, fontsize=9)
        style(a1)
        lo_c, hi_c = cal.centers.min(), cal.centers.max()
        a1.set_xlim(max(0, lo_c - 250), min(2047, hi_c + 250))
        a2 = f.add_subplot(gs[2, 0])
        a2.plot(cal.centers, cal.energies, "o", color=SERIES[0], ms=7)
        xl = np.array([0, 2047])
        a2.plot(xl, cal.to_energy(xl), color=INK, lw=1)
        a2.set_xlabel("fitted centre pixel")
        a2.set_ylabel("mono energy (eV)")
        a2.set_title(f"E = {cal.m:.5f} × pixel + {cal.b:.2f}", color=INK)
        style(a2)
        a3 = f.add_subplot(gs[2, 1])
        a3.axhline(0, color=INK_2, lw=0.8)
        a3.plot(cal.energies, cal.residuals * 1000, "o", color=SERIES[0], ms=7)
        a3.set_xlabel("mono energy (eV)")
        a3.set_ylabel("residual (meV)")
        a3.set_title(f"RMS {cal.rms * 1000:.1f} meV", color=INK)
        style(a3)
        fig12a = img_tag(f, "elastic calibration",
                         "Top: an elastic frame. Middle: the elastic line at each mono "
                         "energy with its Lorentzian fit (black). Bottom: the linear fit "
                         "and its residuals.")
        span_e = cal.energies.max() - cal.energies.min()
        dof = len(cal.centers) - 2
        # Independent check: each XES spectrum has a small elastic peak at its own
        # incident energy, outside the calibrated range.
        checks = []
        figs12 = [fig12a]
        if xes:
            E_ax = cal.to_energy(np.arange(2048))
            f = fig(10, 3.8)
            ax = f.subplots()
            for i, x in enumerate(xes):
                inc = x["m"].incident_energy
                # the elastic peak: the local maximum nearest the incident energy
                near = np.flatnonzero(np.abs(E_ax - inc) < 1.5)
                seg = smooth(x["spec"], 3)[near]
                j = int(np.clip(np.argmax(seg), 1, seg.size - 2))
                y0_, y1_, y2_ = seg[j - 1:j + 2]
                den = y0_ - 2 * y1_ + y2_
                c_px = near[0] + j + (0.5 * (y0_ - y2_) / den if den else 0.0)
                checks.append((x["name"], inc, float(cal.to_energy(c_px))))
                ax.plot(E_ax, x["spec"] / x["spec"].max(), color=SERIES[i], lw=1.4,
                        ls=LINESTYLES[i], label=f"{x['name']} ({inc:g} eV incident)")
                ax.axvline(inc, color=SERIES[i], lw=0.8, ls=":")
            ax.axvspan(cal.energies.min(), cal.energies.max(), color=plotting.ROI,
                       alpha=0.15, lw=0)
            ax.annotate("calibration points", xy=(cal.energies.mean(), 0.97),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=9, color=INK_2)
            ax.set_xlim(cal.energies.min() - 8, max(c[1] for c in checks) + 8)
            ax.set_xlabel("emission energy (eV)")
            ax.set_ylabel("counts, peak = 1")
            ax.legend(frameon=False, fontsize=9, loc="upper left")
            style(ax)
            figs12.append(img_tag(
                f, "XES on eV axis",
                "The three Ag XES spectra on the calibrated axis. Dotted lines: each "
                "sample's incident energy. The small peak there is elastic scattering "
                "from the sample itself, a free check of the calibration."))
            dev = [c[2] - c[1] for c in checks]
            facts["cal_check"] = checks
        S.add("12. From pixel to emission energy", [
            "Up to here the emission axis is a pixel number. To turn it into energy, the "
            "spectrometer looks at <b>elastic scattering</b> from a sample (here boron "
            "nitride, BN) with the monochromator at a few known energies. Elastically "
            "scattered X-rays keep their energy, so each mono energy lights up one "
            "place on the detector. Fitting each line's centre and then a straight line "
            "through (centre pixel, energy) gives <i>energy = m × pixel + b</i>.",
            0,
            f"The folder holds {len(e_data)} elastic scans of {nfr} frames each "
            f"({', '.join(f'{p.energy:.0f}' for p in points)} eV). The indexer treats "
            f"them as calibration files and leaves them out of normal analysis "
            f"(<code>index_beamtime</code> found {len(idx_e)} measurements there). "
            + cal_note,
            f"Result: <b>{abs(cal.m) * 1000:.2f} meV per pixel</b>, energy "
            f"{'rising' if cal.m > 0 else 'falling'} with column number, RMS residual "
            f"{cal.rms * 1000:.1f} meV. The full 2048 columns span "
            f"{abs(cal.m) * 2047:.1f} eV ({cal.to_energy(0):.1f} to "
            f"{cal.to_energy(2047):.1f} eV).",
            f"Two cautions. {len(cal.centers)} points and two fit parameters leave "
            f"{dof} degree{'s' if dof != 1 else ''} of freedom, so the RMS is a rough "
            f"figure of merit; and the fitted centres are Lorentzian fits to lines that "
            f"are visibly asymmetric (broadened on one side by the bend of chapter 5), "
            f"so the centres may be biased by a similar amount for every line. The "
            f"points span only {span_e:.0f} eV (about {span_e / abs(cal.m):.0f} of the "
            f"2048 columns); anything outside {cal.energies.min():.0f}–"
            f"{cal.energies.max():.0f} eV is an extrapolation of the straight line.",
            1,
        ] + ([
            "<b>A free check.</b> Every XES spectrum also contains a small elastic peak at "
            "its own incident energy, 11 to 12 eV beyond the last calibration point "
            "(Ag₂S shows two small peaks there; the one nearest the incident energy is "
            "taken). On the calibrated axis they land at "
            + "; ".join(f"{n}: {e:.2f} eV for {i:g} eV incident" for n, i, e in checks)
            + f", i.e. {min(dev):+.2f} to {max(dev):+.2f} eV off, against a fit RMS of "
            f"{cal.rms * 1000:.0f} meV. "
            + ("So the straight line extrapolates well over this range."
               if max(abs(d) for d in dev) < 0.15 else
               f"So beyond the calibration points the axis is good to about "
               f"{max(abs(d) for d in dev):.1f} eV, not to tens of meV; elastic points "
               "that bracket the emission would settle whether the dispersion bends.")
        ] if checks else []), figs12, box(
            "The elastic energies measured (at least 2; more, spread across the "
            "detector, is better); <code>fit_window</code> (default: fit the whole "
            "profile).",
            "<code>calibration.index_elastic</code>, <code>ElasticCalibration.fit</code>, "
            "<code>ElasticCalibration.to_energy</code>; applied by "
            "<code>export.write_xes_csv(calibration=...)</code>.",
            "A calibration is valid only for the spectrometer geometry it was measured "
            "in: after the analyser or detector moves, measure it again. Elastic scans "
            "without a dark at the same energy are refused by the library. A residual "
            "pattern (a bow) means the dispersion is not linear."))
        S.cite(calibration.index_elastic, calibration.ElasticCalibration.fit,
               calibration.ElasticCalibration.to_energy, export.write_xes_csv)
    else:
        S.add("12. From pixel to emission energy", [], skipped="Skipped: no elastic scans "
              f"(needs {html.escape(edir or '--real DIR/elastic_BN')}); XES stays on a "
              "pixel axis.")

    # ================================================================ 13 interpretation
    paras = []
    figs13 = []
    wl_norm = float(np.max(norm.flat))
    E_peak = float(r1.E[int(np.argmax(norm.flat))])
    pre_region = (r1.E < norm.e0 - 4)
    pre_max = float(np.max(norm.flat[pre_region])) if pre_region.any() else float("nan")
    f = fig(10, 3.8)
    ax = f.subplots()
    ax.plot(r1.E, norm.flat, color=SERIES[0], lw=1.8, label="Na₂SO₄, HERFD, series _01")
    dmu = np.gradient(norm.flat, r1.E)
    ax.plot(r1.E, dmu / np.nanmax(dmu) * 0.5 * wl_norm, color=INK_2, lw=1, ls="--",
            label="first derivative (scaled)")
    ax.axvline(norm.e0, color=INK_2, lw=0.8)
    ax.annotate(f"E0 {norm.e0:.2f}", xy=(norm.e0, 0.4), xycoords=("data", "axes fraction"),
                xytext=(-58, 0), textcoords="offset points", fontsize=9, color=INK)
    ax.annotate(f"white line {E_peak:.2f} eV, {wl_norm:.1f} × edge step",
                xy=(E_peak, wl_norm), xytext=(20, -10), textcoords="offset points",
                fontsize=9, color=INK)
    ax.set_xlabel("incident energy (eV, mono readback, not calibrated)")
    ax.set_ylabel("normalised μ (flat)")
    ax.set_xlim(r1.E.min(), r1.E.max())
    ax.legend(frameon=False, loc="center right")
    style(ax)
    figs13.append(img_tag(f, "sulfate HERFD", "The sulfate HERFD, normalised."))
    paras += [
        "<b>Sulfur K-edge HERFD of sodium sulfate.</b> The pellet is Na₂SO₄ diluted in "
        "sucrose (which contains no sulfur). The spectrum is one intense, narrow white "
        f"line: E0 (steepest rise) at {norm.e0:.2f} eV and the peak at {E_peak:.2f} eV, "
        f"{wl_norm:.1f} times the edge step, followed by weaker, broader features roughly "
        "8 to 16 eV higher.",
        "That shape is what sulfate, sulfur in its highest oxidation state (+6), looks "
        "like: the S K edge moves up in energy as sulfur is oxidised, and sulfate's "
        "white line lies several eV above those of sulfide, elemental sulfur or "
        f"sulfoxides. Below the edge the spectrum is flat (at most {pre_max:.2f} of an "
        "edge step between the start of the scan and 4 eV below E0), so there is no sign "
        "of a reduced sulfur species at the level this scan can see.",
        "The energy axis is the monochromator's readback without a reference "
        "measured alongside it, so absolute energies should not be compared with "
        "published values to better than about 1 eV. Differences between spectra taken "
        "in the same session are more reliable than absolute positions.",
    ]

    paras.insert(1, 0)   # sulfate figure after its first paragraph
    if xes:
        use_e = cal is not None
        E_ax = cal.to_energy(np.arange(2048)) if use_e else np.arange(2048, dtype=float)
        unit = "eV" if use_e else "px"
        # Emission window: from 10 eV (100 px) below the main peak up to 2.5 eV
        # (25 px) below the lowest incident energy, i.e. excluding the elastic peak.
        tot = sum(smooth(x["spec"], 9) / x["spec"].sum() for x in xes)
        pk_i = int(np.argmax(tot))
        if use_e:
            e_hi = min(x["m"].incident_energy for x in xes) - 2.5
            sel = (E_ax >= E_ax[pk_i] - 10) & (E_ax <= e_hi)
        else:
            sel = (np.arange(2048) >= pk_i - 100) & (np.arange(2048) <= pk_i + 50)
        idx_w = np.flatnonzero(sel)
        w = slice(idx_w.min(), idx_w.max() + 1)
        xsl = E_ax[w]
        order_x = np.argsort(xsl)
        trap = getattr(np, "trapezoid", None) or np.trapz
        stats, curves = [], {}
        for x in xes:
            y = smooth(x["spec"], 9)[w].astype(float)
            y = y / abs(trap(y[order_x], xsl[order_x]))
            curves[x["name"]] = y
            pk = float(xsl[int(np.argmax(y))])
            cen = float(np.sum(xsl * y) / np.sum(y))
            stats.append(dict(name=x["name"], peak=pk, fwhm=fwhm(xsl, y), cen=cen,
                              inc=x["m"].incident_energy, nscan=len(x["m"].data_paths)))
        names = [st["name"] for st in stats]
        ref_name = "AgNO3" if "AgNO3" in curves else names[0]
        f = fig(10, 6.2)
        a0, a1 = f.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
        for i, st in enumerate(stats):
            a0.plot(xsl, curves[st["name"]], color=SERIES[i], lw=1.7, ls=LINESTYLES[i],
                    label=f"{st['name']} ({st['inc']:g} eV incident, {st['nscan']} scans)")
            a1.plot(xsl, curves[st["name"]] - curves[ref_name], color=SERIES[i], lw=1.5,
                    ls=LINESTYLES[i])
        a0.set_ylabel(f"intensity, area = 1 (per {unit})")
        a0.legend(frameon=False, fontsize=9, loc="upper left")
        style(a0)
        a1.axhline(0, color=INK_2, lw=0.6)
        a1.set_ylabel(f"minus {ref_name}")
        a1.set_xlabel("emission energy (eV)" if use_e else "dispersive pixel (uncalibrated)")
        style(a1)
        # two-reference linear combination for P12S (non-negative least squares)
        lcf = None
        if {"P12S", "Ag2S", "AgNO3"} <= set(curves):
            from scipy.optimize import nnls
            A = np.column_stack([curves["Ag2S"], curves["AgNO3"]])
            coef, rnorm = nnls(A, curves["P12S"])
            fitc = A @ coef
            resid = float(np.sqrt(np.mean((curves["P12S"] - fitc) ** 2)) / curves["P12S"].max())
            lcf = dict(ag2s=coef[0] / coef.sum(), agno3=coef[1] / coef.sum(), resid=resid,
                       total=float(coef.sum()))
            a0.plot(xsl, fitc, color=INK, lw=0.9,
                    label=f"P12S fit: {100 * lcf['ag2s']:.0f} % Ag2S + {100 * lcf['agno3']:.0f} % AgNO3")
            a0.legend(frameon=False, fontsize=9, loc="upper left")
        if use_e and cal.m < 0:
            a1.invert_xaxis()
        fig_x = img_tag(f, "Ag XES comparison",
                        "Ag L3-valence emission of the three samples, each normalised to unit "
                        "area over the emission window (9-pixel running mean; the elastic "
                        "peak is excluded), and each minus AgNO3.")
        # per-scan drift of the centroid, inside the same window
        drift = []
        for x in xes:
            ps = x["per_scan"][w].astype(float)
            ps = ps / ps.sum(axis=0, keepdims=True)
            c = (xsl[:, None] * ps).sum(axis=0)
            drift.append((x["name"], c - c[0], ps.shape[1]))
        f = fig(10, 3.0)
        ax = f.subplots()
        for i, (nm, dc, nscan) in enumerate(drift):
            ax.plot(np.arange(1, nscan + 1), dc * (1000 if use_e else 1), "o",
                    color=SERIES[i], ls=LINESTYLES[i], ms=6, label=nm)
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.set_xticks(range(1, max(d[2] for d in drift) + 1))
        ax.set_xlabel("scan number")
        ax.set_ylabel(f"centroid shift vs scan 1\n({'meV' if use_e else 'px'})")
        ax.legend(frameon=False)
        style(ax)
        fig_d = img_tag(f, "XES scan-to-scan",
                        "Does the emission move from scan to scan? Centroid of each scan "
                        "(same window) relative to the first.")
        facts["xes_stats"] = stats
        facts["xes_lcf"] = lcf
        facts["xes_drift"] = [(n, float(np.max(np.abs(d)))) for n, d, _ in drift]
        s_by = {st["name"]: st for st in stats}
        lines = "".join(
            f"<li><b>{st['name']}</b>: peak {st['peak']:.2f} {unit}, centroid "
            f"{st['cen']:.2f} {unit}, width at half height {st['fwhm']:.2f} {unit}</li>"
            for st in stats)
        comp = ""
        if "Ag2S" in s_by and "AgNO3" in s_by:
            sa, sn = s_by["Ag2S"], s_by["AgNO3"]
            comp = (f"<b>What differs.</b> Ag₂S's main peak sits {sa['peak'] - sn['peak']:+.2f} "
                    f"{unit} from AgNO₃'s and is {sa['fwhm'] - sn['fwhm']:+.2f} {unit} wider at "
                    f"half height ({sa['fwhm']:.2f} vs {sn['fwhm']:.2f}). The difference curve "
                    "shows where: Ag₂S has clearly more intensity on the low-energy side of "
                    "the peak, a little more on the high-energy side, and less at the top. ")
        xe = xes[0]
        inc_list = ", ".join(f"{st['inc']:g}" for st in stats)
        nscans = sorted({st["nscan"] for st in stats})
        drift_txt = ", ".join(f"{n} {v * (1000 if use_e else 1):.0f} {'meV' if use_e else 'px'}"
                              for n, v in facts["xes_drift"])
        paras += [
            "<b>Ag L3-valence XES of Ag₂S, AgNO₃ and P12S.</b> Here the incident energy is "
            "fixed a few eV above the Ag L3 edge (about 3351 eV) and the emission spectrum "
            "itself is the result. This emission comes from occupied valence levels (mostly "
            "Ag 4d, mixed with the ligand's orbitals) filling the 2p3/2 core hole, so its "
            "shape reflects what silver is bonded to.",
            f"The indexer makes one XES measurement per sample: all scans "
            f"(<code>_01</code>, <code>_02</code>, ...; {'/'.join(map(str, nscans))} of them, "
            f"{xe['nfr'] // len(xe['m'].data_paths)} frames of {xe['exp']:g} s each) at one "
            "incident energy, with all their darks averaged into one background. The "
            f"samples were measured at slightly different incident energies ({inc_list} eV); "
            "this far above the edge that should matter little, but it is a difference.",
            f"<ul>{lines}</ul>",
            1,
            comp + "In broad terms, a more covalent silver–sulfur bond mixes more ligand "
            "character into the valence band and spreads it out, while an ionic "
            "silver–oxygen salt such as the nitrate keeps a narrower, more atomic-like 4d "
            "band. The broader Ag₂S band is consistent with that picture. The differences "
            "are a few tenths of an eV to one eV, so treat them as the result and the "
            "absolute energies as approximate (chapter 12).",
        ]
        if lcf is not None:
            paras.append(
                f"<b>P12S</b> is a sample code; its composition is not in the file names. "
                f"Its main peak ({s_by['P12S']['peak']:.2f} {unit}, width "
                f"{s_by['P12S']['fwhm']:.2f}) lies between the two references. A "
                f"two-reference fit describes it as {100 * lcf['ag2s']:.0f} % Ag₂S-like and "
                f"{100 * lcf['agno3']:.0f} % AgNO₃-like, leaving a residual of "
                f"{100 * lcf['resid']:.0f} % of the peak, mostly on the high-energy side, where "
                "P12S's strong elastic peak (chapter 12) still leaks in. That suggests silver with some "
                "sulfur-like and some oxygen-like character (or a mixture of sites), but two "
                "references cannot establish what the silver is bonded to; more references "
                "(metallic Ag, Ag₂O, a thiolate) would be needed before naming a species.")
        paras += [
            2,
            f"The scan-to-scan panel is a damage check (silver nitrate in particular is "
            f"light- and beam-sensitive). The largest centroid movement across scans is "
            f"{drift_txt}. "
            + ("None of these is large compared with the differences between samples, so "
               "averaging the scans is reasonable."
               if max(v for _, v in facts["xes_drift"]) < (0.15 if use_e else 1.5) else
               "Compare that with the differences between samples before trusting them."),
            f"(One Ag L photon gives about {xe['photon_adu']} ADU here, against "
            f"≈{photon_adu} for S Kα: the default <code>xray = 170</code> gate is well "
            "below both, so the same thresholds serve both lines.)",
        ]
        figs13 += [fig_x, fig_d]
    else:
        paras.append("<i>Ag XES comparison skipped: no Ag2S / AgNO3 / P12S folders under "
                     "--real.</i>")

    offer = """
<ul class="offer">
<li><b>Raw frame, pedestal, dark (1-3).</b> Portal Live tab shows each new image raw, dark-subtracted or as photon events; Processing has a dark auto/none switch. Notebook <code>01_one_image</code> walks one image through every step, with the ADU histogram. <i>Missing:</i> a check that the paired dark is actually dark. The automatic first pass uses the paired dark, so for a series like this one it publishes a distorted HERFD (chapter 3), and nothing flags it. A pedestal-vs-frame check for a drifting detector is also missing.</li>
<li><b>Thresholds (4).</b> Portal Processing form has an 'ADU thresholds' field and presets; <code>02_tune_rixs</code> tunes them against the first pass. <i>Missing:</i> the grain histogram with its photon peak is only in the notebooks, and nothing warns when the photon peak is near the xray gate.</li>
<li><b>Curvature (5).</b> Automatic everywhere; shown in <code>01_one_image</code>. <i>Missing:</i> a correction along the energy axis. The one implemented cannot change a spectrum, and the sideways bend seen in the elastic data broadens every line.</li>
<li><b>Spectrum and ROI (6-7).</b> Live tab: image over spectrum on a shared axis, with a draggable ROI band. Processing: ROI centre and width fields.</li>
<li><b>Whole scan (8-9).</b> Automatic first pass reduces every measurement with default settings; Processing runs <code>tender-herfd</code>; I0 is a checkbox. <i>Missing:</i> the RIXS map with the band drawn is not shown in the portal Viewer, which is the one picture that shows whether the automatic ROI centre is sensible.</li>
<li><b>Normalisation (10).</b> Every <code>tender-herfd</code> CSV carries norm/flat from the same routine, and the Viewer plots it. <i>Missing:</i> Tender-appropriate ranges. With larch's defaults this sulfate scan's edge step is about twice too large, and the portal has no way to set the ranges for a Tender job.</li>
<li><b>Averaging (11).</b> <code>04_average_compare</code> averages repeats with the spread band; the portal's generic Average/Merge skills work on the CSVs (and can align on E0). <i>Missing:</i> energy alignment in <code>tender_analysis.average</code> (these two repeats are offset by more than an eV), a Tender-aware average in the portal that keeps the ROI the same across repeats, and a counting-statistics comparison like chapter 11's.</li>
<li><b>Calibration (12).</b> Notebook <code>calibration.ipynb</code> only. <i>Missing:</i> the portal's <code>tender-xes</code> writes a pixel axis; there is no way yet to attach a saved calibration to a portal job, or to use the elastic peak inside each XES spectrum as a check.</li>
<li><b>Interpretation (13).</b> <code>04_average_compare</code> compares compounds on a normalised scale. <i>Missing:</i> reference spectra and a linear-combination fit, an incident-energy calibration against a reference, and XES comparison on an energy axis in the portal.</li>
</ul>"""
    paras.append("<b>What the chemcat portal and the notebooks already offer, step by "
                 "step, and what is missing.</b>" + offer)
    S.add("13. Interpretation", paras, figs13, box(
        "Normalisation ranges (HERFD); area normalisation window and calibration (XES).",
        "<code>export.normalize_mu</code>; <code>index_beamtime</code> + "
        "<code>Measurement.pipeline</code> for XES; <code>ElasticCalibration.to_energy</code>.",
        "Over-reading small differences: compare only spectra reduced with the same "
        "settings, and check scan-to-scan stability before calling a difference chemistry."))
    S.cite(ta.Measurement.pipeline, ta.Measurement._dark_background)
    S.cite("chemcat: chemcatal/static/live.js, chemcatal/live/service.py, "
           "chemcatal/templates/processing/_tender.html, chemcatal/skills/tender.py, "
           "chemcatal/worker/firstpass.py, chemcatal/notebook/packs/tender/*.ipynb")

    meta = dict(
        generated=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        lib_rev=git_rev(REPO), bundled=os.path.abspath(bundled),
        real=os.path.abspath(real) if real else "(none)",
        seconds=time.time() - t_start,
    )
    write_html(S, meta, out)
    return facts


# ---------------------------------------------------------------- html

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--bg:#fcfcfb;--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
 font-size:17px;line-height:1.55}
main{max-width:980px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:30px;margin:0 0 6px;line-height:1.2}
h2{font-size:23px;margin:56px 0 12px;padding-top:12px;border-top:1px solid var(--rule)}
p{margin:0 0 14px;max-width:46em}
.sub{color:var(--ink2);margin-bottom:22px}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:0.88em;
 background:#f0efeb;padding:1px 4px;border-radius:3px}
figure{margin:18px 0 22px}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:4px;background:#fff}
figcaption{color:var(--ink2);font-size:14.5px;margin-top:6px;max-width:52em}
.box{border:1px solid var(--rule);border-left:4px solid var(--accent);background:#fff;
 border-radius:4px;padding:10px 16px;margin:18px 0 8px;font-size:15.5px}
.box div{margin:4px 0}
.box b.k{display:inline-block;min-width:8.5em;color:var(--ink2);font-weight:600}
.skip{color:var(--ink2);font-style:italic}
nav ol{columns:2;font-size:15.5px;padding-left:22px}
nav a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--rule)}
ul.kv{list-style:none;padding-left:0}
ul.kv li{margin:2px 0}
ul.offer li{margin:6px 0}
.sources{margin-top:48px;color:var(--ink2);font-size:13px;line-height:1.5;word-break:break-word}
"""


def write_html(S, meta, out):
    parts = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>Tender X-ray data, step by step</title>",
             f"<style>{CSS}</style></head><body><main>",
             "<h1>Tender X-ray data, step by step</h1>",
             "<p class='sub'>SSRL BL 6-2a. What the data looks like at every step, from one "
             "raw detector image to a spectrum you can interpret. Every figure below was "
             f"computed from real data by <code>tender_analysis</code> "
             f"({meta['lib_rev']}) on {meta['generated']}.</p>",
             "<p>How to read it: each chapter says what happens, shows it, and ends with a "
             "box naming the <b>setting</b> that controls the step (with its default), the "
             "<b>function</b> that does it, and what to <b>watch out for</b>. Chapters 1 to "
             "10 follow one sulfur K-edge RIXS series of a Na₂SO₄ pellet; 11 to 13 add a "
             "second series, silver XES of three samples, and an energy calibration.</p>",
             "<nav><ol>"]
    for i, ch in enumerate(S.chapters):
        parts.append(f"<li><a href='#c{i}'>{html.escape(ch['title'].split('. ', 1)[-1])}</a></li>")
    parts.append("</ol></nav>")
    for i, ch in enumerate(S.chapters):
        parts.append(f"<section id='c{i}'><h2>{html.escape(ch['title'])}</h2>")
        if ch["skipped"]:
            parts.append(f"<p class='skip'>{ch['skipped']}</p></section>")
            continue
        figs = list(ch["figs"])
        paras = list(ch["paras"])
        # An int in the paragraph list places that figure there; otherwise the
        # first figure follows the first paragraph and the rest close the text.
        if figs and not any(isinstance(p, int) for p in paras):
            paras.insert(1, 0)
            paras.extend(range(1, len(figs)))
        used = set()
        for p in paras:
            if isinstance(p, int):
                parts.append(figs[p])
                used.add(p)
            elif p.lstrip().startswith(("<ul", "<ol")):
                parts.append(p)
            else:
                parts.append(f"<p>{p}</p>")
        parts.extend(f for i, f in enumerate(figs) if i not in used)
        b = ch["box"]
        if b:
            parts.append("<div class='box'>"
                         f"<div><b class='k'>Setting</b> {b['setting']}</div>"
                         f"<div><b class='k'>Function</b> {b['function']}</div>"
                         f"<div><b class='k'>Watch out for</b> {b['watch']}</div></div>")
        parts.append("</section>")
    parts.append(
        f"<p class='sources'><b>Sources.</b> Data: {html.escape(meta['bundled'])}; "
        f"{html.escape(meta['real'])}. Code (tender-analysis {meta['lib_rev']}): "
        + "; ".join(html.escape(s) for s in S.sources)
        + f". Generated by docs/story/make_story.py in {meta['seconds']:.0f} s.</p>")
    parts.append("</main></body></html>")
    text = "\n".join(parts)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundled", default=os.path.join(REPO, "data", "Na2SO4"),
                    help="RIXS series directory (default: data/Na2SO4)")
    ap.add_argument("--real", default=None,
                    help="optional folder with Na2SO4_pellet/, Ag2S/, AgNO3/, P12S/, elastic_BN/")
    ap.add_argument("--out", default="tender-data-story.html", help="output HTML file")
    args = ap.parse_args(argv)
    facts = build(args.bundled, args.real, args.out)
    for k, v in facts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
