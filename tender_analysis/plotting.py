"""Figure builders for notebooks and reports.

matplotlib is an OPTIONAL dependency (the ``[notebook]`` extra): it is
imported inside each builder, so ``import tender_analysis.plotting`` works
without it and only calling a builder needs it. Every builder returns a
``matplotlib.figure.Figure`` built with the object API (no pyplot state), so it
is safe on a headless ``Agg`` backend.

Styling: one fixed categorical order for series identity, single-hue
sequential colormaps for magnitude, neutral ink for text and threshold/ROI
annotations, one y-scale per axes (a second measure gets its own panel, never
a twin axis).
"""

from __future__ import annotations

import numpy as np

__all__ = ["image_with_spectrum", "adu_histogram", "curvature_before_after",
           "rixs_map", "herfd_overlay", "averaged"]

#: Categorical series colours, assigned in this order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
ROI = "#52514e"        # neutral band: annotation, not a data series
IMAGE_CMAP = "gray"    # detector images: lightness = counts
MAP_CMAP = "Blues"     # single-hue sequential for RIXS maps


def _figure(figsize, **kw):
    from matplotlib.figure import Figure
    return Figure(figsize=figsize, layout="constrained", **kw)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def _band(central_pix, n):
    """``(lo, hi)`` pixel edges of the HERFD band, as OnePotRIXS.herfd slices it."""
    lo = central_pix - n // 2
    return lo - 0.5, lo + n - 0.5


def _series_color(i: int) -> str:
    if i >= len(SERIES):
        raise ValueError(f"{i + 1} series: at most {len(SERIES)} are drawn in one "
                         "panel -- facet or fold the rest into 'Other'")
    return SERIES[i]


def image_with_spectrum(data, central_pix=None, n: int = 7, image: str = "events",
                        spectrum=None, title: str | None = None):
    """Detector image stacked over its spectrum, sharing the dispersive x axis.

    ``data`` is a :func:`tender_analysis.preview.preview` dict (``image`` picks
    ``"raw"``/``"bkg_sub"``/``"events"``; its ``downsample`` maps image columns
    back to pixels) or a 2D array (full resolution, then pass ``spectrum``).
    With ``central_pix`` the HERFD band ``[central_pix - n//2, +n)`` is shaded
    on both panels.
    """
    if isinstance(data, dict):
        img = np.asarray(data[image])
        spec = np.asarray(data["spectrum"] if spectrum is None else spectrum)
        fr, fc = data.get("downsample", (1, 1))
    else:
        img = np.asarray(data)
        spec = np.asarray(img.sum(axis=0) if spectrum is None else spectrum)
        fr, fc = 1, 1
    rows = img.shape[0] * fr
    cols = img.shape[1] * fc

    fig = _figure((9, 5.5))
    ax_img, ax_spec = fig.subplots(2, 1, sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    ax_img.imshow(img, cmap=IMAGE_CMAP, aspect="auto", interpolation="nearest",
                  extent=(-0.5, cols - 0.5, rows - 0.5, -0.5))
    ax_img.set_ylabel("detector row", color=INK)
    ax_img.tick_params(colors=INK_2, labelcolor=INK_2)

    x = np.arange(spec.size)
    ax_spec.plot(x, spec, color=SERIES[0], linewidth=1.5)
    _style(ax_spec)
    ax_spec.set_xlabel("dispersive pixel", color=INK)
    ax_spec.set_ylabel("counts", color=INK)
    ax_spec.set_xlim(-0.5, max(cols, spec.size) - 0.5)

    if central_pix is not None:
        lo, hi = _band(int(central_pix), int(n))
        for ax in (ax_img, ax_spec):
            ax.axvspan(lo, hi, color=ROI, alpha=0.18, linewidth=0)
            ax.axvline(lo, color=ROI, linewidth=0.8)
            ax.axvline(hi, color=ROI, linewidth=0.8)
        ax_spec.annotate(f"ROI {int(central_pix)} ± {n // 2}", xy=(hi, 1), xycoords=("data", "axes fraction"),
                         xytext=(4, -4), textcoords="offset points", va="top",
                         color=INK_2, fontsize=9)
    fig.suptitle(title or (data.get("path", "") if isinstance(data, dict) else ""),
                 color=INK, fontsize=10)
    return fig


def adu_histogram(hist, thresholds=None, key: str = "bkg_free", xmax: int | None = None):
    """ADU histogram (log counts) with threshold lines.

    ``hist`` is ``XESResult.histograms`` (a dict; ``key`` picks one, or pass
    ``key=None`` to overlay all), or a 1D count array indexed by ADU.
    ``thresholds`` is a :class:`~tender_analysis.pipeline.Thresholds` or its
    ``[bcg_cutoff, low, xray, hi]`` array.
    """
    if isinstance(hist, dict):
        curves = hist if key is None else {key: hist[key]}
    else:
        curves = {"counts": hist}
    fig = _figure((8, 4))
    ax = fig.subplots()
    top = 0
    for i, (name, h) in enumerate(curves.items()):
        h = np.asarray(h, dtype=float)
        nz = np.flatnonzero(h)
        top = max(top, int(nz[-1]) + 1 if nz.size else 0)
        ax.step(np.arange(h.size), np.where(h > 0, h, np.nan), where="mid",
                color=_series_color(i), linewidth=1.2, label=name)
    if len(curves) > 1:
        ax.legend(frameon=False, labelcolor=INK)
    ax.set_yscale("log")

    if thresholds is not None:
        vals = (thresholds.as_array() if hasattr(thresholds, "as_array")
                else np.atleast_1d(np.asarray(thresholds, dtype=float)))
        names = ["bcg_cutoff", "low", "xray", "hi"][-len(vals):] if len(vals) <= 4 else \
            [f"t{i}" for i in range(len(vals))]
        for name, v in zip(names, vals):
            ax.axvline(v, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(f"{name} {v:g}", xy=(v, 1), xycoords=("data", "axes fraction"),
                        xytext=(3, -3), textcoords="offset points", rotation=90,
                        va="top", color=INK_2, fontsize=8)
        top = max(top, int(np.max(vals) * 1.1))
    ax.set_xlim(0, xmax if xmax is not None else max(top, 10))
    _style(ax)
    ax.set_xlabel("ADU", color=INK)
    ax.set_ylabel("pixels / grains", color=INK)
    return fig


def curvature_before_after(result, rows=None):
    """Summed signal before and after curvature correction (``XESResult``).

    Two image panels sharing both axes, plus the row profile of each so the
    straightening shows as a narrower peak. ``rows`` optionally crops
    ``(start, stop)`` detector rows.
    """
    before = result.signal.sum(axis=0) if result.signal.ndim == 3 else result.signal
    after = result.corr_signal if result.corr_signal is not None else before
    sl = slice(*rows) if rows is not None else slice(None)
    fig = _figure((10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
    vmax = np.percentile(before[sl], 99.5) or None
    y0 = sl.start or 0
    for ax, img, name in ((ax0, before, "before"), (ax1, after, "after")):
        ax.imshow(img[sl], cmap=IMAGE_CMAP, aspect="auto", vmin=0, vmax=vmax,
                  interpolation="nearest",
                  extent=(-0.5, img.shape[1] - 0.5, y0 + img[sl].shape[0] - 0.5, y0 - 0.5))
        ax.set_title(f"{name} curvature correction", color=INK, fontsize=10)
        ax.set_xlabel("dispersive pixel", color=INK)
        ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax0.set_ylabel("detector row", color=INK)

    axp = fig.add_subplot(gs[1, :])
    r = np.arange(before.shape[0])[sl]
    axp.plot(r, before[sl].sum(axis=1), color=SERIES[0], linewidth=1.5, label="before")
    axp.plot(r, after[sl].sum(axis=1), color=SERIES[1], linewidth=1.5, label="after")
    axp.legend(frameon=False, labelcolor=INK)
    _style(axp)
    axp.set_xlabel("detector row", color=INK)
    axp.set_ylabel("counts", color=INK)
    t = getattr(result, "t", None)
    if t is not None and not np.isscalar(t):
        fig.suptitle("t = " + np.array2string(np.asarray(t), precision=3), color=INK_2,
                     fontsize=9)
    return fig


def rixs_map(result, central_pix=None, n: int | None = None, pixels=None):
    """RIXS map (incident energy vs dispersive pixel) with the HERFD band.

    ``result`` is a :class:`RIXSResult`; ``central_pix``/``n`` default to the
    result's own (``n`` from ``meta['herfd_band_width_px']``). ``pixels``
    optionally crops ``(start, stop)``.
    """
    m = np.asarray(result.rixs_map)
    E = np.asarray(result.E)
    cp = result.central_pix if central_pix is None else central_pix
    n = n if n is not None else int((result.meta or {}).get("herfd_band_width_px", 7))
    p0, p1 = pixels if pixels is not None else (0, m.shape[0])
    fig = _figure((8, 6))
    ax = fig.subplots()
    # Incident-energy steps are rarely uniform (dense across the edge), so
    # draw cells between energy midpoints rather than an evenly spaced image.
    if E.size > 1:
        mid = (E[1:] + E[:-1]) / 2
        e_edges = np.concatenate([[E[0] - (mid[0] - E[0])], mid, [E[-1] + (E[-1] - mid[-1])]])
    else:
        e_edges = np.array([E[0] - 0.5, E[0] + 0.5])
    p_edges = np.arange(p0, p1 + 1) - 0.5
    im = ax.pcolormesh(p_edges, e_edges, m[p0:p1].T, cmap=MAP_CMAP, shading="flat",
                       rasterized=True)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("counts" + (" / I0" if (result.meta or {}).get("i0_corrected") else ""),
                 color=INK)
    if cp is not None and cp >= 0:
        lo, hi = _band(int(cp), int(n))
        ax.axvspan(lo, hi, color=ROI, alpha=0.25, linewidth=0)
        ax.annotate(f"HERFD band {int(cp)} ± {n // 2}", xy=(hi, 1),
                    xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", va="top", color=INK, fontsize=9)
    ax.set_xlabel("dispersive pixel (emission)", color=INK)
    ax.set_ylabel("incident energy (eV)", color=INK)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    return fig


def herfd_overlay(results, labels=None, tfy: bool = False, normalize: bool = False):
    """Overlay HERFD line-outs of several results (≤ 8), one colour each.

    ``tfy=True`` adds a second panel below with the TFY traces (its own
    y-scale -- never a twin axis). ``normalize=True`` scales each trace to
    its maximum so shapes compare across counting times.
    """
    results = list(results)
    labels = list(labels) if labels is not None else [str(i) for i in range(len(results))]
    fig = _figure((8, 6 if tfy else 4))
    axes = fig.subplots(2, 1, sharex=True) if tfy else [fig.subplots()]

    def _y(v):
        v = np.asarray(v, dtype=float)
        return v / np.nanmax(v) if normalize and np.nanmax(v) else v

    for i, (r, lab) in enumerate(zip(results, labels)):
        c = _series_color(i)
        axes[0].plot(r.E, _y(r.HERFD), color=c, linewidth=1.8, label=lab)
        if tfy:
            axes[1].plot(r.E, _y(r.TFY), color=c, linewidth=1.8, label=lab)
    axes[0].set_ylabel("HERFD" + (" (norm.)" if normalize else ""), color=INK)
    if tfy:
        axes[1].set_ylabel("TFY" + (" (norm.)" if normalize else ""), color=INK)
    for ax in axes:
        _style(ax)
    axes[-1].set_xlabel("incident energy (eV)", color=INK)
    if len(results) > 1:
        axes[0].legend(frameon=False, labelcolor=INK)
    return fig


def averaged(mean, std=None, E=None, n=None, label: str = "mean"):
    """Averaged line-out with a ±1σ band.

    Pass an :class:`~tender_analysis.average.AveragedSeries` as ``mean`` or the
    arrays (``mean``, ``std``, ``E``). With ``n`` (per-point series count) the
    count is drawn in a small panel beneath.
    """
    if hasattr(mean, "mean") and hasattr(mean, "E"):
        avg = mean
        mean, std, E, n = avg.mean, avg.std, avg.E, avg.n
    mean = np.asarray(mean, dtype=float)
    E = np.arange(mean.size) if E is None else np.asarray(E, dtype=float)
    fig = _figure((8, 5 if n is not None else 4))
    if n is not None:
        ax, axn = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    else:
        ax, axn = fig.subplots(), None
    if std is not None:
        std = np.asarray(std, dtype=float)
        ax.fill_between(E, mean - std, mean + std, color=SERIES[0], alpha=0.2,
                        linewidth=0, label="±1σ")
    ax.plot(E, mean, color=SERIES[0], linewidth=1.8, label=label)
    if std is not None:
        ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    ax.set_ylabel("HERFD", color=INK)
    if axn is not None:
        axn.step(E, np.asarray(n), where="mid", color=INK_2, linewidth=1.2)
        _style(axn)
        axn.set_ylabel("n", color=INK)
        axn.set_xlabel("incident energy (eV)", color=INK)
    else:
        ax.set_xlabel("incident energy (eV)", color=INK)
    return fig
