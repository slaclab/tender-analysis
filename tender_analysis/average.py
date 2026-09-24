"""Averaging repeated RIXS/HERFD series.

A sample/line is often measured as several RIXS series (``..._RIXS_01_*``,
``..._RIXS_02_*``: same sample and emission line, different ``series_index``).
:func:`average_series` puts their line-outs on one incident-energy grid and
returns the point-wise mean, standard deviation and contributing count;
:func:`group_repeats` finds the repeats in a :func:`index_beamtime` result, and
:func:`normalize_average` runs the mean through :func:`export.normalize_mu`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .export import normalize_mu
from .pipeline import RIXSResult

__all__ = ["AveragedSeries", "average_series", "group_repeats", "normalize_average"]


@dataclass
class AveragedSeries:
    """Point-wise average of several line-outs on a common energy grid.

    ``mean``/``std``/``n`` are the averaged signal (``HERFD`` unless another
    ``key`` was asked for); ``tfy_mean``/``tfy_std`` the same for TFY when every
    input carried one. ``n`` counts the series covering each grid point (a
    series contributes only inside its own energy range).
    """

    E: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n: np.ndarray
    tfy_mean: np.ndarray | None = None
    tfy_std: np.ndarray | None = None
    labels: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_result(self) -> RIXSResult:
        """A :class:`RIXSResult` (``HERFD=mean``, ``TFY=tfy_mean``) for the
        existing writers, e.g. :func:`export.write_xas_csv`. It carries no map."""
        tfy = self.tfy_mean if self.tfy_mean is not None else np.full_like(self.mean, np.nan)
        return RIXSResult(E=self.E, HERFD=self.mean, TFY=tfy,
                          central_pix=int(self.meta.get("central_pix", -1)),
                          rixs_map=np.empty((0, self.E.size)), meta=dict(self.meta))


def _xy(item, key: str):
    """``(E, y, tfy_or_None, meta)`` from a RIXSResult-like or an ``(E, y)`` pair."""
    if isinstance(item, (tuple, list)):
        E, y = item[0], item[1]
        tfy = item[2] if len(item) > 2 else None
        meta = {}
    else:
        E, y = item.E, getattr(item, key)
        tfy = getattr(item, "TFY", None)
        meta = getattr(item, "meta", {}) or {}
    E = np.asarray(E, dtype=float)
    order = np.argsort(E, kind="stable")
    y = np.asarray(y, dtype=float)[order]
    tfy = None if tfy is None else np.asarray(tfy, dtype=float)[order]
    return E[order], y, tfy, meta


def _on_grid(grid, E, y):
    """Linear interpolation onto ``grid``; NaN outside ``[E.min(), E.max()]``.
    Grid points that ARE sample energies are taken verbatim (no rounding)."""
    if np.array_equal(grid, E):
        return y.copy()
    out = np.interp(grid, E, y)
    out[(grid < E[0]) | (grid > E[-1])] = np.nan
    return out


def average_series(results, energy=None, key: str = "HERFD", ddof: int = 0,
                   labels=None) -> AveragedSeries:
    """Average line-outs onto a common incident-energy grid.

    Parameters
    ----------
    results:
        :class:`RIXSResult` objects (or ``(E, y[, tfy])`` tuples).
    energy:
        The grid. ``None`` uses the first series' energies.
    key:
        Which attribute of each result to average (``"HERFD"`` or ``"TFY"``).
    ddof:
        Passed to the standard deviation (``0``: a single series has ``std=0``).
    labels:
        Optional names recorded on the result (default ``0..N-1``).
    """
    items = [_xy(r, key) for r in results]
    if not items:
        raise ValueError("average_series needs at least one series")
    grid = items[0][0] if energy is None else np.asarray(energy, dtype=float)

    ys = np.array([_on_grid(grid, E, y) for E, y, _, _ in items])
    n = np.sum(np.isfinite(ys), axis=0)

    def _stats(stack):
        with np.errstate(invalid="ignore", divide="ignore"):
            cnt = np.sum(np.isfinite(stack), axis=0)
            mean = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
            dev = np.nansum((stack - mean) ** 2, axis=0)
            std = np.where(cnt - ddof > 0, np.sqrt(dev / np.maximum(cnt - ddof, 1)), np.nan)
        if len(stack) == 1:
            mean = stack[0].copy()  # one series comes back exactly
        return mean, std

    mean, std = _stats(ys)
    tfy_mean = tfy_std = None
    if key != "TFY" and all(t is not None for _, _, t, _ in items):
        tfy_mean, tfy_std = _stats(np.array([_on_grid(grid, E, t)
                                             for E, _, t, _ in items]))

    first_meta = items[0][3]
    meta = {
        "pipeline": "average_series",
        "averaged": key,
        "n_series": len(items),
        "std_ddof": ddof,
        "energy_grid": "first series" if energy is None else "supplied",
    }
    if "central_pix" in first_meta:
        meta["central_pix"] = first_meta["central_pix"]
    labels = list(labels) if labels is not None else list(range(len(items)))
    meta["series"] = [str(x) for x in labels]
    return AveragedSeries(E=np.array(grid, dtype=float), mean=mean, std=std, n=n,
                          tfy_mean=tfy_mean, tfy_std=tfy_std, labels=labels, meta=meta)


def group_repeats(measurements) -> dict[tuple, list]:
    """Group RIXS :class:`~tender_analysis.dataset.Measurement` repeats.

    Returns ``{(sample, emission_line): [Measurement, ...]}`` ordered by
    ``series_index``; XES measurements are ignored. Accepts a
    ``BeamtimeIndex`` or any iterable of measurements.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for m in measurements:
        if m.kind == "RIXS":
            groups[(m.sample, m.emission_line)].append(m)
    return {k: sorted(v, key=lambda m: m.series_index or 0) for k, v in groups.items()}


def normalize_average(avg: AveragedSeries, overrides: dict | None = None):
    """``export.normalize_mu`` on the averaged mean (``norm``/``flat``/``e0``...).

    Only points covered by at least one series are passed in; the returned
    ``norm``/``flat`` are re-expanded to the full grid with NaN elsewhere.
    """
    ok = np.isfinite(avg.mean)
    res = normalize_mu(avg.E[ok], avg.mean[ok], overrides)
    for k in ("norm", "flat"):
        full = np.full(avg.E.shape, np.nan)
        full[ok] = res[k]
        res[k] = full
    return res
