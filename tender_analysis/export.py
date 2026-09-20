"""chemcat-shaped serialization of a reduction result.

The analysis output already has the right SHAPE for the SSRL chemcat portal's
1D XAS surfaces -- ``RIXSResult.E``/``HERFD`` map one-to-one onto chemcat's
``energy``/``mu`` -- but not the right serialization: ``RIXSResult.save_txt``
writes whitespace ``energy_eV HERFD TFY`` and carries no ``norm``/``flat``.
This module closes that gap without touching the math. ``save_txt`` stays
exactly as it is for notebook users; these writers are the portal's format.

    write_xas_csv   energy_eV, mu, norm, flat, tfy   <- HERFD / RIXS
    write_xes_csv   pixel, intensity                 <- XES, uncalibrated
                    emission_eV, pixel, intensity    <- XES + ElasticCalibration

Both produce the same container: ``#``-commented provenance lines, then one
header row, then the data. That container is what chemcat's ``normalize`` /
``average`` / ``merge`` skills already emit, so the portal viewer needs ONE
reader for every one of these products.

Normalization runs through chemcat's own ``xas_core`` when it is importable
(inside a chemcat worker it always is, so the portal's numbers come from the
portal's code), then xraylarch directly, then a vendored fallback. Which path
ran is recorded in the header -- a spectrum normalized by the fallback is not
interchangeable with a larch-normalized one, and the file should say so.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["write_xas_csv", "write_xes_csv", "normalize_mu", "NormalizationResult"]


class NormalizationResult(dict):
    """``{e0, edge_step, norm, flat, method}`` with attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


# --------------------------------------------------------------- normalization

def _larch_normalize(energy: np.ndarray, mu: np.ndarray, overrides: dict):
    from larch import Group
    from larch.xafs import pre_edge

    g = Group(energy=np.asarray(energy, dtype=float), mu=np.asarray(mu, dtype=float))
    pre_edge(g.energy, g.mu, group=g, **overrides)
    return (np.asarray(g.norm, dtype=float), np.asarray(g.flat, dtype=float),
            float(g.e0), float(g.edge_step))


def _chemcat_normalize(energy: np.ndarray, mu: np.ndarray, overrides: dict):
    """chemcat's own xas_core.xas.xas_normalize, run on a one-column frame.

    Preferred over calling larch directly so that, in the deployment that
    matters, the norm/flat columns are produced by the same function the
    portal's own `normalize` skill uses -- not by a second implementation of
    the same idea that could drift from it.
    """
    import pandas as pd
    from chemcatal.xas_core.xas import xas_normalize

    df = pd.DataFrame({"energy_eV": np.asarray(energy, dtype=float),
                       "mu": np.asarray(mu, dtype=float)})
    meta = xas_normalize(df, overrides or None)
    return (df["norm"].to_numpy(), df["flat"].to_numpy(),
            float(meta["e0"]), float(meta["edge_step"]))


def _fallback_normalize(energy: np.ndarray, mu: np.ndarray):
    """Vendored pre-edge subtraction + edge-step normalization, no larch.

    Deliberately the simple version of what ``pre_edge`` does: e0 at the
    maximum of dmu/dE, a straight line fit to the pre-edge region and a
    quadratic to the post-edge region, edge step = their separation at e0.
    ``flat`` removes the post-edge curvature above e0 the way larch's does.

    This exists so the writers still produce five columns on a machine without
    xraylarch (the plain checkout, a notebook, a test); it is NOT a
    reimplementation intended to agree with larch to the last digit, which is
    why the header names the method that ran.
    """
    e = np.asarray(energy, dtype=float)
    m = np.asarray(mu, dtype=float)
    good = np.isfinite(e) & np.isfinite(m)
    e, m = e[good], m[good]
    if e.size < 5:
        raise ValueError("too few finite points to normalize")
    order = np.argsort(e)
    e, m = e[order], m[order]

    de = np.diff(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        deriv = np.where(de != 0, np.diff(m) / de, np.nan)
    if not np.any(np.isfinite(deriv)):
        raise ValueError("mu(E) has no usable derivative; cannot locate e0")
    e0 = float(e[1:][int(np.nanargmax(deriv))])

    span = float(e[-1] - e[0])
    pre = e < e0 - 0.05 * span
    post = e > e0 + 0.10 * span
    # Degrade the fits rather than fail: a XANES-only scan can leave the
    # post-edge region with too few points for a quadratic.
    pre_c = np.polyfit(e[pre], m[pre], 1) if pre.sum() >= 2 else np.array([0.0, float(m[0])])
    pre_line = np.polyval(pre_c, e)
    # The post-edge polynomial is fit to the PRE-EDGE-SUBTRACTED signal, as
    # larch's pre_edge does. Fitting it to raw mu instead puts the pre-edge
    # slope into both curves, which leaves `flat` sitting below 1.
    sub = m - pre_line
    if post.sum() >= 3:
        post_c = np.polyfit(e[post], sub[post], 2)
    elif post.sum() >= 1:
        post_c = np.array([0.0, 0.0, float(np.mean(sub[post]))])
    else:
        post_c = np.array([0.0, 0.0, float(sub[-1])])

    post_curve = np.polyval(post_c, e)
    edge_step = float(np.polyval(post_c, e0))
    if not np.isfinite(edge_step) or edge_step <= 0:
        # A non-positive edge step means the fit is meaningless -- typically a
        # scan with no flat pre-edge region, where the pre-edge line is fit
        # through the edge itself and extrapolates above the post-edge. Say so
        # rather than dividing by it: the caller turns this into NaN columns
        # and a header line, and `mu` is unaffected either way.
        raise ValueError(
            f"edge step is {edge_step:.4g}; the scan has no usable pre-edge "
            "region to normalize against")
    norm = sub / edge_step
    # flat: above e0, subtract the post-edge curvature's departure from its
    # value at e0, so the normalized spectrum sits flat at 1.
    flat = norm.copy()
    above = e >= e0
    flat[above] -= (post_curve[above] - edge_step) / edge_step
    return norm, flat, e0, edge_step


def normalize_mu(energy, mu, overrides: dict | None = None) -> NormalizationResult:
    """``norm``/``flat``/``e0``/``edge_step`` for a mu(E) pair.

    Tries chemcat's ``xas_core``, then xraylarch, then the vendored fallback,
    and reports which one produced the numbers as ``method``.
    """
    overrides = overrides or {}
    for method, fn in (("chemcatal.xas_core", _chemcat_normalize),
                       ("xraylarch pre_edge", _larch_normalize)):
        try:
            norm, flat, e0, edge_step = fn(energy, mu, overrides)
        except Exception:  # noqa: BLE001 - ImportError, or larch refusing the data
            continue
        return NormalizationResult(norm=norm, flat=flat, e0=e0,
                                   edge_step=edge_step, method=method)
    try:
        norm, flat, e0, edge_step = _fallback_normalize(energy, mu)
    except ValueError as exc:
        # A spectrum too short or too flat to normalize is still a spectrum.
        # Refusing to write it would lose the measurement over a derived
        # column; NaN says "not computed" without inventing a number, and the
        # header says why.
        nan = np.full(np.shape(energy), np.nan)
        return NormalizationResult(norm=nan, flat=nan, e0=float("nan"),
                                   edge_step=float("nan"),
                                   method=f"not normalized ({exc})")
    return NormalizationResult(norm=norm, flat=flat, e0=e0, edge_step=edge_step,
                               method="tender_analysis fallback (no xraylarch)")


# -------------------------------------------------------------------- writing

def _provenance(meta: dict | None, extra: dict) -> list[str]:
    """Header comment lines: the result's own meta first, then our additions.

    Values are flattened to one line each -- a header line with an embedded
    newline would break out of the comment block and be read as data.
    """
    lines: list[str] = []
    for src in (meta or {}, extra):
        for key, value in src.items():
            if value is None:
                continue
            text = " ".join(str(value).split())
            if len(text) > 400:
                text = text[:397] + "..."
            lines.append(f"{key}: {text}")
    return lines


def _write_csv(path, columns: "dict[str, np.ndarray]", header_lines: list[str]) -> str:
    """The chemcat CSV container, written by chemcat's own writer when it is
    importable so the two products are byte-identical in shape."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        import pandas as pd
        from chemcatal.xas_core.specio import write_csv
    except ImportError:
        with open(path, "w") as fh:
            for line in header_lines:
                fh.write(f"# {line}\n")
            fh.write(",".join(columns) + "\n")
            rows = zip(*(np.asarray(v) for v in columns.values()))
            for row in rows:
                fh.write(",".join(f"{v:.8g}" for v in row) + "\n")
        return str(path)
    write_csv(path, pd.DataFrame(columns), header_lines)
    return str(path)


def write_xas_csv(result, path, meta: dict | None = None,
                  overrides: dict | None = None, source: str | None = None) -> str:
    """Write a :class:`RIXSResult` as a chemcat-shaped XAS CSV.

    Columns ``energy_eV, mu, norm, flat, tfy`` -- ``mu`` is the HERFD line-out
    and ``tfy`` the total-fluorescence-yield trace recorded alongside it, so
    nothing the result carries in 1D is dropped. The product is
    indistinguishable in shape from what chemcat's own ``normalize`` skill
    writes, which is what lets the portal viewer plot it with no new format.

    Returns the path written.
    """
    energy = np.asarray(result.E, dtype=float)
    mu = np.asarray(result.HERFD, dtype=float)
    tfy = np.asarray(result.TFY, dtype=float)
    norm_result = normalize_mu(energy, mu, overrides)

    extra = {
        "source": source or "tender_analysis OnePotRIXS.herfd()",
        "mu": "HERFD band line-out",
        "energy axis": "incident energy (mono, eV)",
        "columns": "energy_eV mu norm flat tfy",
        "normalization": norm_result.method,
        "pre_edge": (f"e0={norm_result.e0:.3f} edge_step={norm_result.edge_step:.6g}"
                     if np.isfinite(norm_result.e0) else None),
        "central_pix": getattr(result, "central_pix", None),
    }
    if overrides:
        extra["pre_edge overrides"] = " ".join(f"{k}={v}" for k, v in overrides.items())
    columns = {"energy_eV": energy, "mu": mu, "norm": norm_result.norm,
               "flat": norm_result.flat, "tfy": tfy}
    return _write_csv(path, columns,
                      _provenance(meta if meta is not None else getattr(result, "meta", None),
                                  extra))


def write_xes_csv(result, path, calibration=None, meta: dict | None = None,
                  source: str | None = None) -> str:
    """Write an :class:`XESResult` spectrum as a CSV in the same container.

    ``XESResult.spectrum()`` is indexed by DETECTOR PIXEL, not by energy, so
    without a calibration this writes ``pixel, intensity`` and the x axis is a
    pixel number. Given an :class:`~tender_analysis.calibration.ElasticCalibration`
    it writes ``emission_eV, pixel, intensity`` with the energy first, which is
    the column order the portal reader treats as an x axis. Either way the
    container is the one ``write_xas_csv`` uses -- one reader covers both.
    """
    spectrum = np.asarray(result.spectrum(), dtype=float)
    pixels = np.arange(spectrum.size, dtype=float)
    extra = {
        "source": source or "tender_analysis OnePot.run()",
        "kind": "XES emission spectrum",
    }
    if calibration is not None:
        columns = {"emission_eV": np.asarray(calibration.to_energy(pixels), dtype=float),
                   "pixel": pixels, "intensity": spectrum}
        extra["columns"] = "emission_eV pixel intensity"
        extra["energy axis"] = "emission energy from ElasticCalibration"
        extra["energy_calibration"] = repr(calibration)
    else:
        columns = {"pixel": pixels, "intensity": spectrum}
        extra["columns"] = "pixel intensity"
        extra["energy axis"] = ("detector pixel -- UNCALIBRATED; supply an "
                                "ElasticCalibration for an emission-energy axis")
    return _write_csv(path, columns,
                      _provenance(meta if meta is not None else getattr(result, "meta", None),
                                  extra))
