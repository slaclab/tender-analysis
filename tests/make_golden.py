"""Regenerate ``tests/data/golden_na2so4.npz`` -- the refactor regression fixture.

Run from the repo root ONLY on a revision whose reduction behaviour is the
reference (first captured before ``reduce_frame`` was factored out of
``extract_signal``; recaptured after the upstream curvature-axis fix, b39b2a7,
which intentionally changes every curvature-corrected output):

    python tests/make_golden.py

1D arrays are stored verbatim; 2D/3D arrays as a sha256 of their bytes plus
their sum, so the fixture stays small.
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tender_analysis import OnePot, OnePotRIXS, SifFile  # noqa: E402

DATA = os.path.join(ROOT, "data", "Na2SO4")
OUT = os.path.join(HERE, "data", "golden_na2so4.npz")


def digest(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode()
                          + a.tobytes()).hexdigest()


def xes_outputs(prefix: str, result, out: dict) -> None:
    out[f"{prefix}_spectrum"] = result.spectrum()
    out[f"{prefix}_t"] = np.atleast_1d(np.asarray(result.t, dtype=float))
    for name in ("signal", "corr_signal", "raw", "total_counts_raw",
                 "total_counts_signal", "total_counts_common_mode"):
        arr = getattr(result, name)
        out[f"{prefix}_{name}_sha"] = np.array(digest(arr))
        out[f"{prefix}_{name}_sum"] = np.array(float(np.sum(arr)))
    if result.histograms is not None:
        for key, h in result.histograms.items():
            out[f"{prefix}_hist_{key}_sha"] = np.array(digest(h))


def compute() -> dict:
    files = sorted(glob.glob(os.path.join(DATA, "*.sif")))
    dark = [p for p in files if p.endswith("_dark.sif")][0]
    one = [p for p in files if not p.endswith("_dark.sif")][40]
    bcg = SifFile(dark).data.mean(axis=0)

    out: dict = {"onepot_file": np.array(os.path.basename(one))}
    xes_outputs("onepot", OnePot([one], bcg=bcg, histograms=True).run(), out)
    xes_outputs("evol", OnePot([one], bcg=bcg, evolution=True).run(), out)

    rixs = OnePotRIXS(files, use_dark_as_background=True).herfd(central_pix=None, n=7)
    out["rixs_E"] = rixs.E
    out["rixs_HERFD"] = rixs.HERFD
    out["rixs_TFY"] = rixs.TFY
    out["rixs_central_pix"] = np.array(rixs.central_pix)
    out["rixs_map_sha"] = np.array(digest(rixs.rixs_map))
    out["rixs_map_sum"] = np.array(float(rixs.rixs_map.sum()))
    return out


if __name__ == "__main__":
    np.savez_compressed(OUT, **compute())
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
