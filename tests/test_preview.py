import numpy as np
import pytest

from tender_analysis import OnePot, SifFile
from tender_analysis.preview import block_downsample, preview, stretch_uint8


@pytest.fixture(scope="module")
def bcg(na2so4_dark):
    return SifFile(na2so4_dark).data.mean(axis=0)


def test_spectrum_matches_onepot(na2so4_scan, na2so4_dark, bcg):
    path = na2so4_scan[40]
    ref = OnePot([path], bcg=bcg).run()
    p = preview(path, dark=na2so4_dark)
    assert np.allclose(p["spectrum"], ref.spectrum())
    assert np.allclose(p["t"], ref.t)
    assert p["spectrum"].shape == (2048,)
    assert p["row_profile"].shape == (512,)


def test_thresholds_and_fixed_curvature(na2so4_scan, bcg):
    path = na2so4_scan[5]
    th = [80, 150, 1500]
    ref = OnePot([path], bcg=bcg, threshold=th).run()
    p = preview(path, dark=bcg, thresholds=th, curvature_t=ref.t, frame=0)
    assert np.allclose(p["spectrum"], ref.spectrum())


def test_images_and_metadata(na2so4_scan, na2so4_dark):
    p = preview(na2so4_scan[40], dark=na2so4_dark, downsample=(4, 2))
    for key in ("raw", "bkg_sub", "events"):
        assert p[key].dtype == np.uint8 and p[key].shape == (128, 1024)
        lo, hi = p["limits"][key]
        assert hi > lo
    assert p["events"].any()
    assert p["shape"] == (1, 512, 2048)
    assert p["n_events"] > 0
    assert np.isfinite(p["I0"]) and np.isfinite(p["energy"])


def test_helpers():
    img = np.arange(24, dtype=float).reshape(4, 6)
    assert block_downsample(img, (2, 3)).shape == (2, 2)
    assert block_downsample(img, (2, 3))[0, 0] == img[:2, :3].mean()
    sparse = np.zeros((100, 100))
    sparse[3, 3] = 7
    img8, (lo, hi) = stretch_uint8(sparse)
    assert img8[3, 3] == 255 and hi == 7
