import random

import numpy as np
import pytest

from tender_analysis import OnePotRIXS
from tender_analysis.live import HerfdAccumulator


@pytest.fixture(scope="module")
def reference(na2so4_files):
    pot = OnePotRIXS(na2so4_files, use_dark_as_background=True)
    auto = pot.herfd(central_pix=None, n=7)
    fixed = pot.herfd(central_pix=auto.central_pix + 3, n=3, i0_corr=False)
    return auto, fixed


@pytest.fixture(scope="module")
def acc(na2so4_scan, na2so4_dark):
    a = HerfdAccumulator(central_pix=None, n=7, dark=na2so4_dark)
    shuffled = list(na2so4_scan)
    random.Random(0).shuffle(shuffled)   # add order must not matter
    for p in shuffled:
        assert a.add(p) is not None
    return a


def _same(res, ref):
    assert res.central_pix == ref.central_pix
    assert np.array_equal(res.E, ref.E)
    assert np.allclose(res.HERFD, ref.HERFD)
    assert np.allclose(res.TFY, ref.TFY)
    assert np.allclose(res.rixs_map, ref.rixs_map)


def test_matches_onepotrixs(acc, reference):
    _same(acc.result(), reference[0])


def test_set_roi_reslices(acc, reference, monkeypatch):
    auto, fixed = reference
    acc.set_roi(auto.central_pix + 3, n=3)
    acc.i0_corr = False
    # no file may be re-read: the cached spectra carry the whole result
    monkeypatch.setattr("tender_analysis.live.SifFile", None)
    try:
        _same(acc.result(), fixed)
    finally:
        acc.set_roi(None, n=7)
        acc.i0_corr = True


def test_dark_adopted_when_added_first(na2so4_scan, na2so4_dark):
    a = HerfdAccumulator()
    assert a.add(na2so4_dark) is None
    assert a.dark_path == na2so4_dark
    info = a.add(na2so4_scan[0])
    assert info["energy"] > 2000 and np.isfinite(info["I0"])
    assert a.add(na2so4_scan[0]) is None      # duplicate ignored
    assert len(a) == 1


def test_empty_result_raises():
    with pytest.raises(ValueError):
        HerfdAccumulator().result()
