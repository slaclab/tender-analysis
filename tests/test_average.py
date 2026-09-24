import numpy as np
import pytest

from tender_analysis import RIXSResult
from tender_analysis.average import (average_series, group_repeats,
                                     normalize_average)
from tender_analysis.dataset import Measurement


def _series(shift=0.0, scale=1.0):
    E = np.linspace(2460, 2540, 81) + shift
    herfd = scale * (1 / (1 + np.exp(-(E - 2482) / 1.5)) + 0.02 * (E - 2460) / 80)
    return RIXSResult(E=E, HERFD=herfd, TFY=2 * herfd, central_pix=1300,
                      rixs_map=np.zeros((4, E.size)), meta={"central_pix": 1300})


def test_single_series_unchanged():
    s = _series()
    avg = average_series([s])
    assert np.array_equal(avg.E, s.E)
    assert np.array_equal(avg.mean, s.HERFD)
    assert np.array_equal(avg.tfy_mean, s.TFY)
    assert np.all(avg.std == 0) and np.all(avg.n == 1)


def test_two_identical_copies():
    s = _series()
    avg = average_series([s, _series()])
    assert np.allclose(avg.mean, s.HERFD)
    assert np.allclose(avg.std, 0)
    assert np.all(avg.n == 2)


def test_offset_grid_and_coverage():
    a, b = _series(), _series(shift=10.0, scale=3.0)
    avg = average_series([a, b])
    assert np.all(avg.n[avg.E < b.E[0]] == 1)
    both = avg.n == 2
    assert both.any() and np.all(avg.std[both] > 0)
    assert np.allclose(avg.mean[~both], a.HERFD[~both])


def test_normalize_and_to_result():
    avg = average_series([_series(), _series(scale=1.1)])
    norm = normalize_average(avg)
    assert norm["norm"].shape == avg.E.shape
    assert np.isfinite(norm.e0)
    res = avg.to_result()
    assert np.array_equal(res.HERFD, avg.mean) and res.central_pix == 1300


def test_group_repeats():
    ms = [Measurement("RIXS", "S", "Ka", None, 2), Measurement("RIXS", "S", "Ka", None, 1),
          Measurement("RIXS", "T", "Ka", None, 1), Measurement("XES", "S", "Ka", 2500.0, None)]
    g = group_repeats(ms)
    assert set(g) == {("S", "Ka"), ("T", "Ka")}
    assert [m.series_index for m in g[("S", "Ka")]] == [1, 2]


def test_empty_raises():
    with pytest.raises(ValueError):
        average_series([])
