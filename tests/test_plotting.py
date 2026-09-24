import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from tender_analysis import OnePot, OnePotRIXS, SifFile  # noqa: E402
from tender_analysis import plotting  # noqa: E402
from tender_analysis.average import average_series  # noqa: E402
from tender_analysis.preview import preview  # noqa: E402


@pytest.fixture(scope="module")
def xes(na2so4_scan, na2so4_dark):
    bcg = SifFile(na2so4_dark).data.mean(axis=0)
    return OnePot([na2so4_scan[60]], bcg=bcg, histograms=True).run()


@pytest.fixture(scope="module")
def rixs(na2so4_scan, na2so4_dark):
    return OnePotRIXS(na2so4_scan[::6] + [na2so4_dark],
                      use_dark_as_background=True).herfd(n=7)


def _ok(fig, tmp_path, name):
    assert isinstance(fig, Figure)
    fig.savefig(tmp_path / f"{name}.png", dpi=60)


def test_image_with_spectrum(na2so4_scan, na2so4_dark, tmp_path):
    p = preview(na2so4_scan[60], dark=na2so4_dark)
    fig = plotting.image_with_spectrum(p, central_pix=1305, n=7)
    ax_img, ax_spec = fig.axes[:2]
    assert ax_img.get_shared_x_axes().joined(ax_img, ax_spec)
    assert ax_img.get_xlim() == ax_spec.get_xlim()
    _ok(fig, tmp_path, "image")
    _ok(plotting.image_with_spectrum(np.random.rand(64, 256)), tmp_path, "image_arr")


def test_histogram_and_curvature(xes, tmp_path):
    from tender_analysis import Thresholds
    _ok(plotting.adu_histogram(xes.histograms, Thresholds.from_input(None)), tmp_path, "h")
    _ok(plotting.adu_histogram(xes.histograms, key=None), tmp_path, "h_all")
    _ok(plotting.curvature_before_after(xes), tmp_path, "curv")


def test_rixs_and_herfd(rixs, tmp_path):
    _ok(plotting.rixs_map(rixs), tmp_path, "map")
    _ok(plotting.herfd_overlay([rixs, rixs], labels=["a", "b"], tfy=True), tmp_path, "ov")
    avg = average_series([rixs, rixs])
    _ok(plotting.averaged(avg), tmp_path, "avg")
    _ok(plotting.averaged(avg.mean, avg.std, avg.E), tmp_path, "avg_arr")


def test_too_many_series(rixs):
    with pytest.raises(ValueError):
        plotting.herfd_overlay([rixs] * 9)
