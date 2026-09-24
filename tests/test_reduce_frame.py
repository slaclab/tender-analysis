import numpy as np

from tender_analysis import SifFile, Thresholds, extract_signal
from tender_analysis.analyze import background_common_mode, reduce_frame


def test_reduce_frame_matches_extract_signal(na2so4_scan, na2so4_dark):
    sif = SifFile(na2so4_scan[10])
    bcg = SifFile(na2so4_dark).data.mean(axis=0)
    th = Thresholds.from_input(None)
    ref = extract_signal([sif], [th.low, th.xray, th.hi], bcg=bcg)

    frame, masks = reduce_frame(sif.frame(0), bcg, background_common_mode(bcg), th)
    assert np.array_equal(frame, ref.signal)
    assert set(masks) >= {"cosmic", "events", "grains"}
    for m in masks.values():
        assert m.dtype == bool and m.shape == frame.shape
    # grains is a subset of events; signal lives only on grains
    assert not (masks["grains"] & ~masks["events"]).any()
    assert not frame[~masks["grains"]].any()
    assert not frame[masks["cosmic"]].any()
