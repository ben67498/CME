import pandas as pd

from src.dataset_fusion import align_lasco_sdo, assign_labels


def test_alignment_and_labels():
    sdo = pd.DatetimeIndex(["2011-01-20T01:00:00Z", "2011-01-20T02:00:00Z"])
    las = pd.DatetimeIndex(["2011-01-20T01:10:00Z", "2011-01-20T02:30:00Z"])
    pairs = align_lasco_sdo(sdo, las, max_align_minutes=40)
    assert pairs == [(0, 0), (1, 1)]
    cme = pd.DatetimeIndex(["2011-01-20T01:30:00Z"])
    ys, _, _ = assign_labels(sdo, cme, label_mode="frame", pos_window_hours=1.0, neg_gap_hours=6.0)
    assert ys[0] == 1
