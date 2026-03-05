from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class ItemMeta:
    sdo_time: str
    lasco_time: str
    nearest_cme_time: str | None
    delta_minutes: float | None


def assign_labels(sample_times: pd.DatetimeIndex, cme_times: pd.DatetimeIndex, label_mode: str = "frame", pos_window_hours: float = 1.0, neg_gap_hours: float = 6.0, forecast_horizon_hours: float = 2.0):
    ys, nearest, delta = [], [], []
    for t in sample_times:
        if len(cme_times) == 0:
            ys.append(0); nearest.append(pd.NaT); delta.append(np.nan); continue
        dmins = np.array([(ct - t).total_seconds() / 60.0 for ct in cme_times])
        abs_h = np.abs(dmins) / 60.0
        imin = abs_h.argmin()
        nearest.append(cme_times[imin])
        delta.append(float(dmins[imin]))
        if label_mode == "frame":
            pos = np.any(abs_h <= pos_window_hours)
        else:
            dh = dmins / 60.0
            pos = np.any((dh > 0) & (dh <= forecast_horizon_hours))
        too_close_neg = np.any(abs_h < neg_gap_hours)
        y = 1 if pos else (0 if not too_close_neg else -1)
        ys.append(y)
    return np.array(ys), nearest, delta


def align_lasco_sdo(sdo_times: pd.DatetimeIndex, lasco_times: pd.DatetimeIndex, max_align_minutes: int = 60):
    pairs = []
    for i, st in enumerate(sdo_times):
        if len(lasco_times) == 0:
            continue
        dmins = np.array([abs((lt - st).total_seconds()) / 60.0 for lt in lasco_times])
        j = int(dmins.argmin())
        if dmins[j] <= max_align_minutes:
            pairs.append((i, j))
    return pairs


class FusionDataset(Dataset):
    def __init__(self, x_sdo, x_lasco, sdo_times, lasco_times, cme_df: pd.DataFrame, max_align_minutes=60, label_mode="frame", pos_window_hours=1.0, neg_gap_hours=6.0, forecast_horizon_hours=2.0):
        pairs = align_lasco_sdo(sdo_times, lasco_times, max_align_minutes=max_align_minutes)
        self.items = []
        cme_times = pd.DatetimeIndex(cme_df["first_c2_datetime_utc"]) if len(cme_df) else pd.DatetimeIndex([])
        idx_times = pd.DatetimeIndex([sdo_times[i] for i, _ in pairs])
        ys, nearest, delta = assign_labels(idx_times, cme_times, label_mode=label_mode, pos_window_hours=pos_window_hours, neg_gap_hours=neg_gap_hours, forecast_horizon_hours=forecast_horizon_hours)
        for (k, (i, j)) in enumerate(pairs):
            if ys[k] == -1:
                continue
            self.items.append((x_lasco[j], x_sdo[i], int(ys[k]), ItemMeta(str(sdo_times[i]), str(lasco_times[j]), str(nearest[k]) if pd.notna(nearest[k]) else None, float(delta[k]) if np.isfinite(delta[k]) else None)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        xl, xs, y, m = self.items[idx]
        return xl.unsqueeze(0), xs, torch.tensor(y, dtype=torch.float32), m
