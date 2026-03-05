from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

from .utils_time import parse_datetime_from_filename

ALL_CHANNELS = ["aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600", "aia1700", "hmi_bx", "hmi_by", "hmi_bz", "hmi_m"]


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    a = np.clip(a, lo, hi)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def load_core_sdo_stack(paths: list[Path], channels: list[str], resize: int = 256, use_diff: bool = False):
    xs, times = [], []
    for p in sorted(paths):
        ds = xr.open_dataset(p, engine="h5netcdf")
        arrs = []
        for ch in channels:
            if ch not in ds:
                raise KeyError(f"Channel {ch} missing in {p}")
            arrs.append(_norm(np.asarray(ds[ch]).astype(np.float32)))
        x = np.stack(arrs, axis=0)
        t = torch.from_numpy(x)[None, ...]
        t = F.interpolate(t, size=(resize, resize), mode="bilinear", align_corners=False)
        xs.append(t[0])
        times.append(parse_datetime_from_filename(p.name))
        ds.close()
    x = torch.stack(xs, dim=0)
    if use_diff:
        d = torch.zeros_like(x)
        d[1:] = x[1:] - x[:-1]
        x = d
    return x, pd.DatetimeIndex(times)
