from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from astropy.io import fits

from .utils_time import parse_datetime_from_filename


def extract_fits_time(path: Path) -> pd.Timestamp:
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        date_obs = hdr.get("DATE-OBS") or hdr.get("DATE_OBS")
        time_obs = hdr.get("TIME-OBS")
    if date_obs:
        ts = f"{date_obs} {time_obs}" if time_obs and "T" not in str(date_obs) else str(date_obs)
        return pd.Timestamp(ts, tz="UTC")
    ts = parse_datetime_from_filename(path.name)
    if ts is None:
        raise ValueError(f"Cannot parse timestamp from {path}")
    return ts


def robust_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def load_lasco_stack(paths: list[Path], resize: int = 256, use_diff: bool = False):
    images, times = [], []
    for p in sorted(paths):
        with fits.open(p) as hdul:
            arr = np.array(hdul[0].data, dtype=np.float32)
        arr = robust_norm(arr)
        t = torch.from_numpy(arr)[None, None, ...]
        t = F.interpolate(t, size=(resize, resize), mode="bilinear", align_corners=False)
        images.append(t[0, 0])
        times.append(extract_fits_time(p))
    stack = torch.stack(images, dim=0)
    if use_diff:
        diff = torch.zeros_like(stack)
        diff[1:] = stack[1:] - stack[:-1]
        stack = diff
    return stack, pd.DatetimeIndex(times)
