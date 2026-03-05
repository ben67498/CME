from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float("nan")


def parse_univ_all(path: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^\s*(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:?\d{0,2})")
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        date_s, time_s = m.groups()
        if len(time_s) == 5:
            time_s += ":00"
        dt = pd.Timestamp(f"{date_s} {time_s}", tz="UTC")
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        central = _to_float(nums[3]) if len(nums) > 3 else np.nan
        width = _to_float(nums[4]) if len(nums) > 4 else np.nan
        speed = _to_float(nums[5]) if len(nums) > 5 else np.nan
        rows.append(
            {
                "first_c2_datetime_utc": dt,
                "central_pa_deg": central,
                "width_deg": width,
                "linear_speed_kms": speed,
                "raw": line,
            }
        )
    return pd.DataFrame(rows)
