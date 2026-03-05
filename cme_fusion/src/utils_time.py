from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_utc_datetime(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def parse_datetime_from_filename(name: str) -> pd.Timestamp | None:
    import re

    pats = [r"(\d{8})_(\d{4})", r"(\d{6})_(\d{4})", r"(\d{8})(\d{6})"]
    for p in pats:
        m = re.search(p, name)
        if not m:
            continue
        a, b = m.groups()
        if len(a) == 6:
            dt = datetime.strptime(a + b, "%y%m%d%H%M")
        elif len(b) == 4:
            dt = datetime.strptime(a + b, "%Y%m%d%H%M")
        else:
            dt = datetime.strptime(a + b, "%Y%m%d%H%M%S")
        return pd.Timestamp(dt, tz=timezone.utc)
    return None
