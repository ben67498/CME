from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

from .utils_time import parse_datetime_from_filename

ALL_CHANNELS = [
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "aia1600",
    "aia1700",
    "hmi_bx",
    "hmi_by",
    "hmi_bz",
    "hmi_m",
]
DEFAULT_HDF5_PLUGIN_DIR = "/usr/local/hdf5/lib/plugin"


def _is_plugin_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "plugin" in msg or "can't open directory" in msg or "hdf5" in msg


def ensure_hdf5_plugin_path() -> str:
    plugin_dir = DEFAULT_HDF5_PLUGIN_DIR
    try:
        os.makedirs(plugin_dir, exist_ok=True)
    except PermissionError:
        plugin_dir = tempfile.mkdtemp(prefix="hdf5_plugin_")

    try:
        import hdf5plugin  # type: ignore

        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
        return hdf5plugin.PLUGIN_PATH
    except Exception:  # noqa: BLE001
        os.environ.setdefault("HDF5_PLUGIN_PATH", plugin_dir)
        return os.environ["HDF5_PLUGIN_PATH"]


def safe_open_dataset(path: Path):
    errors = []
    for engine in ("h5netcdf", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine, decode_times=True)
        except OSError as err:
            errors.append(f"{engine}: {err}")
            if _is_plugin_error(err):
                ensure_hdf5_plugin_path()
                try:
                    return xr.open_dataset(path, engine=engine, decode_times=True)
                except OSError as retry_err:
                    errors.append(f"{engine} retry: {retry_err}")
            continue
        except Exception as err:  # noqa: BLE001
            errors.append(f"{engine}: {err}")
            continue
    raise RuntimeError(f"Unable to open netCDF dataset at {path}. Tried h5netcdf and netcdf4. Details: {' | '.join(errors)}")


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    a = np.clip(a, lo, hi)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _downsample_then_read(ds, channel: str, target_resize: int) -> np.ndarray:
    da = ds[channel]
    shape = da.shape
    if len(shape) < 2:
        raise ValueError(f"Channel {channel} is not 2D: shape={shape}")
    y_dim, x_dim = da.dims[-2], da.dims[-1]
    orig = int(max(shape[-2], shape[-1]))
    step = max(1, round(orig / target_resize))
    da_small = da.isel({y_dim: slice(0, None, step), x_dim: slice(0, None, step)})
    return da_small.astype("float32").values


def load_core_sdo_stack(paths: list[Path], channels: list[str], resize: int = 256, use_diff: bool = False):
    ensure_hdf5_plugin_path()
    xs, times = [], []
    for p in sorted(paths):
        ds = safe_open_dataset(p)
        arrs = []
        for ch in channels:
            if ch not in ds:
                ds.close()
                raise KeyError(f"Channel {ch} missing in {p}")
            try:
                arr = _downsample_then_read(ds, ch, resize)
            except OSError as err:
                if not _is_plugin_error(err):
                    ds.close()
                    raise
                ds.close()
                ensure_hdf5_plugin_path()
                ds = safe_open_dataset(p)
                arr = _downsample_then_read(ds, ch, resize)
            arrs.append(_norm(arr))

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
