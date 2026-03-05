from pathlib import Path

import numpy as np
import xarray as xr

from src.load_core_sdo import load_core_sdo_stack


def test_core_sdo_loading(tmp_path: Path):
    p = tmp_path / "20110120_0100.nc"
    ds = xr.Dataset({
        "aia171": (("y", "x"), np.random.rand(16, 16).astype("float32")),
        "aia193": (("y", "x"), np.random.rand(16, 16).astype("float32")),
        "aia304": (("y", "x"), np.random.rand(16, 16).astype("float32")),
        "hmi_m": (("y", "x"), np.random.rand(16, 16).astype("float32")),
    })
    ds.to_netcdf(p, engine="h5netcdf")
    x, t = load_core_sdo_stack([p], channels=["aia171", "aia193", "aia304", "hmi_m"], resize=8)
    assert x.shape == (1, 4, 8, 8)
    assert len(t) == 1
