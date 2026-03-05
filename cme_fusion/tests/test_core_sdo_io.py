from pathlib import Path

import numpy as np
import xarray as xr

from src import load_core_sdo
from src.load_core_sdo import ensure_hdf5_plugin_path, load_core_sdo_stack, safe_open_dataset


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


def test_ensure_hdf5_plugin_path_sets_env_and_dir_exists():
    plugin_path = ensure_hdf5_plugin_path()
    assert plugin_path
    assert "HDF5_PLUGIN_PATH" in __import__("os").environ
    default_dir = Path(load_core_sdo.DEFAULT_HDF5_PLUGIN_DIR)
    env_dir = Path(__import__("os").environ["HDF5_PLUGIN_PATH"])
    assert default_dir.exists() or env_dir.exists()


def test_safe_open_dataset_falls_back_to_netcdf4(monkeypatch, tmp_path: Path):
    p = tmp_path / "fake.nc"
    p.write_bytes(b"placeholder")

    opened = []

    def fake_open_dataset(path, engine=None, decode_times=True):
        opened.append(engine)
        if engine == "h5netcdf":
            raise OSError("can't open directory (/usr/local/hdf5/lib/plugin)")
        return xr.Dataset({"ok": (("x",), np.array([1], dtype=np.int32))})

    monkeypatch.setattr(load_core_sdo.xr, "open_dataset", fake_open_dataset)
    ds = safe_open_dataset(p)
    assert opened[:2] == ["h5netcdf", "h5netcdf"]
    assert opened[-1] == "netcdf4"
    assert "ok" in ds
    ds.close()
