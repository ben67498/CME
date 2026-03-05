from pathlib import Path

import numpy as np
from astropy.io import fits

from src.preprocess_lasco import extract_fits_time, load_lasco_stack


def test_lasco_fits_timestamp_and_load(tmp_path: Path):
    p = tmp_path / "test.fts"
    hdu = fits.PrimaryHDU(np.random.rand(32, 32).astype("float32"))
    hdu.header["DATE-OBS"] = "2011-01-20T01:00:00"
    hdu.writeto(p)
    ts = extract_fits_time(p)
    assert ts.year == 2011
    x, t = load_lasco_stack([p], resize=16, use_diff=True)
    assert x.shape == (1, 16, 16)
    assert len(t) == 1
