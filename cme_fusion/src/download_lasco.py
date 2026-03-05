from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


UMBRA_BASE = "https://umbra.nascom.nasa.gov/pub/lasco_level05/level_05/"
NRL_BASE = "https://lasco-www.nrl.navy.mil/pub/lasco_level05/level_05/"


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def _list_fts_urls(base_dir_url: str) -> list[str]:
    s = _session()
    r = s.get(base_dir_url, timeout=30)
    r.raise_for_status()
    names = re.findall(r'href="([^"]+\.fts(?:\.gz)?)"', r.text, flags=re.IGNORECASE)
    return [urljoin(base_dir_url, n) for n in names]


def _download_files(urls: list[str], out_dir: Path, n_files: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _session()
    paths = []
    for u in urls[:n_files]:
        p = out_dir / Path(u).name
        if not p.exists():
            with s.get(u, stream=True, timeout=60) as r:
                r.raise_for_status()
                with p.open("wb") as f:
                    for c in r.iter_content(8192):
                        if c:
                            f.write(c)
        paths.append(p)
    return paths


def download_lasco_c2(date_str: str, out_dir: Path, n_files: int = 24) -> list[Path]:
    yymmdd = date_str[2:].replace("-", "")
    b1 = f"{UMBRA_BASE}{yymmdd}/c2/"
    b2 = f"{NRL_BASE}{yymmdd}/c2/"
    errs = []
    for b in [b1, b2]:
        try:
            urls = _list_fts_urls(b)
            if not urls:
                raise RuntimeError(f"No .fts files found at {b}")
            return _download_files(urls, out_dir, n_files=n_files)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{b}: {e}")
    raise RuntimeError("LASCO download failed on both backends: " + " | ".join(errs))
