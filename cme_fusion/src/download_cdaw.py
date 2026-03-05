from __future__ import annotations

from pathlib import Path

import requests

CANDIDATE_URLS = [
    "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt",
    "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/univ_all.txt",
]


def download_cdaw_text(output_path: Path, cdaw_url: str | None = None, timeout: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    urls = [cdaw_url] if cdaw_url else CANDIDATE_URLS
    last_err = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with output_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return output_path
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Unable to download CDAW file from {urls}") from last_err
