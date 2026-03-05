from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "nasa-ibm-ai4science/core-sdo"
SMOKE_FILES = [
    "infer_data/20110120_0100.nc",
    "infer_data/20110120_0200.nc",
    "infer_data/20110120_0300.nc",
]


def download_core_sdo_sample(out_dir: Path, files: list[str] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = files or SMOKE_FILES
    paths = []
    for f in files:
        p = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f, local_dir=str(out_dir))
        paths.append(Path(p))
    try:
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="scalers.yaml", local_dir=str(out_dir))
    except Exception:  # noqa: BLE001
        pass
    return paths
