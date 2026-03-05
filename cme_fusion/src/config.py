from dataclasses import dataclass
from pathlib import Path


@dataclass
class SmokeConfig:
    data_dir: Path = Path("data")
    run_dir: Path = Path("runs/smoke")
    lasco_date: str = "2011-01-20"
    lasco_num_files: int = 24
    resize: int = 256
    channels: tuple = ("aia171", "aia193", "aia304", "hmi_m")
    epochs: int = 2
    batch_size: int = 2
    max_align_minutes: int = 60
    pos_window_hours: float = 1.0
    neg_gap_hours: float = 6.0
    label_mode: str = "frame"
    forecast_horizon_hours: float = 2.0
