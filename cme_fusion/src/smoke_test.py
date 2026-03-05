from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .config import SmokeConfig
from .dataset_fusion import FusionDataset
from .download_cdaw import download_cdaw_text
from .download_core_sdo import download_core_sdo_sample
from .download_lasco import download_lasco_c2
from .load_core_sdo import load_core_sdo_stack
from .parse_cdaw import parse_univ_all
from .preprocess_lasco import load_lasco_stack
from .train import run_train
from .utils_time import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", default="aia171,aia193,aia304,hmi_m")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--pos_window_hours", type=float, default=1.0)
    p.add_argument("--neg_gap_hours", type=float, default=6.0)
    p.add_argument("--label_mode", choices=["frame", "forecast"], default="frame")
    p.add_argument("--forecast_horizon_hours", type=float, default=2.0)
    p.add_argument("--max_align_minutes", type=int, default=60)
    args = p.parse_args()

    cfg = SmokeConfig()
    set_seed(42)
    data_dir = cfg.data_dir
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    sdo_files = download_core_sdo_sample(data_dir / "core_sdo")
    lasco_files = download_lasco_c2(cfg.lasco_date, data_dir / "lasco", n_files=cfg.lasco_num_files)
    cdaw_path = download_cdaw_text(data_dir / "cdaw" / "univ_all.txt")

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    x_sdo, t_sdo = load_core_sdo_stack(sdo_files, channels=channels, resize=args.resize, use_diff=True)
    x_lasco, t_lasco = load_lasco_stack(lasco_files, resize=args.resize, use_diff=True)
    cme = parse_univ_all(cdaw_path)

    if len(cme) == 0:
        raise RuntimeError("Parsed zero CME rows")

    # Smoke-only expansion to help positives
    center = pd.Timestamp("2011-01-20T02:00:00Z")
    window = cme[(cme.first_c2_datetime_utc >= center - pd.Timedelta(days=2)) & (cme.first_c2_datetime_utc <= center + pd.Timedelta(days=2))]
    cme_use = window if len(window) else cme

    def make_ds(x_sdo_in, t_sdo_in, x_lasco_in, t_lasco_in):
        return FusionDataset(
            x_sdo=x_sdo_in,
            x_lasco=x_lasco_in,
            sdo_times=t_sdo_in,
            lasco_times=t_lasco_in,
            cme_df=cme_use,
            max_align_minutes=args.max_align_minutes,
            label_mode=args.label_mode,
            pos_window_hours=args.pos_window_hours,
            neg_gap_hours=args.neg_gap_hours,
            forecast_horizon_hours=args.forecast_horizon_hours,
        )

    ds = make_ds(x_sdo, t_sdo, x_lasco, t_lasco)
    ys = [int(ds[i][2].item()) for i in range(len(ds))]
    if len(set(ys)) < 2 and len(cme_use):
        ctimes = pd.DatetimeIndex(cme_use.first_c2_datetime_utc)
        neg_t = ctimes[0] + pd.Timedelta(hours=12)
        while ((abs((ctimes - neg_t).total_seconds()) / 3600.0) < args.neg_gap_hours).any():
            neg_t = neg_t + pd.Timedelta(hours=6)
        x_sdo_aug = torch.cat([x_sdo, x_sdo[:1]], dim=0)
        t_sdo_aug = t_sdo.append(pd.DatetimeIndex([neg_t]))
        x_lasco_aug = torch.cat([x_lasco, x_lasco[:1]], dim=0)
        t_lasco_aug = t_lasco.append(pd.DatetimeIndex([neg_t]))
        ds = make_ds(x_sdo_aug, t_sdo_aug, x_lasco_aug, t_lasco_aug)

    if len(ds) < 2:
        raise RuntimeError("Too few aligned labeled samples for smoke test")

    m = run_train(ds, run_dir=run_dir, epochs=args.epochs, batch_size=args.batch_size)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(x_lasco[0].numpy(), cmap="gray"); plt.title("LASCO")
    plt.subplot(1, 2, 2); plt.imshow(x_sdo[0][0].numpy(), cmap="magma"); plt.title(f"SDO {channels[0]}")
    plt.tight_layout(); plt.savefig(run_dir / "debug_grid.png", dpi=120)

    print(m)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
