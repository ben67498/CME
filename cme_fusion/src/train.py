from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .eval import binary_metrics
from .models import FusionCNN


def collate(batch):
    xl, xs, y, m = zip(*batch)
    return torch.stack(xl), torch.stack(xs), torch.stack(y), list(m)


def run_train(dataset, run_dir: Path, epochs=2, batch_size=2, seed=42):
    run_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    n_tr = max(1, int(n * 0.6))
    n_va = max(1, int(n * 0.2)) if n >= 3 else 0
    n_te = n - n_tr - n_va
    gen = torch.Generator().manual_seed(seed)
    splits = random_split(dataset, [n_tr, n_va, n_te], generator=gen)
    tr, va, te = splits
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionCNN(sdo_channels=dataset[0][1].shape[0]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, collate_fn=collate) if len(va) else None
    best = -1.0
    for _ in range(epochs):
        model.train()
        for xl, xs, y, _ in tr_loader:
            xl, xs, y = xl.to(device), xs.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(xl, xs), y)
            loss.backward(); opt.step()
        if va_loader:
            model.eval(); probs=[]; ys=[]
            with torch.no_grad():
                for xl, xs, y, _ in va_loader:
                    p = torch.sigmoid(model(xl.to(device), xs.to(device))).cpu().numpy()
                    probs.extend(p.tolist()); ys.extend(y.numpy().tolist())
            m = binary_metrics(ys, probs)
            if m["f1"] >= best:
                best = m["f1"]
                torch.save(model.state_dict(), run_dir / "best.pt")
    te_loader = DataLoader(te if len(te) else tr, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval(); probs=[]; ys=[]; table=[]
    with torch.no_grad():
        for xl, xs, y, meta in te_loader:
            p = torch.sigmoid(model(xl.to(device), xs.to(device))).cpu().numpy()
            probs.extend(p.tolist()); ys.extend(y.numpy().tolist())
            for yi, pi, mi in zip(y.numpy().tolist(), p.tolist(), meta):
                table.append({"timestamp": mi.sdo_time, "y_true": yi, "y_pred_prob": pi, "nearest_cme_time": mi.nearest_cme_time, "delta_minutes": mi.delta_minutes})
    metrics = binary_metrics(ys, probs)
    metrics["n_samples"] = len(dataset)
    metrics["pred_table"] = table
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
