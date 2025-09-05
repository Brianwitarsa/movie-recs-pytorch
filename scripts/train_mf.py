#!/usr/bin/env python3
"""
Train a minimal MF model on data/train.csv and eval on data/valid.csv.

Usage:
  python scripts/train_mf.py --epochs 5 --dim 64
"""

import argparse
from pathlib import Path
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from recs.models import MF


class RatingsDS(Dataset):
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        self.u = torch.tensor(df["uid"].values, dtype=torch.long)
        self.i = torch.tensor(df["iid"].values, dtype=torch.long)
        self.r = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self): return len(self.r)
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]


def rmse(model, loader, loss_fn):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for u, i, r in loader:
            pred = model(u, i)
            loss = loss_fn(pred, r)
            total_loss += loss.item() * len(r)
            n += len(r)
    return math.sqrt(total_loss / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/train.csv")
    ap.add_argument("--valid", type=str, default="data/valid.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--artifacts", type=str, default="artifacts/mf.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train_ds = RatingsDS(Path(args.train))
    valid_ds = RatingsDS(Path(args.valid))

    n_users = int(max(train_ds.u.max(), valid_ds.u.max()).item()) + 1
    n_items = int(max(train_ds.i.max(), valid_ds.i.max()).item()) + 1

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    model = MF(n_users=n_users, n_items=n_items, dim=args.dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Training MF: users={n_users} items={n_items} dim={args.dim}")
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for u, i, r in train_loader:
            opt.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            opt.step()
            running += loss.item() * len(r)
            seen += len(r)
        train_rmse = math.sqrt(running / max(seen, 1))
        valid_rmse = rmse(model, valid_loader, loss_fn)
        print(f"Epoch {ep:02d} | train RMSE {train_rmse:.4f} | valid RMSE {valid_rmse:.4f}")

    # Save checkpoint
    Path(args.artifacts).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "dim": args.dim,
        },
        args.artifacts,
    )
    print(f"Saved checkpoint → {args.artifacts}")


if __name__ == "__main__":
    main()
