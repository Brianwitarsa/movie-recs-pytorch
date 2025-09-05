#!/usr/bin/env python3
"""
Load artifacts/mf.pt and print top-K movie titles for a user id (uid).
Usage:
  python -m scripts.recommend --uid 0 --k 10
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from recs.models import MF
from recs.utils import load_item_titles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/mf.pt")
    ap.add_argument("--movies", default="data/movies.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--uid", type=int, required=True, help="internal uid (from prepare_data)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = MF(ckpt["n_users"], ckpt["n_items"], ckpt["dim"])
    model.load_state_dict(ckpt["model_state"]); model.eval()

    all_items = torch.arange(ckpt["n_items"], dtype=torch.long)
    u = torch.full((len(all_items),), args.uid, dtype=torch.long)

    with torch.no_grad():
        scores = model(u, all_items)

    topk_idx = torch.topk(scores, args.k).indices.tolist()

    # Map internal iid -> raw movieId -> title
    # We need the mapping from prepare_data step:
    # Build it from train.csv (contains uid,iid, plus original ids are not saved).
    # Simpler: reconstruct by reading ratings.csv and re-indexing like before.
    ratings = pd.read_csv("data/ratings.csv")
    # Recreate iid mapping in the same way (sorted unique)
    iid_map = {m:i for i,m in enumerate(sorted(ratings["movieId"].unique()))}
    inv_iid_map = {v:k for k,v in iid_map.items()}

    id2title_raw = load_item_titles(args.movies)
    titles = []
    for iid in topk_idx:
        raw_mid = inv_iid_map[iid]
        titles.append(id2title_raw.get(raw_mid, f"<movieId {raw_mid}>"))

    print(f"Top {args.k} for uid={args.uid}:")
    for rank, title in enumerate(titles, 1):
        print(f"{rank:2d}. {title}")

if __name__ == "__main__":
    main()
