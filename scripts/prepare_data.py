
"""
Prepare MovieLens ratings for training:
- Load data/ratings.csv
- Reindex to contiguous uid/iid
- Per-user, time-aware proportional split (~80/10/10 with safeguards)
- Save data/train.csv, data/valid.csv, data/test.csv
- Save artifacts/data_meta.json with counts
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def split_user(df: pd.DataFrame, train_p: float = 0.8, valid_p: float = 0.1) -> pd.DataFrame:
    """
    Chronological split per user with approximate proportions.
    Ensures:
      - at least 1 train row when n>=1
      - when n>=3, ensures at least 1 test row
      - validation may be 0 when user is very sparse
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    # Edge cases first
    if n == 1:
        df["split"] = ["train"]
        return df
    if n == 2:
        df["split"] = ["train", "test"]
        return df

    # Proportional sizes
    n_train = max(1, int(round(n * train_p)))
    n_valid = int(round(n * valid_p))

    # Ensure we leave room for test
    if n_train >= n - 1:
        n_train = n - 2  # leave at least 1 for valid/test bucket
    # Fit valid within remaining
    remaining = n - n_train
    n_valid = min(n_valid, max(0, remaining - 1))  # ensure at least 1 for test
    n_test = n - n_train - n_valid
    if n_test <= 0:  # safety
        n_test = 1
        if n_valid > 0:
            n_valid -= 1
        else:
            n_train -= 1

    assert n_train + n_valid + n_test == n and n_train >= 1 and n_test >= 1

    df = df.copy()
    df["split"] = "train"
    if n_valid > 0:
        df.loc[n_train : n_train + n_valid - 1, "split"] = "valid"
    df.loc[n_train + n_valid :, "split"] = "test"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data", help="Directory containing ratings.csv")
    ap.add_argument("--artifacts-dir", type=str, default="artifacts", help="Where to write metadata")
    ap.add_argument("--train-prop", type=float, default=0.8, help="Approx train proportion per user")
    ap.add_argument("--valid-prop", type=float, default=0.1, help="Approx valid proportion per user")
    args = ap.parse_args()

    DATA = Path(args.data_dir)
    ART = Path(args.artifacts_dir)
    ART.mkdir(parents=True, exist_ok=True)

    ratings_path = DATA / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Could not find {ratings_path}. Place ratings.csv in {DATA}/")

    # Load ratings
    ratings = pd.read_csv(ratings_path)  # expected cols: userId,movieId,rating,timestamp
    required_cols = {"userId", "movieId", "rating", "timestamp"}
    missing = required_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv is missing columns: {missing}")

    # Reindex users/items to contiguous ids
    uid_map = {u: i for i, u in enumerate(sorted(ratings["userId"].unique()))}
    iid_map = {m: i for i, m in enumerate(sorted(ratings["movieId"].unique()))}
    ratings["uid"] = ratings["userId"].map(uid_map).astype("int64")
    ratings["iid"] = ratings["movieId"].map(iid_map).astype("int64")

    # Per-user chronological proportional split
    ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    ratings = ratings.groupby("userId", group_keys=False).apply(
        split_user, train_p=args.train_prop, valid_p=args.valid_prop
    )

    # Save splits
    out_cols = ["uid", "iid", "rating", "timestamp"]
    train_df = ratings.loc[ratings.split == "train", out_cols]
    valid_df = ratings.loc[ratings.split == "valid", out_cols]
    test_df = ratings.loc[ratings.split == "test", out_cols]

    (DATA / "train.csv").parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA / "train.csv", index=False)
    valid_df.to_csv(DATA / "valid.csv", index=False)
    test_df.to_csv(DATA / "test.csv", index=False)

    # Save metadata
    meta = {
        "n_users": int(len(uid_map)),
        "n_items": int(len(iid_map)),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "props_requested": {
            "train": args.train_prop,
            "valid": args.valid_prop,
            "test": round(1.0 - args.train_prop - args.valid_prop, 4),
        },
    }
    with open(ART / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
