import json
from pathlib import Path
import pandas as pd

DATA = Path("data")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

ratings = pd.read_csv(DATA/"ratings.csv")  # cols: userId,movieId,rating,timestamp

# Re-index to contiguous IDs for embeddings
uid_map = {u:i for i,u in enumerate(sorted(ratings["userId"].unique()))}
iid_map = {m:i for i,m in enumerate(sorted(ratings["movieId"].unique()))}
ratings["uid"] = ratings["userId"].map(uid_map)
ratings["iid"] = ratings["movieId"].map(iid_map)

# Sort by time within each user, then assign splits per-user (LOO-style)
ratings = ratings.sort_values(["userId","timestamp"]).reset_index(drop=True)
ratings["split"] = "train"

def mark_splits(df):
    n = len(df)
    if n >= 3:
        df.loc[df.index[-1], "split"] = "test"
        df.loc[df.index[-2], "split"] = "valid"
        # rest stay "train"
    elif n == 2:
        df.loc[df.index[-1], "split"] = "test"
        # first stays "train"; no valid
    else:  # n == 1
        # only "train"
        pass
    return df

ratings = ratings.groupby("userId", group_keys=False).apply(mark_splits)

# Save
out_cols = ["uid","iid","rating","timestamp"]
ratings.loc[ratings.split=="train", out_cols].to_csv(DATA/"train.csv", index=False)
ratings.loc[ratings.split=="valid", out_cols].to_csv(DATA/"valid.csv", index=False)
ratings.loc[ratings.split=="test",  out_cols].to_csv(DATA/"test.csv",  index=False)

meta = {
    "n_users": len(uid_map),
    "n_items": len(iid_map),
    "n_train": int((ratings.split=="train").sum()),
    "n_valid": int((ratings.split=="valid").sum()),
    "n_test":  int((ratings.split=="test").sum()),
}
with open(ART/"data_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))