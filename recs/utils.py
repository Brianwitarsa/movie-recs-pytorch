# recs/utils.py
import pandas as pd
from pathlib import Path

def load_item_titles(movies_csv: str | Path) -> dict[int, str]:
    df = pd.read_csv(movies_csv)  # expects columns: movieId,title
    # Build rawId -> title
    id2title_raw = dict(zip(df["movieId"], df["title"]))
    return id2title_raw
