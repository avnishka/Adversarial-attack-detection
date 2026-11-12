"""Small file helpers shared by the scripts. Image IDs are always strings (file stems)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_index(folder: str | Path) -> dict[str, Path]:
    """Map image_id (file stem) -> path for every image in a folder (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS}


def read_ids(csv_path: str | Path) -> list[str]:
    df = pd.read_csv(csv_path, dtype={"image_id": str})
    return df["image_id"].tolist()


def read_scores(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={"image_id": str})
