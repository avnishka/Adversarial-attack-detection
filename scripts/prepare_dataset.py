"""Phase 3: build the road-scene subset of COCO val2017 and the image-level splits.

Keeps images with at least one road object (config dataset.road_category_ids), shuffles
with a fixed seed, keeps at most dataset.max_images, splits 20/30/50 by image ID, writes
splits/<split>.csv and copies images to dataset/clean/<split>/.

Usage (after unzipping val2017.zip and annotations_trainval2017.zip into dataset/raw/):
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --coco-images dataset/raw/val2017 \
        --coco-ann dataset/raw/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_path  # noqa: E402


def build_splits(ann: dict, road_ids: set[int], max_images: int, fractions: dict, seed: int) -> dict[str, pd.DataFrame]:
    images = {im["id"]: im for im in ann["images"]}
    road_counts: Counter = Counter()
    for a in ann["annotations"]:
        if a["category_id"] in road_ids:
            road_counts[a["image_id"]] += 1
    ids = np.array(sorted(road_counts))
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)[:max_images]

    n = len(ids)
    n_dev = int(round(n * fractions["dev"]))
    n_cal = int(round(n * fractions["calibration"]))
    parts = {"dev": ids[:n_dev], "calibration": ids[n_dev:n_dev + n_cal], "test": ids[n_dev + n_cal:]}
    out = {}
    for split, part in parts.items():
        rows = [{"image_id": Path(images[i]["file_name"]).stem, "coco_id": int(i),
                 "file_name": images[i]["file_name"], "width": images[i]["width"],
                 "height": images[i]["height"], "n_road_objects": road_counts[i]} for i in part]
        out[split] = pd.DataFrame(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-images", default="dataset/raw/val2017")
    ap.add_argument("--coco-ann", default="dataset/raw/annotations/instances_val2017.json")
    ap.add_argument("--out", default="dataset/clean")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    with open(resolve_path(args.coco_ann)) as f:
        ann = json.load(f)
    splits = build_splits(ann, set(dcfg["road_category_ids"]), dcfg["max_images"],
                          dcfg["split_fractions"], cfg["seed"])

    # Safety checks: no image may appear in two splits
    all_ids = [i for df in splits.values() for i in df["image_id"]]
    assert len(all_ids) == len(set(all_ids)), "duplicate image across splits"

    splits_dir, out_dir, img_dir = resolve_path(args.splits_dir), resolve_path(args.out), resolve_path(args.coco_images)
    splits_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": cfg["seed"], "road_category_ids": dcfg["road_category_ids"], "splits": {}}
    for split, df in splits.items():
        df.to_csv(splits_dir / f"{split}.csv", index=False)
        dst = out_dir / split
        dst.mkdir(parents=True, exist_ok=True)
        for fn in df["file_name"]:
            src = img_dir / fn
            if not src.exists():
                sys.exit(f"Missing image {src}. Did you unzip val2017.zip into {img_dir.parent}?")
            if not (dst / fn).exists():
                shutil.copy2(src, dst / fn)
        summary["splits"][split] = {"images": len(df), "road_objects": int(df["n_road_objects"].sum())}
        print(f"{split:>12}: {len(df):5d} images, {int(df['n_road_objects'].sum()):6d} road objects")
    with open(splits_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Splits written to {splits_dir}, images copied to {out_dir}")


if __name__ == "__main__":
    main()
