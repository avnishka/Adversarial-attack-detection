"""Phase 8: screen every image in a folder and save one CSV row per image.

Usage:
    python scripts/run_screening.py --source dataset/clean/test --out results/clean_test/scores.csv
    python scripts/run_screening.py --source dataset/adversarial/pgd_eps8 --out results/pgd_eps8/scores.csv
Options: --ids splits/attack_ids.csv (only these images), --limit N, --device cpu,
         --calibration results/calibration/calibration.json (adds S and the decision).
For numbers in the paper, always run on the same laptop with --device cpu.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import load_calibration  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.io_utils import image_index, read_ids  # noqa: E402
from src.preprocessing import load_rgb  # noqa: E402
from src.run_meta import write_meta  # noqa: E402
from src.screening import Screener  # noqa: E402


def screen_folder(screener: Screener, source: Path, ids: list[str] | None = None, limit: int | None = None,
                  label: str = "") -> pd.DataFrame:
    index = image_index(source)
    if ids is not None:
        missing = [i for i in ids if i not in index]
        if missing:
            print(f"WARNING: {len(missing)} ids not found in {source} (e.g. {missing[:3]})")
        index = {i: index[i] for i in ids if i in index}
    items = list(index.items())[:limit] if limit else list(index.items())
    rows, t0 = [], time.time()
    for k, (image_id, path) in enumerate(items, 1):
        row = {"image_id": image_id}
        row.update(screener.screen(load_rgb(path)))
        rows.append(row)
        if k % 50 == 0 or k == len(items):
            print(f"  {label} {k}/{len(items)} images ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cal = load_calibration(resolve_path(args.calibration)) if args.calibration else None
    screener = Screener(cfg, device=args.device, calibration=cal)
    print(f"device={screener.det.device}, feature_layer={screener.det.feature_layer}, feat_mode={cfg['features']['mode']}")
    df = screen_folder(screener, resolve_path(args.source), read_ids(args.ids) if args.ids else None,
                       args.limit, label=Path(args.source).name)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    write_meta(out.parent / "screening_meta.json", cfg, screener.det.device, n_rows=len(df),
               source=str(args.source), ids=args.ids, limit=args.limit)
    print(f"Saved {len(df)} rows to {out}")
    if "decision" in df:
        print(df["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
