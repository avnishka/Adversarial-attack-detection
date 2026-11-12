"""Phase 7: learn clean statistics and thresholds from the CALIBRATION split only.

Usage:
    python scripts/run_calibration.py                      # screens dataset/clean/calibration, then fits
    python scripts/run_calibration.py --from-csv results/calibration/raw_scores.csv   # refit only
Writes results/calibration/calibration.json (commit this file).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import fit_calibration, save_calibration  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.io_utils import read_scores  # noqa: E402
from src.run_meta import config_hash, run_info  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dataset/clean/calibration")
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--raw-out", default="results/calibration/raw_scores.csv")
    ap.add_argument("--out", default="results/calibration/calibration.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "from-csv"
    if args.from_csv:
        df = read_scores(resolve_path(args.from_csv))
    else:
        from scripts.run_screening import screen_folder  # imported here: needs torch
        from src.screening import Screener
        screener = Screener(cfg, device=args.device)
        device = screener.det.device
        df = screen_folder(screener, resolve_path(args.source), label="calibration")
        raw = resolve_path(args.raw_out)
        raw.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw, index=False)

    cal = fit_calibration(df, tuple(cfg["calibration"]["fprs"]))
    cal["config"] = {k: cfg[k] for k in ("model", "transforms", "matching", "features")}
    cal["config_hash"] = config_hash(cfg)
    cal["run"] = run_info(device)
    save_calibration(cal, resolve_path(args.out))
    print(json.dumps({"n_images": cal["n_images"], "full_thresholds": cal["thresholds"]["full"],
                      "notes": cal["fpr_notes"]}, indent=2))
    print(f"Saved {resolve_path(args.out)}")


if __name__ == "__main__":
    main()
