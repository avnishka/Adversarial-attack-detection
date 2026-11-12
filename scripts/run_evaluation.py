"""Phase 10: all screening tables for the paper, from saved score CSVs (no model needed).

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --sets pgd_eps4 pgd_eps8 transfer_eps8 patch patch_random

Reads  results/calibration/calibration.json, results/clean_test/scores.csv,
       results/<set>/scores.csv and results/<set>/attack_success.csv
Writes results/tables/main_results.csv   proposed method, every set, all attempts + successful only
       results/tables/ablation.csv       experiments A-G (Section 20)
       results/tables/baselines.csv      no-transform, JPEG-only, blur-only vs full (Section 19)
       results/tables/attack_success.csv attack success per set
       results/tables/clean_fpr.csv      false-positive rates on clean test images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import ABLATION_METHODS, BASELINE_METHODS, load_calibration, method_scores  # noqa: E402
from src.config import resolve_path  # noqa: E402
from src.evaluation import evaluate, rate_above  # noqa: E402
from src.io_utils import read_scores  # noqa: E402

DEFAULT_SETS = ["pgd_eps4", "pgd_eps8", "transfer_eps8", "patch", "patch_random"]


def load_sets(names: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame | None]]:
    sets = {}
    for name in names:
        p = resolve_path("results") / name / "scores.csv"
        if not p.exists():
            print(f"skip {name}: {p} not found")
            continue
        s = resolve_path("results") / name / "attack_success.csv"
        sets[name] = (read_scores(p), read_scores(s) if s.exists() else None)
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--clean", default="results/clean_test/scores.csv")
    ap.add_argument("--calibration", default="results/calibration/calibration.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    cal = load_calibration(resolve_path(args.calibration))
    clean = read_scores(resolve_path(args.clean))
    sets = load_sets(args.sets)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    main_rows, abl_rows, base_rows, asr_rows = [], [], [], []
    for name, (adv, succ) in sets.items():
        subsets = {"all": adv}
        if succ is not None:
            ok_ids = set(succ.loc[succ["success"].astype(str).str.lower() == "true", "image_id"])
            subsets["successful"] = adv[adv["image_id"].isin(ok_ids)]
            asr_rows.append({"set": name, "n_images": len(succ), "attack_success_rate": succ["success"].astype(str).str.lower().eq("true").mean(),
                             "mean_clean_det": succ["n_clean_det"].mean(), "mean_adv_det": succ["n_adv_det"].mean(),
                             "mean_linf_255": succ["linf_255"].mean()})
        for subset, df in subsets.items():
            if len(df) < 2:
                continue
            main_rows.append({"set": name, "subset": subset, **evaluate(clean, df, cal, "full", args.n_boot)})
        for m in ABLATION_METHODS:
            abl_rows.append({"set": name, **evaluate(clean, adv, cal, m, args.n_boot)})
        for m in BASELINE_METHODS:
            base_rows.append({"set": name, **evaluate(clean, adv, cal, m, args.n_boot)})
        allrow = next((r for r in main_rows if r["set"] == name and r["subset"] == "all"), None)
        if allrow:
            print(f"{name}: AUROC full = {allrow['auroc']:.3f}, TPR@5% FPR = {allrow['tpr@5']:.3f}")

    clean_rows = [{"method": m, "n_clean_test": len(clean),
                   **{f"fpr@{int(round(f * 100))}": rate_above(method_scores(clean, m, cal["stats"]),
                                                               cal["thresholds"][m][str(f)]) for f in cal["fprs"]}}
                  for m in BASELINE_METHODS]

    for fname, rows in [("main_results", main_rows), ("ablation", abl_rows), ("baselines", base_rows),
                        ("attack_success", asr_rows), ("clean_fpr", clean_rows)]:
        if rows:
            pd.DataFrame(rows).round(4).to_csv(out / f"{fname}.csv", index=False)
            print(f"wrote {out / (fname + '.csv')}")


if __name__ == "__main__":
    main()
