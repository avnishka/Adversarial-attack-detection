"""Choose the feature layer and feature mode on the DEVELOPMENT split only.

Screens dev clean images and dev adversarial images (e.g. 50 PGD images made from the
dev split) for each candidate layer, then reports AUROC of the full score and of the
feature signal alone, for both feature modes. Statistics come from dev clean data, so
the test split is never touched.

Usage:
    python scripts/select_params.py --clean dataset/clean/dev --adv dataset/adversarial/pgd_dev_eps8 --layers 9 15 21
Then set model.feature_layer and features.mode in configs/default.yaml, commit, and tag frozen-config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_screening import screen_folder  # noqa: E402
from src.calibration import fit_stats, method_scores  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.feature_consistency import FEATURE_MODES  # noqa: E402
from src.io_utils import image_index  # noqa: E402
from src.score_aggregation import VIEWS  # noqa: E402
from src.screening import Screener  # noqa: E402


def with_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()
    for v in VIEWS:
        df[f"D_feat_{v}"] = df[f"D_feat_{mode}_{v}"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="dataset/clean/dev")
    ap.add_argument("--adv", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 15, 21])
    ap.add_argument("--clean-limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/tables/param_selection.csv")
    args = ap.parse_args()
    cfg = load_config()

    adv_ids = list(image_index(resolve_path(args.adv)))
    # Clean dev images that were attacked are excluded, so clean and adversarial sets are disjoint images
    clean_ids = [i for i in image_index(resolve_path(args.clean)) if i not in set(adv_ids)]
    if args.clean_limit:
        clean_ids = clean_ids[:args.clean_limit]

    rows = []
    for layer in args.layers:
        print(f"== layer {layer} ==")
        scr = Screener(cfg, device=args.device, feature_layer=layer)
        clean = screen_folder(scr, resolve_path(args.clean), clean_ids, label="dev clean")
        adv = screen_folder(scr, resolve_path(args.adv), label="dev adv")
        for mode in FEATURE_MODES:
            c, a = with_mode(clean, mode), with_mode(adv, mode)
            stats = fit_stats(c)
            y = [0] * len(c) + [1] * len(a)
            row = {"layer": layer, "feat_mode": mode, "n_clean": len(c), "n_adv": len(a)}
            for m in ("full", "C_feat", "A_box", "B_conf"):
                s = list(method_scores(c, m, stats)) + list(method_scores(a, m, stats))
                row[f"auroc_{m}"] = roc_auc_score(y, s)
            rows.append(row)
            print(f"  mode={mode:<8} AUROC full={row['auroc_full']:.3f} feat={row['auroc_C_feat']:.3f}")
    res = pd.DataFrame(rows).sort_values("auroc_full", ascending=False)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    best = res.iloc[0]
    print(f"\nBest on dev: model.feature_layer={int(best['layer'])}, features.mode={best['feat_mode']}")


if __name__ == "__main__":
    main()
