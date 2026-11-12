"""Clean-data calibration, score fusion and thresholds (Sections 14-16).

All screening "methods" (the main one, ablations A-G and baselines) are defined
here once, so calibration and evaluation always compute scores identically.
Only pandas/numpy: no PyTorch needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6
VIEW_SETS = {"jpeg+blur": ("jpeg", "blur"), "jpeg": ("jpeg",), "blur": ("blur",)}

# name -> (view set, signals). "full" is the proposed method (= ablation G).
METHODS: dict[str, tuple[str, tuple[str, ...]]] = {
    "full": ("jpeg+blur", ("box", "conf", "feat")),
    "A_box": ("jpeg+blur", ("box",)),
    "B_conf": ("jpeg+blur", ("conf",)),
    "C_feat": ("jpeg+blur", ("feat",)),
    "D_box+conf": ("jpeg+blur", ("box", "conf")),
    "E_box+feat": ("jpeg+blur", ("box", "feat")),
    "F_conf+feat": ("jpeg+blur", ("conf", "feat")),
    "G_box+conf+feat": ("jpeg+blur", ("box", "conf", "feat")),
    "jpeg_only": ("jpeg", ("box", "conf", "feat")),
    "blur_only": ("blur", ("box", "conf", "feat")),
}
# Baseline that uses no transformations at all: low detector confidence = suspicious.
NO_TRANSFORM = "no_transform"
ALL_METHODS = list(METHODS) + [NO_TRANSFORM]
ABLATION_METHODS = [m for m in METHODS if m[:2] in {"A_", "B_", "C_", "D_", "E_", "F_", "G_"}]
BASELINE_METHODS = ["full", "jpeg_only", "blur_only", NO_TRANSFORM]


def aggregated(df: pd.DataFrame, views: tuple[str, ...], signals=("box", "conf", "feat")) -> pd.DataFrame:
    """Max over the given views of each raw signal (Section 13: aggregate BEFORE standardising)."""
    return pd.DataFrame({s: df[[f"D_{s}_{v}" for v in views]].max(axis=1) for s in signals}, index=df.index)


def fit_stats(df: pd.DataFrame) -> dict:
    """Mean and std of every aggregated signal, for every view set, on CLEAN calibration data."""
    stats = {}
    for key, views in VIEW_SETS.items():
        agg = aggregated(df, views)
        stats[key] = {s: {"mean": float(agg[s].mean()), "std": float(agg[s].std(ddof=1))} for s in agg}
    return stats


def method_scores(df: pd.DataFrame, method: str, stats: dict) -> np.ndarray:
    """Final screening score S for each row (higher = more suspicious)."""
    if method == NO_TRANSFORM:
        return (1.0 - df["mean_conf_orig"].astype(float)).to_numpy()
    view_key, signals = METHODS[method]
    agg = aggregated(df, VIEW_SETS[view_key], signals)
    z = [(agg[s] - stats[view_key][s]["mean"]) / max(stats[view_key][s]["std"], EPS) for s in signals]
    return (sum(z) / len(z)).to_numpy()  # equal weights (Section 15)


def threshold_at_fpr(clean_scores: np.ndarray, fpr: float) -> float:
    """Smallest tau such that at most `fpr` of the calibration scores are > tau."""
    return float(np.quantile(np.asarray(clean_scores, float), 1.0 - fpr, method="higher"))


def fit_calibration(df: pd.DataFrame, fprs=(0.05, 0.01)) -> dict:
    stats = fit_stats(df)
    thresholds = {}
    for m in ALL_METHODS:
        s = method_scores(df, m, stats)
        thresholds[m] = {str(f): threshold_at_fpr(s, f) for f in fprs}
    n = len(df)
    notes = {str(f): ("reliable" if n * f >= 10 else f"only {n * f:.0f} calibration images above the threshold: indicative only")
             for f in fprs}
    return {"n_images": n, "fprs": list(fprs), "stats": stats, "thresholds": thresholds, "fpr_notes": notes}


def save_calibration(cal: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def load_calibration(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def decide(score: float, cal: dict, method: str = "full", fpr: float = 0.05) -> str:
    tau = cal["thresholds"][method][str(fpr)]
    return "POTENTIALLY ADVERSARIAL" if score > tau else "CLEAN"
