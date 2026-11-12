"""Screening metrics (Section 18.2) with bootstrap confidence intervals.

Thresholds always come from the calibration file (clean calibration split), never from
the test data, so reported FPR and TPR are honest operating points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .calibration import method_scores


def auroc(clean: np.ndarray, adv: np.ndarray) -> float:
    y = np.r_[np.zeros(len(clean)), np.ones(len(adv))]
    return float(roc_auc_score(y, np.r_[clean, adv]))


def rate_above(scores: np.ndarray, tau: float) -> float:
    return float(np.mean(np.asarray(scores) > tau)) if len(scores) else float("nan")


def bootstrap_ci(clean: np.ndarray, adv: np.ndarray, fn, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile CI, resampling clean and adversarial images separately."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        c = clean[rng.integers(0, len(clean), len(clean))]
        a = adv[rng.integers(0, len(adv), len(adv))]
        vals.append(fn(c, a))
    lo, hi = np.nanpercentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def evaluate(clean_df: pd.DataFrame, adv_df: pd.DataFrame, cal: dict, method: str,
             n_boot: int = 1000, seed: int = 0) -> dict:
    stats = cal["stats"]
    c = method_scores(clean_df, method, stats)
    a = method_scores(adv_df, method, stats)
    row = {"method": method, "n_clean": len(c), "n_adv": len(a), "auroc": auroc(c, a)}
    row["auroc_lo"], row["auroc_hi"] = bootstrap_ci(c, a, auroc, n_boot, seed)
    for fpr in cal["fprs"]:
        tau = cal["thresholds"][method][str(fpr)]
        tag = f"{int(round(fpr * 100))}"
        row[f"tpr@{tag}"] = rate_above(a, tau)
        row[f"tpr@{tag}_lo"], row[f"tpr@{tag}_hi"] = bootstrap_ci(c, a, lambda cc, aa: rate_above(aa, tau),
                                                                  n_boot, seed)
        row[f"fpr@{tag}"] = rate_above(c, tau)  # measured on the clean TEST set
    return row


def roc_points(clean_df: pd.DataFrame, adv_df: pd.DataFrame, cal: dict, method: str = "full"):
    c = method_scores(clean_df, method, cal["stats"])
    a = method_scores(adv_df, method, cal["stats"])
    y = np.r_[np.zeros(len(c)), np.ones(len(a))]
    fpr, tpr, _ = roc_curve(y, np.r_[c, a])
    return fpr, tpr
