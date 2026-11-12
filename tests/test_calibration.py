import numpy as np
import pandas as pd
import pytest

from src.calibration import (ALL_METHODS, NO_TRANSFORM, decide, fit_calibration, fit_stats, method_scores,
                             threshold_at_fpr)
from src.evaluation import auroc, bootstrap_ci, evaluate, rate_above


def fake_scores(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    data = {"image_id": [f"{i:06d}" for i in range(n)], "mean_conf_orig": rng.uniform(0.3, 0.9, n) - shift / 10}
    for s, scale in (("box", 0.1), ("conf", 0.05), ("feat", 0.02)):
        for v in ("jpeg", "blur"):
            data[f"D_{s}_{v}"] = np.abs(rng.normal(scale, scale / 2, n)) + shift * scale
    return pd.DataFrame(data)


def test_stats_and_standardised_mean_is_zero_on_calibration():
    cal_df = fake_scores(600)
    stats = fit_stats(cal_df)
    s = method_scores(cal_df, "A_box", stats)
    assert abs(s.mean()) < 1e-9 and s.std(ddof=1) == pytest.approx(1.0)


def test_thresholds_respect_target_fpr_on_calibration():
    cal_df = fake_scores(600)
    cal = fit_calibration(cal_df, (0.05, 0.01))
    for m in ALL_METHODS:
        s = method_scores(cal_df, m, cal["stats"])
        for f in (0.05, 0.01):
            assert rate_above(s, cal["thresholds"][m][str(f)]) <= f
    assert "indicative" in cal["fpr_notes"]["0.01"]          # 600 * 1% = 6 images -> flagged as indicative
    assert cal["fpr_notes"]["0.05"] == "reliable"


def test_threshold_quantile():
    assert threshold_at_fpr(np.arange(100), 0.05) == 95


def test_shifted_data_is_detected():
    cal = fit_calibration(fake_scores(600, seed=1))
    clean, adv = fake_scores(1000, seed=2), fake_scores(200, shift=2.0, seed=3)
    row = evaluate(clean, adv, cal, "full", n_boot=200)
    assert row["auroc"] > 0.9 and row["auroc_lo"] <= row["auroc"] <= row["auroc_hi"]
    assert row["fpr@5"] < 0.1 and row["tpr@5"] > 0.5
    assert decide(10.0, cal) == "POTENTIALLY ADVERSARIAL" and decide(-10.0, cal) == "CLEAN"


def test_no_transform_baseline():
    df = pd.DataFrame({"mean_conf_orig": [0.9, 0.0]})
    assert list(method_scores(df, NO_TRANSFORM, {})) == pytest.approx([0.1, 1.0])


def test_auroc_and_bootstrap():
    clean, adv = np.zeros(50), np.ones(50)
    assert auroc(clean, adv) == 1.0
    lo, hi = bootstrap_ci(clean, adv, auroc, n_boot=50)
    assert lo == hi == 1.0
