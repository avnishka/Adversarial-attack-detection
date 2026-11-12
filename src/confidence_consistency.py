"""Confidence consistency (Section 12.2).

D_conf = [ sum over matches |c_orig - c_view| + |U| ] / n, same matching and
same empty convention as box consistency (n == 0 -> 0).
"""
from __future__ import annotations

import numpy as np

from .box_matching import MatchResult


def confidence_difference(match: MatchResult, scores_ref: np.ndarray, scores_view: np.ndarray) -> float:
    n = match.n_total
    if n == 0:
        return 0.0
    scores_ref = np.asarray(scores_ref, dtype=np.float64).reshape(-1)
    scores_view = np.asarray(scores_view, dtype=np.float64).reshape(-1)
    unmatched = len(match.unmatched_a) + len(match.unmatched_b)
    diff = sum(abs(scores_ref[i] - scores_view[j]) for i, j, _ in match.matches)
    return float((diff + unmatched) / n)
