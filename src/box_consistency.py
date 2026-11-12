"""Bounding-box consistency (Section 12.1).

D_box = [ sum over matches (1 - IoU) + |U| ] / n,   n = |M| + |U|
U = unmatched detections in BOTH views. Empty convention: n == 0 -> D = 0
(neither view sees anything, so the views agree).
"""
from __future__ import annotations

from .box_matching import MatchResult


def box_difference(match: MatchResult) -> float:
    n = match.n_total
    if n == 0:
        return 0.0
    unmatched = len(match.unmatched_a) + len(match.unmatched_b)
    return (sum(1.0 - iou for _, _, iou in match.matches) + unmatched) / n
