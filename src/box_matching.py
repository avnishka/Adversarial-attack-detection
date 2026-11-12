"""One-to-one matching of detections between two views (Section 12.1).

Hungarian assignment on cost = 1 - IoU. Pairs with different classes (if
class_aware) or IoU below `iou_min` are rejected and counted as unmatched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    matches: list[tuple[int, int, float]]  # (index in A, index in B, IoU)
    unmatched_a: list[int]
    unmatched_b: list[int]

    @property
    def n_total(self) -> int:
        """|M| + |U|, with U = unmatched in A plus unmatched in B (the n_v of the proposal)."""
        return len(self.matches) + len(self.unmatched_a) + len(self.unmatched_b)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_detections(boxes_a: np.ndarray, classes_a: np.ndarray,
                     boxes_b: np.ndarray, classes_b: np.ndarray,
                     iou_min: float = 0.3, class_aware: bool = True) -> MatchResult:
    boxes_a = np.asarray(boxes_a).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b).reshape(-1, 4)
    classes_a = np.asarray(classes_a).reshape(-1)
    classes_b = np.asarray(classes_b).reshape(-1)
    na, nb = len(boxes_a), len(boxes_b)
    if na == 0 or nb == 0:
        return MatchResult([], list(range(na)), list(range(nb)))

    iou = box_iou(boxes_a, boxes_b)
    valid = iou >= iou_min
    if class_aware:
        valid &= classes_a[:, None] == classes_b[None, :]
    cost = np.where(valid, 1.0 - iou, 1e6)  # invalid pairs are effectively forbidden

    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if valid[r, c]]
    matched_a = {m[0] for m in matches}
    matched_b = {m[1] for m in matches}
    return MatchResult(matches=matches,
                       unmatched_a=[i for i in range(na) if i not in matched_a],
                       unmatched_b=[j for j in range(nb) if j not in matched_b])
