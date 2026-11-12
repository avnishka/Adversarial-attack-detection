"""Per-view differences and maximum-view aggregation (Section 13).

Column naming used everywhere (CSV files, calibration, evaluation):
    D_box_jpeg, D_box_blur, D_conf_jpeg, D_conf_blur, D_feat_jpeg, D_feat_blur
    D_feat_gap_<view>, D_feat_flatten_<view>   (both feature modes, for parameter selection)
Per-view values are always kept, so view baselines (JPEG-only, blur-only) need no new inference.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .box_consistency import box_difference
from .box_matching import match_detections
from .confidence_consistency import confidence_difference
from .feature_consistency import FEATURE_MODES, feature_distance

VIEWS = ("jpeg", "blur")            # transformed views, in batch order 1, 2 (0 = original)
SIGNALS = ("box", "conf", "feat")


class DetLike(Protocol):
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray


def view_differences(dets: Sequence[DetLike], feats: np.ndarray | None = None, iou_min: float = 0.3,
                     class_aware: bool = True, feat_mode: str = "gap") -> dict[str, float]:
    """dets[0] = original view, dets[1..] = VIEWS in order; feats = (3, C, h, w) or None."""
    if len(dets) != 1 + len(VIEWS):
        raise ValueError(f"Expected {1 + len(VIEWS)} views, got {len(dets)}")
    ref = dets[0]
    out: dict[str, float] = {}
    for k, view in enumerate(VIEWS, start=1):
        d = dets[k]
        m = match_detections(ref.boxes, ref.classes, d.boxes, d.classes, iou_min=iou_min, class_aware=class_aware)
        out[f"D_box_{view}"] = box_difference(m)
        out[f"D_conf_{view}"] = confidence_difference(m, ref.scores, d.scores)
        if feats is not None:
            for mode in FEATURE_MODES:
                out[f"D_feat_{mode}_{view}"] = feature_distance(feats[0], feats[k], mode)
            out[f"D_feat_{view}"] = out[f"D_feat_{feat_mode}_{view}"]
    return out


def aggregate(row: dict, views: Sequence[str] = VIEWS, signals: Sequence[str] = SIGNALS) -> dict[str, float]:
    """D_k = max over views of D_k,view (raw values, BEFORE standardisation)."""
    return {s: max(float(row[f"D_{s}_{v}"]) for v in views) for s in signals}
