"""Complete screening for one image (Sections 10-16):
letterbox -> three views -> ONE batched YOLOv8n pass -> per-view differences
-> max-view aggregation -> clean standardisation -> fusion -> threshold.

Every intermediate value is returned so it can be saved and re-analysed later
(ablations and baselines are computed from these columns without new inference).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .calibration import decide, method_scores
from .detector import YOLOv8nDetector
from .preprocessing import letterbox, stack_views
from .score_aggregation import view_differences
from .transformations import VIEW_NAMES, make_views


class Screener:
    def __init__(self, cfg: dict, device: str | None = None, calibration: dict | None = None,
                 feature_layer: int | None = None):
        if feature_layer is not None:
            cfg = {**cfg, "model": {**cfg["model"], "feature_layer": feature_layer}}
        self.cfg = cfg
        self.det = YOLOv8nDetector.from_config(cfg, device=device)
        if self.det.hook is None:
            raise ValueError("model.feature_layer must be set for screening")
        self.calibration = calibration

    def screen(self, img_rgb: np.ndarray) -> dict:
        cfg = self.cfg
        t0 = time.perf_counter()
        lb, info = letterbox(img_rgb, cfg["model"]["imgsz"], cfg["preprocessing"]["pad_value"])
        views = make_views(lb, info, cfg["transforms"]["jpeg_quality"], cfg["transforms"]["blur_sigma"])
        batch = stack_views([views[n] for n in VIEW_NAMES])
        t1 = time.perf_counter()

        dets, feats = self.det(batch)
        feats_np = feats.float().cpu().numpy()   # .cpu() also waits for the GPU
        self.det.synchronize()
        t2 = time.perf_counter()

        row = view_differences(dets, feats_np, iou_min=cfg["matching"]["iou_min"],
                               class_aware=cfg["matching"]["class_aware"],
                               feat_mode=cfg["features"]["mode"])
        for name, d in zip(VIEW_NAMES, dets):
            row[f"n_det_{name}"] = len(d)
        row["mean_conf_orig"] = float(dets[0].scores.mean()) if len(dets[0]) else 0.0
        t3 = time.perf_counter()

        if self.calibration is not None:
            row["S"] = float(method_scores(pd.DataFrame([row]), "full", self.calibration["stats"])[0])
            row["decision"] = decide(row["S"], self.calibration, "full", cfg["calibration"]["decision_fpr"])
        t4 = time.perf_counter()

        row.update({"t_views_ms": (t1 - t0) * 1e3, "t_inference_ms": (t2 - t1) * 1e3,
                    "t_scoring_ms": (t4 - t2) * 1e3, "t_total_ms": (t4 - t0) * 1e3})
        return row
