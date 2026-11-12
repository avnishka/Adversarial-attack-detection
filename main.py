"""Three-view demo on one image: prints detections per view and feature cosine distances.

    python main.py path/to/image.jpg

The full screening pipeline (scores, calibration, decision) is added in Phases 6-8.
"""
from __future__ import annotations

import sys

import torch

from src.box_matching import match_detections
from src.config import load_config
from src.detector import YOLOv8nDetector
from src.preprocessing import letterbox, load_rgb, stack_views
from src.transformations import VIEW_NAMES, make_views


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cfg = load_config()
    det = YOLOv8nDetector.from_config(cfg)
    img = load_rgb(sys.argv[1])
    lb, info = letterbox(img, cfg["model"]["imgsz"], cfg["preprocessing"]["pad_value"])
    views = make_views(lb, info, cfg["transforms"]["jpeg_quality"], cfg["transforms"]["blur_sigma"])
    dets, feats = det(stack_views([views[n] for n in VIEW_NAMES]))

    for name, d in zip(VIEW_NAMES, dets):
        labels = ", ".join(f"{r['class_name']}({r['confidence']:.2f})" for r in d.to_records(info))
        print(f"{name:>8}: {len(d)} -> {labels}")

    m = cfg["matching"]
    for v in (1, 2):
        res = match_detections(dets[0].boxes, dets[0].classes, dets[v].boxes, dets[v].classes,
                               iou_min=m["iou_min"], class_aware=m["class_aware"])
        mean_iou = sum(x[2] for x in res.matches) / max(len(res.matches), 1)
        print(f"original vs {VIEW_NAMES[v]}: matched={len(res.matches)} "
              f"unmatched={len(res.unmatched_a) + len(res.unmatched_b)} mean IoU={mean_iou:.3f}")

    if feats is not None:
        f = feats.flatten(1)
        d = 1 - torch.nn.functional.cosine_similarity(f[0:1], f[1:], dim=1)
        print(f"feature cosine distance: jpeg={d[0]:.4f} blur={d[1]:.4f}")


if __name__ == "__main__":
    main()
