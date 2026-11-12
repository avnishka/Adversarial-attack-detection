"""Object detection metrics (Section 18.1): mAP@0.50 and mAP@0.50:0.95 of YOLOv8n on the
attacked images, compared with the same images clean. Uses COCO ground truth (pycocotools).

Usage:
    pip install pycocotools
    python scripts/run_detection_eval.py --sets pgd_eps4 pgd_eps8 transfer_eps8 patch patch_random
Writes results/tables/detection_map.csv. Uses conf 0.001 (standard for mAP), not the screening 0.25.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_path  # noqa: E402
from src.detector import YOLOv8nDetector  # noqa: E402
from src.io_utils import image_index, read_ids  # noqa: E402
from src.preprocessing import letterbox, load_rgb, stack_views  # noqa: E402

# YOLO's 80 class indices -> COCO category ids
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
assert len(COCO80_TO_91) == 80


def predict_coco(det: YOLOv8nDetector, folder: Path, ids: list[str], meta: pd.DataFrame, cfg: dict) -> list[dict]:
    """Detections in COCO format, in ORIGINAL image coordinates (adversarial PNGs are rescaled)."""
    index = image_index(folder)
    preds = []
    for image_id in ids:
        img = load_rgb(index[image_id])
        lb, info = letterbox(img, cfg["model"]["imgsz"], cfg["preprocessing"]["pad_value"])
        d = det(stack_views([lb]))[0][0]
        r = meta.loc[image_id]
        sx, sy = r["width"] / img.shape[1], r["height"] / img.shape[0]
        for rec in d.to_records(info):
            x1, y1, x2, y2 = rec["box_xyxy"]
            preds.append({"image_id": int(r["coco_id"]), "category_id": COCO80_TO_91[rec["class_id"]],
                          "bbox": [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy], "score": rec["confidence"]})
    return preds


def coco_map(gt, preds: list[dict], coco_ids: list[int]) -> tuple[float, float]:
    from pycocotools.cocoeval import COCOeval
    if not preds:
        return 0.0, 0.0
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
    dt = gt.loadRes(f.name)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = coco_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])  # mAP@0.50, mAP@0.50:0.95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["pgd_eps4", "pgd_eps8", "transfer_eps8", "patch", "patch_random"])
    ap.add_argument("--ids", default="splits/attack_ids.csv")
    ap.add_argument("--clean", default="dataset/clean/test")
    ap.add_argument("--ann", default="dataset/raw/annotations/instances_val2017.json")
    ap.add_argument("--meta", default="splits/test.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/tables/detection_map.csv")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    cfg = load_config()
    m = cfg["model"]
    det = YOLOv8nDetector(m["weights"], device=args.device or m["device"], conf_thres=0.001,
                          iou_thres=0.7, max_det=300, feature_layer=None, fuse=m["fuse"])
    meta = pd.read_csv(resolve_path(args.meta), dtype={"image_id": str}).set_index("image_id")
    ids = read_ids(resolve_path(args.ids))
    coco_ids = [int(meta.loc[i, "coco_id"]) for i in ids]
    gt = COCO(str(resolve_path(args.ann)))

    rows = []
    folders = {"clean": resolve_path(args.clean)}
    folders.update({s: resolve_path("dataset/adversarial") / s for s in args.sets})
    for name, folder in folders.items():
        if not folder.exists():
            print(f"skip {name}: {folder} not found")
            continue
        m50, m5095 = coco_map(gt, predict_coco(det, folder, ids, meta, cfg), coco_ids)
        rows.append({"set": name, "n_images": len(ids), "mAP50": m50, "mAP50_95": m5095})
        print(f"{name}: mAP@0.50={m50:.3f}  mAP@0.50:0.95={m5095:.3f}")
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
