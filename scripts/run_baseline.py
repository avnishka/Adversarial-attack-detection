"""Phase 2: baseline YOLOv8n detection on clean images.

Saves per-image predictions (classes, confidences, boxes in ORIGINAL-image pixels),
an optional annotated image, a reproducibility check and latency statistics.

Usage:
    python scripts/run_baseline.py --source path/to/image_or_folder [--save-images] [--timing-runs 50]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.detector import YOLOv8nDetector  # noqa: E402
from src.preprocessing import letterbox, load_rgb, stack_views  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def draw(img_rgb: np.ndarray, records: list[dict]) -> np.ndarray:
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for r in records:
        x1, y1, x2, y2 = map(int, r["box_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"{r['class_name']} {r['confidence']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--timing-runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (overrides config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_path(args.out)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    det = YOLOv8nDetector.from_config(cfg, device=args.device)
    size, pad = cfg["model"]["imgsz"], cfg["preprocessing"]["pad_value"]

    images = list_images(args.source)
    if not images:
        sys.exit(f"No images found in {args.source}")
    print(f"{len(images)} image(s), device={det.device}")

    reproducible = True
    for path in images:
        img = load_rgb(path)
        lb, info = letterbox(img, size, pad)
        batch = stack_views([lb])
        dets, _ = det(batch)
        dets2, _ = det(batch)  # same input twice -> must give identical output
        same = (len(dets[0]) == len(dets2[0]) and np.allclose(dets[0].boxes, dets2[0].boxes, atol=1e-4)
                and np.allclose(dets[0].scores, dets2[0].scores, atol=1e-5))
        reproducible &= same
        records = dets[0].to_records(info)
        with open(out_dir / "predictions" / f"{path.stem}.json", "w") as f:
            json.dump({"image": str(path), "orig_size_wh": [info.orig_w, info.orig_h],
                       "detections": records}, f, indent=2)
        if args.save_images:
            cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), draw(img, records))
        print(f"  {path.name}: {len(records)} detections, reproducible={same}")

    # Timing on the first image: model forward + NMS only (preprocessing excluded)
    lb, _ = letterbox(load_rgb(images[0]), size, pad)
    batch = stack_views([lb])
    for _ in range(args.warmup):
        det(batch)
    times = []
    for _ in range(args.timing_runs):
        det.synchronize()
        t0 = time.perf_counter()
        det(batch)
        det.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    import torch
    import ultralytics
    summary = {
        "n_images": len(images),
        "reproducible": bool(reproducible),
        "latency_ms": {"median": float(np.median(times)), "p95": float(np.percentile(times, 95)),
                       "mean": float(times.mean()), "runs": args.timing_runs, "warmup": args.warmup,
                       "scope": "forward pass + NMS, batch of 1, preprocessing excluded"},
        "fps_from_median": float(1000 / np.median(times)),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "ultralytics": ultralytics.__version__, "device": det.device,
                        "gpu": torch.cuda.get_device_name(0) if det.device.startswith("cuda") else None,
                        "cpu": platform.processor() or platform.machine()},
        "config": cfg,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReproducible: {reproducible}")
    print(f"Latency: median {summary['latency_ms']['median']:.1f} ms, p95 {summary['latency_ms']['p95']:.1f} ms")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
