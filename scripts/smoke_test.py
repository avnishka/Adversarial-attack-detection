"""Phase 1 (environment) + Phase 5 (batched three-view inference) acceptance checks.

Usage:
    python scripts/smoke_test.py                 # uses Ultralytics' bundled bus.jpg
    python scripts/smoke_test.py path/to/img.jpg
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    results = []
    print("== Phase 1: environment ==")
    results.append(check("Python", True, platform.python_version()))
    try:
        import torch
        results.append(check("PyTorch imports", True,
                             f"{torch.__version__}, CUDA available: {torch.cuda.is_available()}"))
    except ImportError as e:
        check("PyTorch imports", False, str(e))
        return 1
    try:
        import ultralytics
        results.append(check("Ultralytics imports", True, ultralytics.__version__))
    except ImportError as e:
        check("Ultralytics imports", False, str(e))
        return 1

    from src.config import load_config
    from src.detector import YOLOv8nDetector
    from src.feature_extractor import describe_layers
    from src.preprocessing import letterbox, load_rgb, stack_views
    from src.transformations import VIEW_NAMES, make_views

    cfg = load_config()
    try:
        det = YOLOv8nDetector.from_config(cfg)
        results.append(check("YOLOv8n loads", True, f"device={det.device}, {len(det.names)} classes"))
    except Exception as e:
        check("YOLOv8n loads", False, str(e))
        return 1

    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        from ultralytics.utils import ASSETS
        img_path = ASSETS / "bus.jpg"
    img = load_rgb(img_path)
    size = cfg["model"]["imgsz"]
    lb, info = letterbox(img, size, cfg["preprocessing"]["pad_value"])

    dets, _ = det(stack_views([lb]))
    results.append(check("Sample image produces detections", len(dets[0]) > 0,
                         f"{img_path.name}: {len(dets[0])} objects"))
    for r in dets[0].to_records(info)[:5]:
        print(f"      {r['class_name']:<12} conf={r['confidence']:.2f} box={[round(v) for v in r['box_xyxy']]}")

    print("\n== Phase 5: batched three-view inference ==")
    views = make_views(lb, info, cfg["transforms"]["jpeg_quality"], cfg["transforms"]["blur_sigma"])
    batch = stack_views([views[n] for n in VIEW_NAMES])
    x = det.to_tensor(batch)
    results.append(check("Batch tensor shape", tuple(x.shape) == (3, 3, size, size), str(tuple(x.shape))))
    dets, feats = det(batch)
    results.append(check("Three sets of detections", len(dets) == 3,
                         ", ".join(f"{n}={len(d)}" for n, d in zip(VIEW_NAMES, dets))))
    if feats is not None:
        results.append(check("Three feature maps from one pass", feats.shape[0] == 3,
                             f"layer {det.feature_layer}: {tuple(feats.shape)}"))
        f = feats.flatten(1)
        cos = torch.nn.functional.cosine_similarity(f[0:1], f[1:], dim=1)
        print(f"      cosine(original, jpeg)={cos[0]:.4f}  cosine(original, blur)={cos[1]:.4f}")
    # Running a second batch must give fresh (not stale) features of the right size
    dets1, feats1 = det(stack_views([lb]))
    if feats1 is not None:
        results.append(check("No stale hook outputs", feats1.shape[0] == 1))

    print("\n== YOLOv8n layers (use this to choose model.feature_layer) ==")
    for i, name, shape in describe_layers(det.net, size, det.device):
        mark = "  <- current feature_layer" if i == det.feature_layer else ""
        print(f"  {i:>2}  {name:<10} {shape}{mark}")

    ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
