"""Runtime measurement (Section 18.3): per-stage latency of the screening pipeline compared
with the plain detector, on ONE fixed laptop.

Usage (plugged in, other apps closed):
    python scripts/run_runtime.py --source dataset/clean/test --device cpu
Writes results/tables/runtime.csv and results/tables/runtime_env.csv.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_path  # noqa: E402
from src.io_utils import image_index  # noqa: E402
from src.preprocessing import letterbox, load_rgb, stack_views  # noqa: E402
from src.screening import Screener  # noqa: E402


def summarise(name: str, ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"stage": name, "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "fps_from_median": float(1000 / np.median(a)), "runs": len(a)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dataset/clean/test")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import ultralytics
    cfg = load_config()
    scr = Screener(cfg, device=args.device)
    det = scr.det
    paths = list(image_index(resolve_path(args.source)).values())[: args.n_images]
    images = [load_rgb(p) for p in paths]  # loaded up front: disk time is not measured

    for i in range(args.warmup):
        scr.screen(images[i % len(images)])
    stage = {"t_views_ms": [], "t_inference_ms": [], "t_scoring_ms": [], "t_total_ms": []}
    for i in range(args.runs):
        row = scr.screen(images[i % len(images)])
        for k in stage:
            stage[k].append(row[k])

    # Plain detector: one view, batch of 1, same letterbox, no screening
    plain = []
    lbs = [letterbox(im, cfg["model"]["imgsz"], cfg["preprocessing"]["pad_value"])[0] for im in images]
    for i in range(args.warmup):
        det(stack_views([lbs[i % len(lbs)]]))
    for i in range(args.runs):
        det.synchronize()
        t0 = time.perf_counter()
        det(stack_views([lbs[i % len(lbs)]]))
        det.synchronize()
        plain.append((time.perf_counter() - t0) * 1e3)

    rows = [summarise("plain YOLOv8n (1 view)", plain),
            summarise("screening: views", stage["t_views_ms"]),
            summarise("screening: batched inference (3 views)", stage["t_inference_ms"]),
            summarise("screening: scoring + decision", stage["t_scoring_ms"]),
            summarise("screening: end to end", stage["t_total_ms"])]
    df = pd.DataFrame(rows).round(2)
    df["overhead_vs_plain"] = (df["median_ms"] / df.loc[0, "median_ms"]).round(2)
    out = resolve_path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runtime.csv", index=False)
    env = {"device": det.device, "cpu": platform.processor() or platform.machine(), "os": platform.platform(),
           "python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
           "torch_threads": torch.get_num_threads(), "imgsz": cfg["model"]["imgsz"],
           "warmup": args.warmup, "runs": args.runs, "n_images": len(images),
           "gpu": torch.cuda.get_device_name(0) if det.device.startswith("cuda") else ""}
    pd.DataFrame(list(env.items()), columns=["key", "value"]).to_csv(out / "runtime_env.csv", index=False)
    print(df.to_string(index=False))
    print("Add the RAM size and laptop model to runtime_env.csv by hand.")


if __name__ == "__main__":
    main()
