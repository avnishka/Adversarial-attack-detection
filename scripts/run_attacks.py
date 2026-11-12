"""Phase 9: generate adversarial images and measure attack success on the TARGET YOLOv8n.

Attacks (all digital, offline, on the authorised COCO subset):
  pgd           white-box PGD on YOLOv8n                       --eps 4 or 8   (in /255)
  transfer      PGD on a surrogate (yolov5nu.pt), applied to YOLOv8n (black-box)   --eps 8
  patch         universal DPatch-style patch trained on dev images
  patch_random  occlusion control: random-pixel patch, same size and positions

Typical order (Member B):
  # 50 dev images for parameter selection (Member A needs these first)
  python scripts/run_attacks.py --attack pgd --eps 8 --source dataset/clean/dev --n 50 --name pgd_dev_eps8
  # choose the 200 test images once, then reuse them for every attack
  python scripts/run_attacks.py --attack pgd --eps 8 --save-ids splits/attack_ids.csv --name pgd_eps8
  python scripts/run_attacks.py --attack pgd --eps 4 --ids splits/attack_ids.csv --name pgd_eps4
  python scripts/run_attacks.py --attack transfer --eps 8 --ids splits/attack_ids.csv --name transfer_eps8
  python scripts/run_attacks.py --attack patch --ids splits/attack_ids.csv --name patch
  python scripts/run_attacks.py --attack patch_random --ids splits/attack_ids.csv --name patch_random

Outputs: dataset/adversarial/<name>/<image_id>.png (share via Drive, never commit),
         results/<name>/attack_success.csv and attack_config.json (commit these).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.common import (content_mask, detection_change, is_success, load_letterboxed,  # noqa: E402
                            save_content_png, to_tensor, to_uint8)
from attacks.dpatch.patch_attack import apply_patch, fixed_position, random_patch, train_patch  # noqa: E402
from attacks.pgd.pgd_attack import pgd_attack  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.detector import YOLOv8nDetector  # noqa: E402
from src.io_utils import image_index, read_ids  # noqa: E402
from src.preprocessing import stack_views  # noqa: E402


def select_ids(target: YOLOv8nDetector, index: dict, cfg: dict, n: int, seed: int) -> list[str]:
    """First n images (in a seeded random order) with at least one clean detection."""
    ids = list(index)
    order = np.random.default_rng(seed).permutation(len(ids))
    chosen = []
    for k in order:
        lb, _ = load_letterboxed(index[ids[k]], cfg)
        dets, _ = target(stack_views([lb]))
        if len(dets[0]) > 0:
            chosen.append(ids[k])
        if len(chosen) == n:
            break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True, choices=["pgd", "transfer", "patch", "patch_random"])
    ap.add_argument("--name", required=True, help="output folder name, e.g. pgd_eps8")
    ap.add_argument("--source", default="dataset/clean/test")
    ap.add_argument("--ids", default=None, help="CSV with image_id column (reuse the same images)")
    ap.add_argument("--save-ids", default=None, help="write the selected ids here")
    ap.add_argument("--n", type=int, default=None, help="number of images (default: config attacks.n_images)")
    ap.add_argument("--eps", type=float, default=8.0, help="L-inf budget in /255 units (pgd, transfer)")
    ap.add_argument("--train-source", default="dataset/clean/dev", help="patch training images")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    acfg = cfg["attacks"]
    seed = acfg["seed"]
    n = args.n or acfg["n_images"]
    m = cfg["model"]
    target = YOLOv8nDetector(m["weights"], device=args.device or m["device"], conf_thres=m["conf_thres"],
                             iou_thres=m["iou_thres"], max_det=m["max_det"], feature_layer=None, fuse=m["fuse"])
    device = target.device
    index = image_index(resolve_path(args.source))
    if args.ids:
        ids = read_ids(resolve_path(args.ids))
    else:
        print(f"Selecting {n} images with at least one clean detection...")
        ids = select_ids(target, index, cfg, n, seed)
    if args.save_ids:
        p = resolve_path(args.save_ids)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"image_id": ids}).to_csv(p, index=False)
        print(f"Saved {len(ids)} ids to {p}")

    eps = args.eps / 255.0
    attack_net, patch = None, None
    if args.attack == "pgd":
        attack_net = target.net
    elif args.attack == "transfer":
        surrogate = YOLOv8nDetector(acfg["transfer"]["surrogate_weights"], device=device, feature_layer=None,
                                    fuse=m["fuse"])
        attack_net = surrogate.net
    elif args.attack == "patch":
        pc = acfg["patch"]
        train_index = image_index(resolve_path(args.train_source))
        train_ids = list(train_index)[: pc["n_train"]]
        print(f"Training patch on {len(train_ids)} images from {args.train_source}...")
        imgs = [load_letterboxed(train_index[i], cfg) for i in train_ids]
        patch = train_patch(target.net, imgs, pc["size"], pc["steps"], pc["lr"], pc["batch"],
                            acfg["conf_floor"], seed, device)
    else:  # patch_random
        patch = random_patch(acfg["patch"]["size"], seed, device)

    out_img = resolve_path("dataset/adversarial") / args.name
    out_res = resolve_path("results") / args.name
    out_img.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)
    if patch is not None:
        import cv2
        fig_dir = resolve_path("results/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        p = (patch.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        cv2.imwrite(str(fig_dir / f"{args.name}_patch.png"), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))

    rows, t0 = [], time.time()
    succ = acfg["success"]
    for k, image_id in enumerate(ids, 1):
        lb, info = load_letterboxed(index[image_id], cfg)
        x = to_tensor(lb, device)
        if args.attack in ("pgd", "transfer"):
            x_adv = pgd_attack(attack_net, x, content_mask(info, device), eps, acfg["pgd"]["steps"],
                               eps * acfg["pgd"]["alpha_frac"], acfg["conf_floor"], seed + k)
        else:
            pos = fixed_position(image_id, info, patch.shape[-1], seed)
            if pos is None:
                print(f"  skip {image_id}: too small for the patch")
                continue
            x_adv = apply_patch(x, patch, [pos])
        lb_adv = to_uint8(x_adv)   # exactly what will be saved
        save_content_png(lb_adv, info, out_img / image_id)

        clean_dets, _ = target(stack_views([lb]))
        adv_dets, _ = target(stack_views([lb_adv]))
        n_clean, n_kept = detection_change(clean_dets[0], adv_dets[0], succ["iou"])
        diff = np.abs(lb_adv.astype(int) - lb.astype(int))
        rows.append({"image_id": image_id, "n_clean_det": n_clean, "n_adv_det": len(adv_dets[0]),
                     "n_kept": n_kept, "success": is_success(n_clean, n_kept, succ["lost_fraction"]),
                     "linf_255": int(diff.max()), "mean_abs_255": float(diff.mean())})
        if k % 10 == 0 or k == len(ids):
            sr = np.mean([r["success"] for r in rows])
            print(f"  {k}/{len(ids)} images, success so far {sr:.1%} ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_res / "attack_success.csv", index=False)
    config = {"attack": args.attack, "name": args.name, "eps_255": args.eps if args.attack in ("pgd", "transfer") else None,
              "n_images": len(df), "source": args.source, "device": device, "seed": seed,
              "attack_config": acfg, "torch": torch.__version__,
              "attack_success_rate": float(df["success"].mean()) if len(df) else None}
    with open(out_res / "attack_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Attack success rate: {config['attack_success_rate']:.1%} on {len(df)} images")
    print(f"Images: {out_img}\nResults: {out_res}")


if __name__ == "__main__":
    main()
