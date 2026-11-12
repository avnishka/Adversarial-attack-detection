"""Member B's whole workflow in one command (MacBook Air M2; Colab as a fallback).

Run the steps in this order; each one is safe to re-run (finished work is skipped):

    python scripts/member_b.py setup        # Thu: checks MPS, weights and dataset; downloads yolov5nu.pt
    python scripts/member_b.py dev          # Thu: 50 dev PGD images for A's parameter selection
    python scripts/member_b.py pack pgd_dev_eps8              # zip it for Drive
    python scripts/member_b.py attacks      # Thu night/Fri: all 5 test sets on the same 200 images
    python scripts/member_b.py pack --all   # Fri: one zip per finished set, for Drive
    python scripts/member_b.py map          # Sat: mAP of YOLOv8n on clean vs attacked images
    python scripts/member_b.py qualitative  # Sat: example figure for the paper
    python scripts/member_b.py check        # any time: what is done, what is wrong, what is next

Attacks run on the Mac GPU (mps) by default. Screening is NOT done here: A screens every set
on one CPU so the paper's numbers come from one machine.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.io_utils import IMG_EXTS  # noqa: E402
from src.team_checks import ATTACK_SETS, PLAIN_LABELS, Row, read_ids_csv, report, run_script  # noqa: E402

CLEAN = ROOT / "dataset" / "clean"
ADV = ROOT / "dataset" / "adversarial"
UPLOAD = ROOT / "dataset" / "upload"
RESULTS = ROOT / "results"
SPLITS = ROOT / "splits"
IDS = SPLITS / "attack_ids.csv"
DEV_SET = "pgd_dev_eps8"
DEV_N = 50

# name -> (run_attacks.py arguments, L-inf budget in /255 or None)
ATTACKS = {
    "pgd_eps8": (["--attack", "pgd", "--eps", "8"], 8),
    "transfer_eps8": (["--attack", "transfer", "--eps", "8"], 8),
    "patch": (["--attack", "patch"], None),
    "pgd_eps4": (["--attack", "pgd", "--eps", "4"], 4),
    "patch_random": (["--attack", "patch_random"], None),
}
RUN_ORDER = list(ATTACKS)   # most important first, so a short night still gives the core sets


def pngs(folder: Path) -> set[str]:
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS} if folder.is_dir() else set()


def set_done(name: str, ids: list[str] | None) -> bool:
    succ = RESULTS / name / "attack_success.csv"
    if not succ.exists():
        return False
    got = pd.read_csv(succ, dtype={"image_id": str})["image_id"].tolist()
    want = ids if ids is not None else got
    return len(got) > 0 and set(got) == set(want) and set(got) <= pngs(ADV / name)


def device_args(args) -> list[str]:
    return ["--device", args.device] if args.device else []


# ----------------------------------------------------------------------------- steps
def cmd_setup(args) -> None:
    rows: list[Row] = []
    try:
        import torch
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        cuda = torch.cuda.is_available()
        rows.append(("PASS", "PyTorch", f"{torch.__version__}; mps={mps}, cuda={cuda}"))
        if not (mps or cuda):
            rows.append(("WARN", "GPU", "no mps/cuda: attacks will run on CPU and be slow; consider Colab"))
    except ImportError:
        rows.append(("FAIL", "PyTorch", "not installed in this environment (activate .venv?)"))
    for w in ("yolov8n.pt", "yolov5nu.pt"):
        p = ROOT / "models" / w
        if not p.exists():
            print(f"Downloading {w} into models/ ...")
            subprocess.run([sys.executable, "-c", f"from ultralytics import YOLO; YOLO('{w}')"],
                           cwd=ROOT / "models", check=False)
        rows.append(("PASS" if p.exists() else "FAIL", f"weights {w}", "present" if p.exists() else "download failed"))
    for s in ("dev", "test"):
        n = len(pngs(CLEAN / s))
        rows.append(("PASS" if n else "TODO", f"dataset {s}",
                     f"{n} images" if n else "unzip A's team_pack.zip in the repo folder"))
    sys.exit(report(rows, "Setup complete. Next: python scripts/member_b.py dev"))


def cmd_dev(args) -> None:
    if not pngs(CLEAN / "dev"):
        sys.exit("No dev images. Unzip A's team_pack.zip in the repo folder first.")
    if set_done(DEV_SET, None) and not args.force:
        print(f"{DEV_SET} already done. Next: python scripts/member_b.py pack {DEV_SET}")
        return
    run_script("run_attacks.py", "--attack", "pgd", "--eps", "8", "--source", "dataset/clean/dev",
               "--n", DEV_N, "--name", DEV_SET, *device_args(args))
    print(f"\nNext: python scripts/member_b.py pack {DEV_SET}  (A needs it first thing tomorrow)")


def cmd_attacks(args) -> None:
    if not pngs(CLEAN / "test"):
        sys.exit("No test images. Unzip A's team_pack.zip in the repo folder first.")
    if not (ROOT / "models" / "yolov5nu.pt").exists():
        sys.exit("models/yolov5nu.pt is missing (needed for the transfer attack). Run: python scripts/member_b.py setup")
    names = [n for n in RUN_ORDER if not args.only or n in args.only]
    for name in names:
        ids = read_ids_csv(IDS)
        if ids is not None and set_done(name, ids) and not args.force:
            print(f"done {name}: {len(ids)} images")
            continue
        base, _ = ATTACKS[name]
        if ids is None:   # first set chooses the 200 test images and saves them for every other set
            sel = ["--save-ids", "splits/attack_ids.csv"]
        else:
            sel = ["--ids", "splits/attack_ids.csv"]
        run_script("run_attacks.py", *base, *sel, "--name", name, *device_args(args))
        print(f"Finished {name}. Pack it for A: python scripts/member_b.py pack {name}")
    print("\nNext: python scripts/member_b.py check, then commit splits/attack_ids.csv and results/<set>/ files.")


def pack_set(name: str, adv_root: Path = ADV, upload: Path = UPLOAD, root: Path = ROOT) -> Path:
    folder = adv_root / name
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS) if folder.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    upload.mkdir(parents=True, exist_ok=True)
    out = upload / f"{name}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:   # PNGs: lossless, keep exact pixels
        for p in files:
            zf.write(p, p.relative_to(root).as_posix())
    return out


def cmd_pack(args) -> None:
    names = [n for n in [DEV_SET] + RUN_ORDER if pngs(ADV / n)] if args.all else args.names
    if not names:
        sys.exit("Nothing to pack. Name a set (e.g. pgd_eps8) or use --all.")
    for n in names:
        out = pack_set(n)
        print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")
    print("Upload the zip(s) to the Drive folder 'adversarial/'. A unzips each one in the repo folder "
          "(the zip already contains dataset/adversarial/<set>/).")


def cmd_map(args) -> None:
    try:
        import pycocotools  # noqa: F401
    except ImportError:
        sys.exit("Install it first: pip install pycocotools")
    if not (ROOT / "dataset/raw/annotations/instances_val2017.json").exists():
        sys.exit("Missing dataset/raw/annotations/instances_val2017.json (it is inside A's team_pack.zip).")
    sets = [n for n in RUN_ORDER if pngs(ADV / n)]
    run_script("run_detection_eval.py", "--sets", *sets, *device_args(args))
    print("\nCommit results/tables/detection_map.csv.")


# ----------------------------------------------------------------------------- qualitative figure
def choose_example(success: dict[str, pd.DataFrame], ids: list[str]) -> str:
    """An image where PGD and the patch both succeeded and the clean image has 2-6 detections."""
    ok = set(ids)
    for name in ("pgd_eps8", "patch"):
        if name in success:
            df = success[name]
            ok &= set(df.loc[df["success"].astype(str).str.lower() == "true", "image_id"])
    if ok and "pgd_eps8" in success:
        df = success["pgd_eps8"].set_index("image_id")
        good = [i for i in ids if i in ok and 2 <= int(df.loc[i, "n_clean_det"]) <= 6]
        if good:
            return good[0]
    return next((i for i in ids if i in ok), ids[0])


def draw_panel(ax, img: np.ndarray, records: list[dict], title: str) -> None:
    from matplotlib.patches import Rectangle
    ax.imshow(img)
    for r in records:
        x1, y1, x2, y2 = r["box_xyxy"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#2a78d6", linewidth=1.2))
        ax.text(x1, y1 - 2, f"{r['class_name']} {r['confidence']:.2f}", fontsize=5, color="white",
                bbox={"facecolor": "#2a78d6", "edgecolor": "none", "pad": 0.8})
    ax.set_title(title, fontsize=7, loc="left")
    ax.axis("off")


def score_text(name: str, image_id: str) -> str:
    """Screening score and decision for this image, if A's results are available (git pull)."""
    from src.calibration import decide, load_calibration, method_scores
    cal_p = RESULTS / "calibration" / "calibration.json"
    sc_p = RESULTS / ("clean_test" if name == "clean" else name) / "scores.csv"
    if not (cal_p.exists() and sc_p.exists()):
        return ""
    cal = load_calibration(cal_p)
    df = pd.read_csv(sc_p, dtype={"image_id": str})
    row = df[df["image_id"] == image_id]
    if row.empty:
        return ""
    s = float(method_scores(row, "full", cal["stats"])[0])
    return f"S = {s:.2f} → {decide(s, cal).lower()}"


def cmd_qualitative(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.detector import YOLOv8nDetector
    from src.preprocessing import letterbox, load_rgb, stack_views

    ids = read_ids_csv(IDS)
    if ids is None:
        sys.exit("Run the attacks first (splits/attack_ids.csv is missing).")
    success = {n: pd.read_csv(RESULTS / n / "attack_success.csv", dtype={"image_id": str})
               for n in ("pgd_eps8", "patch") if (RESULTS / n / "attack_success.csv").exists()}
    image_id = args.id or choose_example(success, ids)
    cfg = load_config()
    m = cfg["model"]
    det = YOLOv8nDetector(m["weights"], device=args.device or m["device"], conf_thres=m["conf_thres"],
                          iou_thres=m["iou_thres"], max_det=m["max_det"], feature_layer=None, fuse=m["fuse"])
    clean_path = next(p for p in (CLEAN / "test").iterdir() if p.stem == image_id)
    panels = [("clean", clean_path)] + [(n, ADV / n / f"{image_id}.png") for n in ("pgd_eps8", "patch")
                                        if (ADV / n / f"{image_id}.png").exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.2))
    axes = np.atleast_1d(axes)
    for ax, (name, path) in zip(axes, panels):
        img = load_rgb(path)
        lb, info = letterbox(img, m["imgsz"], cfg["preprocessing"]["pad_value"])
        recs = det(stack_views([lb]))[0][0].to_records(info)
        title = f"{PLAIN_LABELS.get(name, name)}: {len(recs)} detections"
        st = score_text(name, image_id)
        draw_panel(ax, img, recs, title + (f"\n{st}" if st else ""))
    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"qualitative_{image_id}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Wrote results/figures/qualitative_{image_id}.pdf/.png (image {image_id}). "
          "Use --id <image_id> to pick another image.")


# ----------------------------------------------------------------------------- checks
def run_checks() -> list[Row]:
    rows: list[Row] = []
    split = {s: set(read_ids_csv(SPLITS / f"{s}.csv") or []) for s in ("dev", "calibration", "test")}
    if not split["test"]:
        return [("TODO", "splits", "git pull (A commits splits/*.csv), and unzip team_pack.zip")]
    cfg = load_config()

    # Dev set (parameter selection only)
    dev_ids = pngs(ADV / DEV_SET)
    if not dev_ids:
        rows.append(("TODO", DEV_SET, "python scripts/member_b.py dev"))
    else:
        bad = dev_ids - split["dev"]
        rows.append(("FAIL" if bad else "PASS", DEV_SET,
                     f"{len(bad)} images are not from the dev split" if bad else f"{len(dev_ids)} images, all from dev"))
        if not (UPLOAD / f"{DEV_SET}.zip").exists():
            rows.append(("TODO", f"pack {DEV_SET}", f"python scripts/member_b.py pack {DEV_SET}"))

    # The 200 test images
    ids = read_ids_csv(IDS)
    if ids is None:
        rows.append(("TODO", "attack images", "python scripts/member_b.py attacks"))
        return rows
    n_want = cfg["attacks"]["n_images"]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    if set(ids) - split["test"]:
        problems.append(f"{len(set(ids) - split['test'])} ids not in the test split")
    rows.append(("FAIL" if problems else ("PASS" if len(ids) == n_want else "WARN"), "attack_ids.csv",
                 "; ".join(problems) if problems else f"{len(ids)} test images (config wants {n_want})"))

    # Each set
    for name in RUN_ORDER:
        succ_p = RESULTS / name / "attack_success.csv"
        if not succ_p.exists():
            rows.append(("TODO", name, f"python scripts/member_b.py attacks --only {name}"))
            continue
        df = pd.read_csv(succ_p, dtype={"image_id": str})
        status, notes = "PASS", []
        if set(df["image_id"]) != set(ids):
            status = "FAIL"; notes.append("images differ from splits/attack_ids.csv")
        missing = set(df["image_id"]) - pngs(ADV / name)
        if missing:
            status = "FAIL"; notes.append(f"{len(missing)} PNGs missing in dataset/adversarial/{name}")
        eps = ATTACKS[name][1]
        if eps is not None and len(df) and int(df["linf_255"].max()) > eps:
            status = "FAIL"; notes.append(f"perturbation {int(df['linf_255'].max())}/255 exceeds the budget {eps}/255")
        if not (RESULTS / name / "attack_config.json").exists():
            status = "WARN" if status == "PASS" else status; notes.append("attack_config.json missing")
        asr = df["success"].astype(str).str.lower().eq("true").mean() if len(df) else 0.0
        notes.insert(0, f"{len(df)} images, attack success {asr:.0%}")
        if name == "pgd_eps8" and asr < 0.3:
            status = "WARN" if status == "PASS" else status
            notes.append("unusually low for white-box PGD: check it attacked YOLOv8n")
        if name == "patch_random":
            notes.append("occlusion control: expected well below the trained patch")
        if not (UPLOAD / f"{name}.zip").exists():
            notes.append(f"not packed yet (pack {name})")
        rows.append((status, name, ", ".join(notes)))

    # mAP and figures
    mp = RESULTS / "tables" / "detection_map.csv"
    rows.append(("PASS" if mp.exists() else "TODO", "mAP table",
                 "results/tables/detection_map.csv present" if mp.exists() else "python scripts/member_b.py map"))
    figs = list((RESULTS / "figures").glob("qualitative_*.pdf")) if (RESULTS / "figures").is_dir() else []
    rows.append(("PASS" if figs else "TODO", "qualitative figure",
                 figs[0].name if figs else "python scripts/member_b.py qualitative"))
    return rows


def cmd_check(args) -> None:
    sys.exit(report(run_checks(), "Everything for Member B is done."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default=None, help="default: auto (mps on the Mac); cuda on Colab; cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", parents=[common], help="check PyTorch/MPS, weights, dataset")
    p = sub.add_parser("dev", parents=[common], help="50 dev PGD images for parameter selection")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("attacks", parents=[common], help="all test attack sets")
    p.add_argument("--only", nargs="+", choices=RUN_ORDER)
    p.add_argument("--force", action="store_true", help="regenerate even if done")
    p = sub.add_parser("pack", parents=[common], help="zip attack sets for Drive")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    sub.add_parser("map", parents=[common], help="mAP on clean vs attacked images")
    p = sub.add_parser("qualitative", parents=[common], help="example figure for the paper")
    p.add_argument("--id", default=None)
    sub.add_parser("check", parents=[common], help="status and sanity checks")
    args = ap.parse_args()
    {"setup": cmd_setup, "dev": cmd_dev, "attacks": cmd_attacks, "pack": cmd_pack, "map": cmd_map,
     "qualitative": cmd_qualitative, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
