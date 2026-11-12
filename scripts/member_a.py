"""Member A's whole workflow in one command (Zenbook, Windows, CPU).

Run the steps in this order; each one is safe to re-run (finished work is skipped):

    python scripts/member_a.py download     # Thu: COCO val2017 + annotations into dataset/raw (resumable)
    python scripts/member_a.py prepare      # Thu: road-scene subset + splits, then checks them
    python scripts/member_a.py pack         # Thu: dataset/team_pack.zip for Google Drive
    python scripts/member_a.py select --adv dataset/adversarial/pgd_dev_eps8 --apply   # Fri 9:30
    python scripts/member_a.py calibrate    # Fri: after the config is committed and tagged frozen-config
    python scripts/member_a.py screen       # Fri/Sat: clean test + every attack set in dataset/adversarial
    python scripts/member_a.py runtime      # Sat: plugged in, other apps closed
    python scripts/member_a.py check        # any time: what is done, what is wrong, what is next

All model runs use --device cpu, so every screening number in the paper comes from one machine.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import load_calibration, method_scores  # noqa: E402
from src.config import load_config  # noqa: E402
from src.io_utils import IMG_EXTS, read_scores  # noqa: E402
from src.run_meta import config_hash, read_meta  # noqa: E402

COCO_URLS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
RAW = ROOT / "dataset" / "raw"
CLEAN = ROOT / "dataset" / "clean"
ADV = ROOT / "dataset" / "adversarial"
RESULTS = ROOT / "results"
SPLITS = ROOT / "splits"
ANN = RAW / "annotations" / "instances_val2017.json"
CAL = RESULTS / "calibration" / "calibration.json"
FROZEN_TAG = "frozen-config"
SPLIT_NAMES = ("dev", "calibration", "test")


# ----------------------------------------------------------------------------- helpers
def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
    print("\n$ python " + " ".join(cmd[1:]).replace(str(ROOT) + "/", "").replace(str(ROOT) + "\\", ""))
    subprocess.run(cmd, cwd=ROOT, check=True)


def n_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS) if folder.is_dir() else 0


def git_tag_exists(tag: str) -> bool | None:
    """True/False, or None when git is unavailable or this is not a repo."""
    try:
        out = subprocess.run(["git", "tag", "-l", tag], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() == tag


def apply_selection(config_text: str, layer: int, mode: str) -> str:
    """Set model.feature_layer and features.mode in the YAML text, keeping every comment."""
    new, n1 = re.subn(r"^(\s*feature_layer:\s*)\S+", rf"\g<1>{int(layer)}", config_text, flags=re.M)
    new, n2 = re.subn(r"^(\s+mode:\s*)(gap|flatten)\b", rf"\g<1>{mode}", new, flags=re.M)
    if n1 != 1 or n2 != 1:
        raise ValueError(f"Could not find exactly one feature_layer ({n1}) and one features.mode ({n2}) line")
    return new


def screening_targets(adv_root: Path = ADV, clean_test: Path = CLEAN / "test") -> list[tuple[str, Path]]:
    """clean_test first, then every attack set folder. Dev sets (name contains '_dev') are for
    parameter selection only and are never part of the test results."""
    targets = [("clean_test", clean_test)]
    if adv_root.is_dir():
        for d in sorted(p for p in adv_root.iterdir() if p.is_dir()):
            if "_dev" not in d.name and n_images(d) > 0:
                targets.append((d.name, d))
    return targets


# ----------------------------------------------------------------------------- download
def download(url: str, dest: Path) -> None:
    """Resumable download with a progress line. A partial file (.part) is continued."""
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if done and r.status != 206:      # server ignored the range: start again
            done = 0
        total = done + int(r.headers.get("Content-Length", 0))
        mode = "ab" if done else "wb"
        t0, last = time.time(), 0.0
        with open(part, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 2:
                    last = time.time()
                    speed = (done / 1e6) / max(last - t0, 1e-6)
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    print(f"\r  {dest.name}: {done / 1e6:7.1f} MB {pct} ({speed:.1f} MB/s)   ", end="", flush=True)
    print()
    part.replace(dest)


def cmd_download(args) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    val_dir = RAW / "val2017"
    if n_images(val_dir) >= 5000:
        print(f"Images already there: {val_dir} ({n_images(val_dir)} files)")
    else:
        z = RAW / "val2017.zip"
        if not z.exists():
            print("Downloading COCO val2017 images (about 1 GB; re-run to resume if it stops)...")
            download(COCO_URLS["val2017.zip"], z)
        print("Unzipping images...")
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if m.startswith("val2017/") and m.lower().endswith(".jpg")]
            if len(members) != 5000:
                sys.exit(f"{z} looks incomplete ({len(members)} images, expected 5000). Delete it and re-run.")
            zf.extractall(RAW, members)
        if not args.keep_zips:
            z.unlink()
    if ANN.exists():
        print(f"Annotations already there: {ANN}")
    else:
        z = RAW / "annotations_trainval2017.zip"
        if not z.exists():
            print("Downloading COCO annotations (about 250 MB)...")
            download(COCO_URLS["annotations_trainval2017.zip"], z)
        with zipfile.ZipFile(z) as zf:
            zf.extract("annotations/instances_val2017.json", RAW)
        if not args.keep_zips:
            z.unlink()
    print(f"Done: {n_images(val_dir)} images and {ANN.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- dataset
def cmd_prepare(args) -> None:
    if not ANN.exists() or n_images(RAW / "val2017") < 5000:
        sys.exit("COCO is not downloaded yet. Run: python scripts/member_a.py download")
    run("prepare_dataset.py")
    problems = [r for r in check_splits() if r[0] == "FAIL"]
    for r in check_splits():
        print(f"  [{r[0]}] {r[1]}: {r[2]}")
    if problems:
        sys.exit("Fix the problems above before continuing.")
    print("\nNext: commit the splits (git add splits/ ; git commit -m \"Add dataset splits\"), "
          "then run: python scripts/member_a.py pack")


def cmd_pack(args) -> None:
    if not all((CLEAN / s).is_dir() for s in SPLIT_NAMES):
        sys.exit("Run prepare first.")
    out = ROOT / "dataset" / "team_pack.zip"
    files = [p for s in SPLIT_NAMES for p in sorted((CLEAN / s).iterdir()) if p.suffix.lower() in IMG_EXTS]
    files.append(ANN)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:   # JPEGs do not compress further
        for p in files:
            zf.write(p, p.relative_to(ROOT).as_posix())
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB, {len(files)} files).")
    print("Upload it to the shared Drive folder. Teammates unzip it in the repo folder "
          "(the zip already contains the dataset/ paths).")


# ----------------------------------------------------------------------------- settings
def cmd_select(args) -> None:
    adv = ROOT / args.adv
    if n_images(adv) == 0:
        sys.exit(f"No images in {args.adv}. Download B's pgd_dev_eps8 folder from Drive into dataset/adversarial/.")
    if args.apply and git_tag_exists(FROZEN_TAG):
        sys.exit(f"The config is already frozen (tag {FROZEN_TAG}). Changing it now would invalidate the "
                 "calibration and every scores.csv. Ask the team first.")
    run("select_params.py", "--adv", args.adv, "--device", "cpu", "--layers", *args.layers)
    res = pd.read_csv(RESULTS / "tables" / "param_selection.csv")
    best = res.sort_values("auroc_full", ascending=False).iloc[0]
    layer, mode = int(best["layer"]), str(best["feat_mode"])
    if not args.apply:
        print(f"\nBest on dev: feature_layer={layer}, mode={mode}. Re-run with --apply to write it into the config.")
        return
    cfg_path = ROOT / "configs" / "default.yaml"
    old = cfg_path.read_text(encoding="utf-8")
    new = apply_selection(old, layer, mode)
    cfg_path.write_text(new, encoding="utf-8")
    print(f"\nconfigs/default.yaml updated: feature_layer={layer}, features.mode={mode}")
    print("Now commit, merge the PR, and freeze it:\n"
          "  git checkout -b a/frozen-config\n"
          "  git add configs/default.yaml results/tables/param_selection.csv\n"
          "  git commit -m \"Freeze feature settings chosen on dev\"\n"
          "  git push -u origin a/frozen-config   (open the PR, merge it, then on main:)\n"
          "  git checkout main ; git pull ; git tag frozen-config ; git push --tags\n"
          "Then: python scripts/member_a.py calibrate")


def cmd_calibrate(args) -> None:
    if git_tag_exists(FROZEN_TAG) is False and not args.force:
        sys.exit(f"The config is not frozen yet (no tag {FROZEN_TAG}). Finish `select --apply` first, "
                 "or add --force to calibrate anyway.")
    if n_images(CLEAN / "calibration") == 0:
        sys.exit("No calibration images. Run prepare first.")
    run("run_calibration.py", "--device", "cpu")
    print("\nNext: commit results/calibration/calibration.json, then run: python scripts/member_a.py screen")


def cmd_screen(args) -> None:
    cfg = load_config()
    h = config_hash(cfg)
    cal = load_calibration(CAL) if CAL.exists() else None
    if cal is None:
        print("NOTE: no calibration.json yet; scores are saved without S/decision (evaluation adds them later).")
    elif cal.get("config_hash") not in (None, h):
        sys.exit("calibration.json was made with a different config than the current one. "
                 "Re-run calibrate (or restore the frozen config) first.")
    targets = screening_targets()
    if args.only:
        targets = [t for t in targets if t[0] in args.only]
    for name, folder in targets:
        out = RESULTS / name / "scores.csv"
        meta = read_meta(out.parent / "screening_meta.json")
        n = n_images(folder)
        if n == 0:
            print(f"skip {name}: no images in {folder.relative_to(ROOT)}")
            continue
        if out.exists() and meta and meta.get("config_hash") == h and meta.get("n_rows") == n and not args.force:
            print(f"done {name}: {n} images already screened with the current config")
            continue
        extra = ["--calibration", str(CAL.relative_to(ROOT))] if cal else []
        run("run_screening.py", "--source", folder.relative_to(ROOT), "--out", out.relative_to(ROOT),
            "--device", "cpu", *extra)
    print("\nNext: commit the new results/*/scores.csv and screening_meta.json files, "
          "then run: python scripts/member_a.py check")


def cmd_runtime(args) -> None:
    if n_images(CLEAN / "test") == 0:
        sys.exit("No test images. Run prepare first.")
    print("Plug the laptop in and close other apps (browser, Teams, OneDrive sync). Starting in 5 s...")
    time.sleep(5)
    run("run_runtime.py", "--device", "cpu", "--runs", args.runs)
    print("\nAdd the RAM size and laptop model to results/tables/runtime_env.csv, then commit both runtime files.")


# ----------------------------------------------------------------------------- checks
def check_splits() -> list[tuple[str, str, str]]:
    rows = []
    csvs = {s: SPLITS / f"{s}.csv" for s in SPLIT_NAMES}
    if not all(p.exists() for p in csvs.values()):
        return [("TODO", "splits", "not created yet: python scripts/member_a.py prepare")]
    ids = {s: set(pd.read_csv(p, dtype={"image_id": str})["image_id"]) for s, p in csvs.items()}
    overlap = (ids["dev"] & ids["calibration"]) | (ids["dev"] & ids["test"]) | (ids["calibration"] & ids["test"])
    rows.append(("FAIL" if overlap else "PASS", "splits disjoint",
                 f"{len(overlap)} shared images" if overlap else
                 " / ".join(f"{s} {len(ids[s])}" for s in SPLIT_NAMES)))
    for s in SPLIT_NAMES:
        have = {p.stem for p in (CLEAN / s).iterdir() if p.suffix.lower() in IMG_EXTS} if (CLEAN / s).is_dir() else set()
        missing = ids[s] - have
        rows.append(("FAIL" if missing else "PASS", f"images {s}",
                     f"{len(missing)} listed in splits/{s}.csv but missing in dataset/clean/{s}" if missing
                     else f"{len(have)} images present"))
    return rows


def run_checks() -> list[tuple[str, str, str]]:
    cfg = load_config()
    h = config_hash(cfg)
    rows = check_splits()

    # Settings
    sel = RESULTS / "tables" / "param_selection.csv"
    rows.append(("PASS" if sel.exists() else "TODO", "parameter selection",
                 "results/tables/param_selection.csv present" if sel.exists() else
                 "python scripts/member_a.py select --adv dataset/adversarial/pgd_dev_eps8 --apply"))
    tag = git_tag_exists(FROZEN_TAG)
    rows.append(({True: "PASS", False: "TODO", None: "WARN"}[tag], "config frozen",
                 {True: f"tag {FROZEN_TAG} exists", False: f"no tag {FROZEN_TAG} yet",
                  None: "could not ask git (not a repo, or git missing)"}[tag]))
    rows.append(("INFO", "current config", f"feature_layer={cfg['model']['feature_layer']}, "
                 f"features.mode={cfg['features']['mode']}, fingerprint {h}"))

    # Calibration
    cal = None
    if not CAL.exists():
        rows.append(("TODO", "calibration", "python scripts/member_a.py calibrate"))
    else:
        cal = load_calibration(CAL)
        ch = cal.get("config_hash")
        if ch is None:
            rows.append(("WARN", "calibration", "made by an older script (no fingerprint): re-run calibrate"))
        elif ch != h:
            rows.append(("FAIL", "calibration", f"fingerprint {ch} differs from the current config {h}: re-run calibrate"))
        else:
            rows.append(("PASS", "calibration", f"{cal['n_images']} images, fingerprint matches, "
                         f"device {cal.get('run', {}).get('device', '?')}"))
        if cal["n_images"] < 300:
            rows.append(("WARN", "calibration size", f"only {cal['n_images']} images; thresholds will be noisy"))

    # Screening outputs
    attack_ids = set(pd.read_csv(SPLITS / "attack_ids.csv", dtype={"image_id": str})["image_id"]) \
        if (SPLITS / "attack_ids.csv").exists() else None
    devices = set()
    for name, folder in screening_targets():
        out = RESULTS / name / "scores.csv"
        if not out.exists():
            rows.append(("TODO", f"screen {name}", "python scripts/member_a.py screen"))
            continue
        df = read_scores(out)
        meta = read_meta(out.parent / "screening_meta.json")
        status, notes = "PASS", [f"{len(df)} rows"]
        if meta is None:
            status = "WARN"; notes.append("no screening_meta.json (re-run screen)")
        else:
            devices.add(meta.get("device"))
            if meta.get("config_hash") != h:
                status = "FAIL"; notes.append(f"made with config {meta.get('config_hash')}, current is {h}")
            if meta.get("device") != "cpu":
                status = "WARN" if status == "PASS" else status; notes.append(f"device {meta.get('device')}")
        n = n_images(folder)
        if len(df) < n:
            status = "WARN" if status == "PASS" else status; notes.append(f"only {len(df)} of {n} images")
        dcols = [c for c in df.columns if c.startswith("D_")]
        if df[dcols].isna().any().any():
            status = "FAIL"; notes.append("missing values in score columns")
        if name != "clean_test" and attack_ids is not None and set(df["image_id"]) - attack_ids:
            status = "WARN" if status == "PASS" else status
            notes.append(f"{len(set(df['image_id']) - attack_ids)} images not in splits/attack_ids.csv")
        rows.append((status, f"screen {name}", ", ".join(notes)))
    if len(devices) > 1:
        rows.append(("FAIL", "one device", f"scores come from several devices: {sorted(map(str, devices))}"))

    # Clean false-positive rate: should be close to the calibration target
    clean = RESULTS / "clean_test" / "scores.csv"
    if cal is not None and clean.exists():
        s = method_scores(read_scores(clean), "full", cal["stats"])
        for f in cal["fprs"]:
            fpr = float(np.mean(s > cal["thresholds"]["full"][str(f)]))
            se = np.sqrt(f * (1 - f) / max(len(s), 1))
            ok = fpr <= f + 3 * se + 0.01
            rows.append(("PASS" if ok else "WARN", f"clean FPR at {f:.0%}",
                         f"{fpr:.1%} on {len(s)} clean test images"
                         + ("" if ok else " (much higher than the target: calibration and test may differ)")))

    # Runtime
    rt = RESULTS / "tables" / "runtime.csv"
    rows.append(("PASS" if rt.exists() else "TODO", "runtime",
                 "results/tables/runtime.csv present" if rt.exists() else "python scripts/member_a.py runtime"))
    return rows


def cmd_check(args) -> None:
    rows = run_checks()
    width = max(len(r[1]) for r in rows)
    for status, item, detail in rows:
        print(f"[{status:<4}] {item:<{width}}  {detail}")
    todo = next((r for r in rows if r[0] == "TODO"), None)
    fails = [r for r in rows if r[0] == "FAIL"]
    print()
    if fails:
        print(f"{len(fails)} problem(s) to fix before using these results.")
    if todo:
        print(f"Next step: {todo[2]}")
    elif not fails:
        print("Everything for Member A is done.")
    sys.exit(1 if fails else 0)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="COCO val2017 images + annotations")
    p.add_argument("--keep-zips", action="store_true", help="keep the downloaded zip files")
    sub.add_parser("prepare", help="build the road-scene subset and splits")
    sub.add_parser("pack", help="zip the dataset for Google Drive")
    p = sub.add_parser("select", help="choose feature layer and mode on dev data")
    p.add_argument("--adv", default="dataset/adversarial/pgd_dev_eps8")
    p.add_argument("--layers", nargs="+", default=["9", "15", "21"])
    p.add_argument("--apply", action="store_true", help="write the best setting into configs/default.yaml")
    p = sub.add_parser("calibrate", help="thresholds from the clean calibration split")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("screen", help="screen clean test + every attack set")
    p.add_argument("--only", nargs="+", help="only these set names, e.g. clean_test pgd_eps8")
    p.add_argument("--force", action="store_true", help="re-screen even if already done")
    p = sub.add_parser("runtime", help="latency measurement")
    p.add_argument("--runs", default="200")
    sub.add_parser("check", help="status and sanity checks")
    args = ap.parse_args()
    {"download": cmd_download, "prepare": cmd_prepare, "pack": cmd_pack, "select": cmd_select,
     "calibrate": cmd_calibrate, "screen": cmd_screen, "runtime": cmd_runtime, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
