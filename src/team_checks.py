"""Small helpers shared by the member_*.py runners: status rows and the final report."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT

Row = tuple[str, str, str]   # (status, item, detail); status in PASS WARN FAIL TODO INFO

# Attack sets in the order the paper presents them
ATTACK_SETS = ["pgd_eps4", "pgd_eps8", "transfer_eps8", "patch", "patch_random"]
LABELS = {"pgd_eps4": "PGD $\\epsilon$=4/255", "pgd_eps8": "PGD $\\epsilon$=8/255",
          "transfer_eps8": "Transfer $\\epsilon$=8/255", "patch": "Adversarial patch",
          "patch_random": "Random patch (control)", "clean": "Clean"}
PLAIN_LABELS = {k: v.replace("$\\epsilon$", "eps") for k, v in LABELS.items()}


def run_script(script: str, *args, cwd: Path = PROJECT_ROOT) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *map(str, args)]
    print("\n$ python scripts/" + script + (" " + " ".join(map(str, args)) if args else ""))
    subprocess.run(cmd, cwd=cwd, check=True)


def read_ids_csv(path: Path) -> list[str] | None:
    return pd.read_csv(path, dtype={"image_id": str})["image_id"].tolist() if path.exists() else None


def report(rows: list[Row], done_message: str) -> int:
    """Print the status table and the next step; return the exit code (1 if any FAIL)."""
    width = max(len(r[1]) for r in rows) if rows else 0
    for status, item, detail in rows:
        print(f"[{status:<4}] {item:<{width}}  {detail}")
    fails = [r for r in rows if r[0] == "FAIL"]
    todo = next((r for r in rows if r[0] == "TODO"), None)
    print()
    if fails:
        print(f"{len(fails)} problem(s) to fix before using these results.")
    if todo:
        print(f"Next step: {todo[2]}")
    elif not fails:
        print(done_message)
    return 1 if fails else 0
