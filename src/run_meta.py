"""Run metadata: which config produced a result file, and on which device.

Every scores.csv gets a screening_meta.json beside it, and calibration.json stores the same
fingerprint, so `python scripts/member_a.py check` can prove that every number in the
paper came from the frozen config on one machine.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from pathlib import Path

CONFIG_SECTIONS = ("model", "preprocessing", "transforms", "matching", "features")


def config_hash(cfg: dict) -> str:
    """Fingerprint of every setting that changes screening scores. `model.device` is excluded:
    the device is recorded separately, because it does not change the method."""
    part = {k: cfg[k] for k in CONFIG_SECTIONS if k in cfg}
    if "model" in part:
        part["model"] = {k: v for k, v in part["model"].items() if k != "device"}
    return hashlib.sha256(json.dumps(part, sort_keys=True).encode()).hexdigest()[:12]


def run_info(device: str) -> dict:
    info = {"device": device, "host": platform.node(), "os": platform.platform(),
            "python": platform.python_version(), "created": _dt.datetime.now().isoformat(timespec="seconds")}
    try:
        import torch
        import ultralytics
        info.update(torch=torch.__version__, ultralytics=ultralytics.__version__)
    except ImportError:
        pass
    return info


def write_meta(path: str | Path, cfg: dict, device: str, **extra) -> None:
    meta = {"config_hash": config_hash(cfg), **run_info(device), **extra}
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def read_meta(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)
