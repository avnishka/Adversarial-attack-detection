"""Shared attack utilities: loading in model space, the vanishing loss, saving, attack success.

All attacks work in the SAME 640x640 letterboxed space as the screening pipeline and
only change the content region. Results are saved as PNG crops of the content region
(no padding), so re-letterboxing them reproduces exactly the attacked model input.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.box_matching import match_detections
from src.preprocessing import LetterboxInfo, letterbox, load_rgb


def load_letterboxed(path: str | Path, cfg: dict) -> tuple[np.ndarray, LetterboxInfo]:
    return letterbox(load_rgb(path), cfg["model"]["imgsz"], cfg["preprocessing"]["pad_value"])


def to_tensor(lb: np.ndarray, device: str) -> torch.Tensor:
    """(H,W,3) uint8 -> (1,3,H,W) float in [0,1]."""
    return torch.from_numpy(lb).to(device).permute(2, 0, 1).float().div(255.0).unsqueeze(0).contiguous()


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) float in [0,1] -> (H,W,3) uint8 (rounded, as it will be saved)."""
    return (x.detach()[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())


def content_mask(info: LetterboxInfo, device: str) -> torch.Tensor:
    m = torch.zeros(1, 1, info.size, info.size, device=device)
    rows, cols = info.content_slice
    m[..., rows, cols] = 1.0
    return m


def vanishing_loss(preds, conf_floor: float = 0.1) -> torch.Tensor:
    """Sum of the best class score of every anchor still above `conf_floor`, averaged over the batch.
    Minimising it makes objects disappear. Works for YOLOv8 and YOLOv5u heads:
    eval-mode output is (B, 4 + num_classes, num_anchors) with sigmoid class scores."""
    y = preds[0] if isinstance(preds, (list, tuple)) else preds
    conf = y[:, 4:, :].amax(dim=1)
    return (conf * (conf > conf_floor)).sum() / y.shape[0]


def save_content_png(lb: np.ndarray, info: LetterboxInfo, path: str | Path) -> None:
    rows, cols = info.content_slice
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(lb[rows, cols], cv2.COLOR_RGB2BGR))


def detection_change(clean, adv, iou: float = 0.5) -> tuple[int, int]:
    """(clean detections, clean detections still present: same class, IoU >= iou)."""
    m = match_detections(clean.boxes, clean.classes, adv.boxes, adv.classes, iou_min=iou, class_aware=True)
    return len(clean), len(m.matches)


def is_success(n_clean: int, n_kept: int, lost_fraction: float = 0.5) -> bool:
    return n_clean > 0 and (n_clean - n_kept) / n_clean >= lost_fraction
