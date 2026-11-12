"""Frozen YOLOv8n wrapper: one batched forward pass -> detections + hooked features
(Sections 7, 10.4, 11).

We call the underlying PyTorch DetectionModel directly and run NMS ourselves,
rather than `YOLO.predict()`. That guarantees:
  * exactly one forward pass per batch (no hidden warm-up passes that pollute the hook),
  * our own letterbox geometry is used unchanged,
  * all views go through the network together, as the proposal requires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

try:  # newer Ultralytics releases
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # older 8.x releases
    from ultralytics.utils.ops import non_max_suppression

from .config import resolve_path
from .feature_extractor import FeatureHook
from .preprocessing import LetterboxInfo, boxes_to_original


@dataclass
class Detections:
    """Detections for one view, in letterboxed (model-input) pixel coordinates."""
    boxes: np.ndarray    # (N,4) float32 xyxy
    scores: np.ndarray   # (N,)  float32
    classes: np.ndarray  # (N,)  int64
    names: dict[int, str] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.scores)

    def to_records(self, info: LetterboxInfo | None = None) -> list[dict]:
        """JSON-friendly list; boxes mapped to original-image pixels if `info` is given."""
        boxes = boxes_to_original(self.boxes, info) if info is not None else self.boxes
        return [{"class_id": int(c), "class_name": self.names.get(int(c), str(int(c))),
                 "confidence": float(s), "box_xyxy": [float(v) for v in b]}
                for b, s, c in zip(boxes, self.scores, self.classes)]


def resolve_device(device: str = "auto") -> str:
    """"auto" picks an NVIDIA GPU (cuda), then an Apple-silicon GPU (mps), then the CPU."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class YOLOv8nDetector:
    def __init__(self, weights: str | Path = "models/yolov8n.pt", device: str = "auto",
                 conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 300,
                 feature_layer: int | None = 9, fuse: bool = True):
        weights = resolve_path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} not found. Download it with:\n"
                f"  cd models && python -c \"from ultralytics import YOLO; YOLO('{weights.name}')\"")
        self.device = resolve_device(device)
        self.conf_thres, self.iou_thres, self.max_det = conf_thres, iou_thres, max_det

        self.net = YOLO(str(weights)).model.to(self.device).float().eval()
        for p in self.net.parameters():
            p.requires_grad_(False)  # detector stays frozen (Section 4)
        if fuse:
            self.net.fuse(verbose=False)
        self.names: dict[int, str] = dict(self.net.names)

        self.feature_layer = feature_layer
        self.hook = FeatureHook(self.net.model[feature_layer]) if feature_layer is not None else None

    @classmethod
    def from_config(cls, cfg: dict, device: str | None = None) -> "YOLOv8nDetector":
        """`device` (e.g. "cpu", "mps", "cuda") overrides the config without editing it."""
        m = cfg["model"]
        return cls(weights=m["weights"], device=device or m.get("device", "auto"),
                   conf_thres=m["conf_thres"], iou_thres=m["iou_thres"], max_det=m["max_det"],
                   feature_layer=m.get("feature_layer"), fuse=m.get("fuse", True))

    def to_tensor(self, batch: np.ndarray) -> torch.Tensor:
        """(B,H,W,3) uint8 RGB -> (B,3,H,W) float32 in [0,1] on device. The only normalisation step."""
        if batch.ndim != 4 or batch.shape[-1] != 3 or batch.dtype != np.uint8:
            raise ValueError(f"Expected (B,H,W,3) uint8 batch, got {batch.shape} {batch.dtype}")
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        return x.permute(0, 3, 1, 2).float().div_(255.0).contiguous()

    @torch.no_grad()
    def __call__(self, batch: np.ndarray) -> tuple[list[Detections], torch.Tensor | None]:
        """Run ONE forward pass over the whole batch.

        Returns one `Detections` per image and the hooked feature map (B,C,h,w), or None.
        """
        x = self.to_tensor(batch)
        if self.hook is not None:
            self.hook.reset()
        preds = self.net(x)
        feats = self.hook.pop(expected_batch=x.shape[0]) if self.hook is not None else None

        nms_out = non_max_suppression(preds, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                      max_det=self.max_det)
        dets = []
        for d in nms_out:
            d = d.detach().cpu().numpy()
            dets.append(Detections(boxes=d[:, :4].astype(np.float32),
                                   scores=d[:, 4].astype(np.float32),
                                   classes=d[:, 5].astype(np.int64),
                                   names=self.names))
        return dets, feats

    def synchronize(self) -> None:
        """Call before reading a timer so GPU work is actually finished."""
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
