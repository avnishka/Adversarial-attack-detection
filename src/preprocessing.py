"""Image loading and letterbox preprocessing (Section 9).

We do the letterboxing ourselves instead of letting Ultralytics do it, because:
  * the three views must share exactly the same geometry, and
  * transforms must touch only the image content, never the grey padding.
`LetterboxInfo` records where the content sits so boxes can be mapped back.

Convention throughout the project: images are HxWx3 uint8 **RGB** numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float      # resize factor applied to the original image
    pad_left: int     # x offset of the content inside the letterboxed image
    pad_top: int      # y offset of the content
    content_w: int    # width of the resized content
    content_h: int    # height of the resized content
    orig_w: int
    orig_h: int
    size: int         # side of the square letterboxed image

    @property
    def content_slice(self) -> tuple[slice, slice]:
        """(rows, cols) slice of the content region inside the letterboxed image."""
        return (slice(self.pad_top, self.pad_top + self.content_h),
                slice(self.pad_left, self.pad_left + self.content_w))


def load_rgb(path: str | Path) -> np.ndarray:
    """Load any image file as an HxWx3 uint8 RGB array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # OpenCV gives BGR
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def letterbox(img: np.ndarray, size: int = 640, pad_value: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio, then pad to size x size. Does not modify `img`."""
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 RGB image, got {img.shape} {img.dtype}")
    if size % 32 != 0:
        raise ValueError("size must be a multiple of 32 for YOLO")

    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w = min(size, max(1, int(round(w * scale))))
    new_h = min(size, max(1, int(round(h * scale))))

    if (new_w, new_h) != (w, h):
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    else:
        resized = img.copy()

    left = (size - new_w) // 2
    top = (size - new_h) // 2
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    out[top:top + new_h, left:left + new_w] = resized

    info = LetterboxInfo(scale=scale, pad_left=left, pad_top=top, content_w=new_w,
                         content_h=new_h, orig_w=w, orig_h=h, size=size)
    return out, info


def boxes_to_original(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map (N,4) xyxy boxes from letterboxed coordinates back to original-image pixels."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_left) / info.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_top) / info.scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.orig_h)
    return boxes


def boxes_to_letterbox(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Inverse of boxes_to_original (useful for mapping ground-truth annotations)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * info.scale + info.pad_left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * info.scale + info.pad_top
    return boxes


def stack_views(views: list[np.ndarray]) -> np.ndarray:
    """Stack equally sized HxWx3 views into a (B,H,W,3) uint8 batch."""
    shapes = {v.shape for v in views}
    if len(shapes) != 1:
        raise ValueError(f"All views must have the same shape, got {shapes}")
    return np.stack(views, axis=0)
