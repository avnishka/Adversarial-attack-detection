"""Three-view generation: original, JPEG, Gaussian blur (Section 10).

Transforms are applied to the content region of the *letterboxed* image, so:
  * their strength is the same at model resolution regardless of the dataset's
    native resolution (COCO, KITTI and BDD100K differ a lot), and
  * the grey padding is never touched (Phase 4 acceptance criterion).
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .preprocessing import LetterboxInfo

VIEW_NAMES = ("original", "jpeg", "blur")


def jpeg_compress(img: np.ndarray, quality: int = 75) -> np.ndarray:
    """Encode/decode an RGB uint8 image as JPEG at the given quality (1-100)."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur with a kernel covering +/-3 sigma. Borders are reflected,
    so when applied to the content crop no padding colour leaks in."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    k = 2 * math.ceil(3 * sigma) + 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def apply_to_content(lb_img: np.ndarray, info: LetterboxInfo,
                     fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply `fn` only to the content region; return a new image with padding untouched."""
    rows, cols = info.content_slice
    out = lb_img.copy()
    content = np.ascontiguousarray(lb_img[rows, cols])
    transformed = fn(content)
    if transformed.shape != content.shape or transformed.dtype != np.uint8:
        raise RuntimeError("Transform changed shape or dtype")
    out[rows, cols] = transformed
    return out


def make_views(lb_img: np.ndarray, info: LetterboxInfo, jpeg_quality: int = 75,
               blur_sigma: float = 1.0) -> dict[str, np.ndarray]:
    """Return the three views, in VIEW_NAMES order. The input is not modified."""
    return {
        "original": lb_img.copy(),
        "jpeg": apply_to_content(lb_img, info, lambda x: jpeg_compress(x, jpeg_quality)),
        "blur": apply_to_content(lb_img, info, lambda x: gaussian_blur(x, blur_sigma)),
    }
