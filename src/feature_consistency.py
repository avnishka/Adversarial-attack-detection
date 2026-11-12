"""Feature consistency (Section 12.3): D_feat = 1 - cosine(F_original, F_view).

Works on numpy arrays (C, h, w) so it can be tested without PyTorch.
  mode="gap"     : global average pool -> (C,) vector (robust to small shifts)
  mode="flatten" : the whole map as one vector (sensitive to spatial changes)
The mode is chosen on the development split (scripts/select_params.py), then frozen.
"""
from __future__ import annotations

import numpy as np

FEATURE_MODES = ("gap", "flatten")


def _vector(f: np.ndarray, mode: str) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    if mode == "gap":
        return f.reshape(f.shape[0], -1).mean(axis=1) if f.ndim > 1 else f
    if mode == "flatten":
        return f.ravel()
    raise ValueError(f"Unknown feature mode {mode!r}; use one of {FEATURE_MODES}")


def feature_distance(f_ref: np.ndarray, f_view: np.ndarray, mode: str = "gap") -> float:
    a, b = _vector(f_ref, mode), _vector(f_view, mode)
    if a.shape != b.shape:
        raise ValueError(f"Feature shapes differ: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        return 1.0
    cos = float(np.dot(a, b) / (na * nb))
    return float(np.clip(1.0 - cos, 0.0, 2.0))
