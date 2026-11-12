"""Universal adversarial patch in the style of DPatch (Section 17.2).

One square patch is optimised on DEVELOPMENT images with the vanishing loss, placed at a
random position inside the content region at each step. At test time it is pasted at a
fixed, seeded position per image. `random_patch` gives the occlusion control: same size and
position, random pixels, so the effect of covering part of the image can be separated
from the effect of the optimised pattern.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

from attacks.common import vanishing_loss
from src.preprocessing import LetterboxInfo


def patch_position(info: LetterboxInfo, size: int, rng: np.random.Generator) -> tuple[int, int] | None:
    """Random (top, left) so the patch lies fully inside the content region, or None if it cannot fit."""
    if info.content_h < size or info.content_w < size:
        return None
    top = info.pad_top + int(rng.integers(0, info.content_h - size + 1))
    left = info.pad_left + int(rng.integers(0, info.content_w - size + 1))
    return top, left


def fixed_position(image_id: str, info: LetterboxInfo, size: int, seed: int) -> tuple[int, int] | None:
    rng = np.random.default_rng(seed + zlib.crc32(image_id.encode()))
    return patch_position(info, size, rng)


def apply_patch(x: torch.Tensor, patch: torch.Tensor, positions: list[tuple[int, int]]) -> torch.Tensor:
    """Paste `patch` (3,s,s) into each image of x (B,3,H,W); differentiable w.r.t. the patch."""
    s = patch.shape[-1]
    out = x.clone()
    for b, (top, left) in enumerate(positions):
        out[b, :, top:top + s, left:left + s] = patch
    return out


def train_patch(net: torch.nn.Module, images: list[tuple[np.ndarray, LetterboxInfo]], size: int = 80,
                steps: int = 400, lr: float = 0.01, batch: int = 8, conf_floor: float = 0.1,
                seed: int = 0, device: str = "cpu", log_every: int = 50) -> torch.Tensor:
    images = [(lb, info) for lb, info in images if info.content_h >= size and info.content_w >= size]
    if not images:
        raise ValueError("No training image is large enough for the patch")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    patch = (0.5 + 0.1 * torch.randn(3, size, size, device=device)).clamp(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=lr)
    for step in range(1, steps + 1):
        idx = rng.choice(len(images), size=min(batch, len(images)), replace=False)
        xs = torch.stack([torch.from_numpy(images[i][0]).permute(2, 0, 1) for i in idx]).to(device).float() / 255
        pos = [patch_position(images[i][1], size, rng) for i in idx]
        with torch.enable_grad():
            loss = vanishing_loss(net(apply_patch(xs, patch.clamp(0, 1), pos)), conf_floor)
            opt.zero_grad()
            loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
        if step % log_every == 0 or step == 1:
            print(f"  patch step {step}/{steps}: loss {loss.item():.3f}")
    return patch.detach()


def random_patch(size: int, seed: int, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, size, size, generator=g).to(device)
