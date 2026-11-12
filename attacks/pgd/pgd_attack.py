"""White-box L-infinity PGD "vanishing" attack (Section 17.1), restricted to the content region.

Also used for the black-box transfer study: the same function run on a SURROGATE
detector; the target YOLOv8n is never queried for gradients.
"""
from __future__ import annotations

import torch

from attacks.common import vanishing_loss


def pgd_attack(net: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, eps: float, steps: int = 10,
               alpha: float | None = None, conf_floor: float = 0.1, seed: int = 0) -> torch.Tensor:
    """x: (1,3,H,W) in [0,1]; eps and alpha in [0,1] units (8/255 etc.). Returns x_adv in [0,1]."""
    alpha = eps / 4 if alpha is None else alpha
    g = torch.Generator().manual_seed(seed)
    delta = ((torch.rand(x.shape, generator=g) * 2 - 1) * eps).to(x.device) * mask  # random start
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = vanishing_loss(net(x + delta), conf_floor)
            (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = delta - alpha * grad.sign() * mask      # descend: fewer / weaker detections
            delta = delta.clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()
