"""Forward hook for intermediate features (Section 11.4).

The hook is reset before every forward pass and validated afterwards, so stale
outputs (e.g. from warm-up passes) can never be mistaken for the current batch.
"""
from __future__ import annotations

import torch
from torch import nn


class FeatureHook:
    def __init__(self, module: nn.Module):
        self._out: torch.Tensor | None = None
        self._calls = 0
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Hooked layer returned {type(output).__name__}, expected a Tensor. "
                            "Pick a backbone/neck layer, not the Detect head.")
        self._out = output.detach()
        self._calls += 1

    def reset(self) -> None:
        self._out, self._calls = None, 0

    def pop(self, expected_batch: int) -> torch.Tensor:
        """Return the captured features for the forward pass just run, then clear them."""
        if self._calls != 1 or self._out is None:
            raise RuntimeError(f"Expected exactly 1 hooked forward call, saw {self._calls}. "
                               "Call reset() before each forward pass.")
        if self._out.shape[0] != expected_batch:
            raise RuntimeError(f"Hooked batch size {self._out.shape[0]} != expected {expected_batch}")
        out = self._out
        self.reset()
        return out

    def remove(self) -> None:
        self._handle.remove()


def describe_layers(net: nn.Module, imgsz: int = 640, device: str = "cpu") -> list[tuple[int, str, tuple]]:
    """Run one dummy forward pass and report (index, layer type, output shape) for
    every layer in a YOLO DetectionModel. Used to choose `model.feature_layer`."""
    layers = net.model
    records: dict[int, tuple] = {}
    handles = []
    for i, layer in enumerate(layers):
        def hook(m, inp, out, i=i):
            records[i] = tuple(out.shape) if isinstance(out, torch.Tensor) else (type(out).__name__,)
        handles.append(layer.register_forward_hook(hook))
    try:
        with torch.no_grad():
            net(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        for h in handles:
            h.remove()
    return [(i, type(layer).__name__, records.get(i, ())) for i, layer in enumerate(layers)]
