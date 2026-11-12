"""End-to-end check through the real model. Skipped if torch/ultralytics or weights are missing."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from src.config import load_config, resolve_path  # noqa: E402
from src.preprocessing import letterbox, load_rgb, stack_views  # noqa: E402
from src.transformations import VIEW_NAMES, make_views  # noqa: E402

cfg = load_config()
if not resolve_path(cfg["model"]["weights"]).exists():
    pytest.skip("YOLOv8n weights not downloaded", allow_module_level=True)


@pytest.fixture(scope="module")
def detector():
    from src.detector import YOLOv8nDetector
    return YOLOv8nDetector.from_config(cfg)


@pytest.fixture(scope="module")
def batch():
    from ultralytics.utils import ASSETS
    lb, info = letterbox(load_rgb(ASSETS / "bus.jpg"), 640)
    views = make_views(lb, info)
    return stack_views([views[n] for n in VIEW_NAMES])


def test_three_view_batch(detector, batch):
    dets, feats = detector(batch)
    assert len(dets) == 3
    assert all(len(d) > 0 for d in dets)
    assert feats is not None and feats.shape[0] == 3


def test_deterministic(detector, batch):
    d1, f1 = detector(batch)
    d2, f2 = detector(batch)
    assert np.allclose(d1[0].boxes, d2[0].boxes, atol=1e-4)
    assert torch.allclose(f1, f2, atol=1e-5)


def test_batch_matches_single(detector, batch):
    """Batching must not change a view's result compared with running it alone."""
    d_batch, f_batch = detector(batch)
    d_single, f_single = detector(batch[:1])
    assert len(d_batch[0]) == len(d_single[0])
    assert np.allclose(d_batch[0].boxes, d_single[0].boxes, atol=1e-2)
    assert torch.allclose(f_batch[:1], f_single, atol=1e-3)
