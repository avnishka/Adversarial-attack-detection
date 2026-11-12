import cv2
import numpy as np
import pytest

from src.preprocessing import letterbox
from src.transformations import VIEW_NAMES, gaussian_blur, jpeg_compress, make_views


def scene(h=375, w=1242):
    """Synthetic image with texture and edges, so JPEG and blur visibly change it."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (400, 300), (255, 0, 0), -1)
    return img


@pytest.fixture
def lb_info():
    return letterbox(scene(), 640)


def test_three_views_shapes(lb_info):
    lb, info = lb_info
    views = make_views(lb, info)
    assert tuple(views) == VIEW_NAMES
    for v in views.values():
        assert v.shape == lb.shape and v.dtype == np.uint8


def test_original_unchanged_and_input_not_mutated(lb_info):
    lb, info = lb_info
    before = lb.copy()
    views = make_views(lb, info)
    assert np.array_equal(views["original"], before)
    assert np.array_equal(lb, before)


def test_padding_untouched(lb_info):
    lb, info = lb_info
    rows, cols = info.content_slice
    mask = np.ones(lb.shape[:2], bool)
    mask[rows, cols] = False
    for name, v in make_views(lb, info).items():
        assert np.array_equal(v[mask], lb[mask]), f"{name} modified the padding"


def test_content_changes(lb_info):
    lb, info = lb_info
    rows, cols = info.content_slice
    views = make_views(lb, info)
    for name in ("jpeg", "blur"):
        diff = np.abs(views[name][rows, cols].astype(int) - lb[rows, cols].astype(int)).mean()
        assert diff > 1.0, f"{name} barely changed the content ({diff:.2f})"


def test_jpeg_quality_ordering():
    img = scene(200, 200)
    err = lambda q: np.abs(jpeg_compress(img, q).astype(int) - img.astype(int)).mean()
    assert err(30) > err(95)


def test_blur_sigma_ordering():
    img = scene(200, 200)
    err = lambda s: np.abs(gaussian_blur(img, s).astype(int) - img.astype(int)).mean()
    assert err(2.0) > err(0.5)


def test_invalid_params():
    img = scene(64, 64)
    with pytest.raises(ValueError):
        jpeg_compress(img, 0)
    with pytest.raises(ValueError):
        gaussian_blur(img, 0)
