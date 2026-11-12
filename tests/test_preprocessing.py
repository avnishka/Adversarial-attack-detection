import numpy as np
import pytest

from src.preprocessing import boxes_to_letterbox, boxes_to_original, letterbox, stack_views


def make_img(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("h,w", [(480, 640), (375, 1242), (720, 1280), (640, 480), (640, 640)])
def test_letterbox_shape_and_aspect(h, w):
    img = make_img(h, w)
    lb, info = letterbox(img, 640)
    assert lb.shape == (640, 640, 3) and lb.dtype == np.uint8
    assert max(info.content_w, info.content_h) == 640
    assert abs(info.content_w / info.content_h - w / h) < 0.01  # aspect ratio preserved


def test_padding_value_and_content_location():
    img = make_img(375, 1242)  # KITTI-like, wide image -> padding top and bottom
    lb, info = letterbox(img, 640, pad_value=114)
    rows, cols = info.content_slice
    mask = np.ones(lb.shape[:2], bool)
    mask[rows, cols] = False
    assert mask.any() and (lb[mask] == 114).all()


def test_input_not_modified():
    img = make_img(480, 640)
    before = img.copy()
    letterbox(img, 640)
    assert np.array_equal(img, before)


def test_box_round_trip():
    img = make_img(375, 1242)
    _, info = letterbox(img, 640)
    boxes = np.array([[10, 20, 300, 200], [600, 100, 1240, 370]], np.float32)
    back = boxes_to_original(boxes_to_letterbox(boxes, info), info)
    assert np.allclose(back, boxes, atol=1e-3)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        letterbox(make_img(10, 10), size=630)


def test_stack_views():
    a = make_img(640, 640)
    assert stack_views([a, a, a]).shape == (3, 640, 640, 3)
    with pytest.raises(ValueError):
        stack_views([a, make_img(320, 320)])
