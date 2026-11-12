import numpy as np
import pytest

from src.box_matching import box_iou, match_detections


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    iou = box_iou(a, b)[0]
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(50 / 150)
    assert iou[2] == pytest.approx(0.0)


def test_iou_empty():
    assert box_iou(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_identical_sets_match_perfectly():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [50, 0, 60, 30]], float)
    cls = np.array([0, 2, 0])
    res = match_detections(boxes, cls, boxes[::-1], cls[::-1])
    assert len(res.matches) == 3 and not res.unmatched_a and not res.unmatched_b
    assert all(m[2] == pytest.approx(1.0) for m in res.matches)
    assert {(m[0], m[1]) for m in res.matches} == {(0, 2), (1, 1), (2, 0)}


def test_class_mismatch_is_unmatched():
    boxes = np.array([[0, 0, 10, 10]], float)
    res = match_detections(boxes, [0], boxes, [2], class_aware=True)
    assert res.matches == [] and res.unmatched_a == [0] and res.unmatched_b == [0]
    res = match_detections(boxes, [0], boxes, [2], class_aware=False)
    assert len(res.matches) == 1


def test_low_iou_is_unmatched():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[8, 8, 18, 18]], float)  # IoU ~ 0.02
    res = match_detections(a, [0], b, [0], iou_min=0.3)
    assert res.matches == [] and res.n_total == 2


def test_one_to_one_assignment():
    # Two A boxes overlap the same B box; only the better one may claim it
    a = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], float)
    b = np.array([[0, 0, 10, 10]], float)
    res = match_detections(a, [0, 0], b, [0])
    assert len(res.matches) == 1 and res.matches[0][0] == 0
    assert res.unmatched_a == [1] and res.n_total == 2


def test_empty_inputs():
    empty = np.zeros((0, 4))
    res = match_detections(empty, [], empty, [])
    assert res.n_total == 0
    res = match_detections(np.array([[0, 0, 5, 5]]), [1], empty, [])
    assert res.unmatched_a == [0] and res.n_total == 1
