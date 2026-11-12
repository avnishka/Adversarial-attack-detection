from types import SimpleNamespace

import numpy as np
import pytest

from src.box_consistency import box_difference
from src.box_matching import match_detections
from src.confidence_consistency import confidence_difference
from src.feature_consistency import feature_distance
from src.score_aggregation import aggregate, view_differences


def dets(boxes, scores, classes):
    return SimpleNamespace(boxes=np.array(boxes, float).reshape(-1, 4), scores=np.array(scores, float),
                           classes=np.array(classes, int))


REF = dets([[0, 0, 10, 10], [20, 20, 40, 40]], [0.9, 0.8], [2, 0])


def test_identical_views_give_zero():
    m = match_detections(REF.boxes, REF.classes, REF.boxes, REF.classes)
    assert box_difference(m) == 0.0
    assert confidence_difference(m, REF.scores, REF.scores) == 0.0


def test_missing_box_known_value():
    view = dets([[0, 0, 10, 10]], [0.7], [2])        # second object disappeared, first lost 0.2 conf
    m = match_detections(REF.boxes, REF.classes, view.boxes, view.classes)
    # one match (IoU 1), one unmatched -> n = 2
    assert box_difference(m) == pytest.approx((0 + 1) / 2)
    assert confidence_difference(m, REF.scores, view.scores) == pytest.approx((0.2 + 1) / 2)


def test_empty_views_give_zero():
    empty = dets([], [], [])
    m = match_detections(empty.boxes, empty.classes, empty.boxes, empty.classes)
    assert box_difference(m) == 0.0 and confidence_difference(m, [], []) == 0.0


def test_everything_disappears_gives_one():
    empty = dets([], [], [])
    m = match_detections(REF.boxes, REF.classes, empty.boxes, empty.classes)
    assert box_difference(m) == 1.0 and confidence_difference(m, REF.scores, []) == 1.0


def test_feature_distance():
    rng = np.random.default_rng(0)
    f = rng.random((8, 4, 4))
    for mode in ("gap", "flatten"):
        assert feature_distance(f, f, mode) == pytest.approx(0.0, abs=1e-12)
        assert feature_distance(f, 3 * f, mode) == pytest.approx(0.0, abs=1e-12)   # scale invariant
        assert feature_distance(f, -f, mode) == pytest.approx(2.0)
    assert feature_distance(np.zeros((2, 2, 2)), np.zeros((2, 2, 2))) == 0.0
    with pytest.raises(ValueError):
        feature_distance(f, f, "bad")


def test_view_differences_and_max_aggregation():
    jpeg = REF
    blur = dets([[0, 0, 10, 10]], [0.9], [2])
    feats = np.stack([np.ones((4, 2, 2)), np.ones((4, 2, 2)), -np.ones((4, 2, 2))])
    row = view_differences([REF, jpeg, blur], feats, feat_mode="gap")
    assert row["D_box_jpeg"] == 0.0 and row["D_box_blur"] == pytest.approx(0.5)
    assert row["D_feat_jpeg"] == pytest.approx(0.0) and row["D_feat_blur"] == pytest.approx(2.0)
    assert {"D_feat_gap_jpeg", "D_feat_flatten_blur"} <= set(row)
    agg = aggregate(row)
    assert agg == pytest.approx({"box": 0.5, "conf": 0.5, "feat": 2.0})
    assert aggregate(row, views=("jpeg",))["box"] == 0.0


def test_wrong_number_of_views():
    with pytest.raises(ValueError):
        view_differences([REF, REF])
