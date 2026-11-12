import numpy as np
import pytest

from scripts.member_a import apply_selection, screening_targets
from src.config import DEFAULT_CONFIG, load_config
from src.run_meta import config_hash


def test_apply_selection_keeps_comments_and_changes_only_two_values():
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    new = apply_selection(text, 15, "flatten")
    import yaml
    old_cfg, new_cfg = yaml.safe_load(text), yaml.safe_load(new)
    assert new_cfg["model"]["feature_layer"] == 15 and new_cfg["features"]["mode"] == "flatten"
    new_cfg["model"]["feature_layer"] = old_cfg["model"]["feature_layer"]
    new_cfg["features"]["mode"] = old_cfg["features"]["mode"]
    assert new_cfg == old_cfg                                  # nothing else changed
    assert text.count("#") == new.count("#")                   # comments kept


def test_apply_selection_refuses_ambiguous_text():
    with pytest.raises(ValueError):
        apply_selection("model:\n  device: auto\n", 9, "gap")


def test_config_hash_ignores_device_only():
    cfg = load_config()
    h = config_hash(cfg)
    cfg["model"]["device"] = "mps"
    assert config_hash(cfg) == h
    cfg["model"]["feature_layer"] = 21
    assert config_hash(cfg) != h


def test_screening_targets_skip_dev_and_empty(tmp_path):
    import cv2
    adv = tmp_path / "adv"
    for name, n in [("pgd_eps8", 2), ("pgd_dev_eps8", 2), ("empty", 0)]:
        (adv / name).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(adv / name / f"{i:012d}.png"), np.zeros((8, 8, 3), np.uint8))
    names = [n for n, _ in screening_targets(adv, tmp_path / "clean_test")]
    assert names == ["clean_test", "pgd_eps8"]
