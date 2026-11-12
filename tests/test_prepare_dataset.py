from scripts.prepare_dataset import build_splits

FRACS = {"dev": 0.2, "calibration": 0.3, "test": 0.5}


def fake_coco(n_road=50, n_other=20):
    images, anns = [], []
    for i in range(1, n_road + n_other + 1):
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": 640, "height": 480})
        anns.append({"image_id": i, "category_id": 3 if i <= n_road else 1})   # car vs person-only
    return {"images": images, "annotations": anns}


def test_only_road_images_and_disjoint_splits():
    splits = build_splits(fake_coco(), {2, 3, 4, 6, 8, 10, 13}, max_images=2000, fractions=FRACS, seed=0)
    ids = [i for df in splits.values() for i in df["image_id"]]
    assert len(ids) == 50 and len(set(ids)) == 50                  # person-only images excluded, no duplicates
    assert [len(splits[s]) for s in ("dev", "calibration", "test")] == [10, 15, 25]
    assert all(len(i) == 12 for i in ids)                          # ids keep their leading zeros


def test_cap_and_determinism():
    a = build_splits(fake_coco(), {3}, max_images=20, fractions=FRACS, seed=0)
    b = build_splits(fake_coco(), {3}, max_images=20, fractions=FRACS, seed=0)
    assert sum(len(df) for df in a.values()) == 20
    assert all(a[s]["image_id"].tolist() == b[s]["image_id"].tolist() for s in a)
