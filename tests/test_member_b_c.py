import zipfile

import numpy as np
import pandas as pd

from scripts.member_b import choose_example, pack_set
from scripts.member_c import bold_best, latex_by_method, latex_main, table


def test_pack_set_keeps_repo_paths(tmp_path):
    import cv2
    adv = tmp_path / "dataset" / "adversarial"
    (adv / "pgd_eps8").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(adv / "pgd_eps8" / f"{i:012d}.png"), np.zeros((4, 4, 3), np.uint8))
    out = pack_set("pgd_eps8", adv, tmp_path / "upload", tmp_path)
    names = zipfile.ZipFile(out).namelist()
    assert len(names) == 3 and all(n.startswith("dataset/adversarial/pgd_eps8/") for n in names)


def test_choose_example_prefers_both_successful_and_few_objects():
    ids = ["a", "b", "c"]
    pgd = pd.DataFrame({"image_id": ids, "success": [True, True, True], "n_clean_det": [12, 3, 4]})
    patch = pd.DataFrame({"image_id": ids, "success": [True, False, True]})
    assert choose_example({"pgd_eps8": pgd, "patch": patch}, ids) == "c"
    assert choose_example({}, ids) == "a"


def test_bold_best_ties_as_printed():
    out = bold_best([0.9996, 1.0, 0.95])
    assert out[0].startswith("\\textbf") and out[1].startswith("\\textbf") and not out[2].startswith("\\textbf")


def test_latex_tables_are_balanced():
    main = pd.DataFrame([{"set": "pgd_eps8", "subset": "all", "n_adv": 200, "auroc": 0.95, "auroc_lo": 0.9,
                          "auroc_hi": 0.97, "tpr@5": 0.8, "tpr@5_lo": 0.7, "tpr@5_hi": 0.85, "tpr@1": 0.5}])
    tex = latex_main(main, None)
    for a, b in [("\\begin{table*}", "\\end{table*}"), ("\\begin{tabular}", "\\end{tabular}"),
                 ("\\begin{adjustbox}", "\\end{adjustbox}")]:
        assert tex.count(a) == tex.count(b) == 1
    assert "0.950 [0.900, 0.970]" in tex and "80.0\\%" in tex
    df = pd.DataFrame([{"set": s, "method": m, "auroc": v} for s in ("pgd_eps8", "patch_random")
                       for m, v in (("full", 0.9), ("no_transform", 0.6))])
    tex = latex_by_method(df, ["no_transform", "full"], str, "cap", "tab:x")
    assert tex.count("\\textbf") == 1          # bolded in the attack column only, never in the control column
    assert table("c", "l", "ll", "a & b", ["1 & 2"]).count("\\\\") >= 2
