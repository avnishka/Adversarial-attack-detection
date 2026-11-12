# Consistency-Based Screening of Adversarial Inputs to YOLOv8n

A frozen YOLOv8n detector looks at three views of each road-scene image (original, JPEG, Gaussian blur) in a single batched forward pass. Box, confidence and feature consistency are combined into a screening score, and the image is flagged as CLEAN or POTENTIALLY ADVERSARIAL.

**Team guides:** start with `COMMON.md`, then `MEMBER_A.md`, `MEMBER_B.md` or `MEMBER_C.md`.

## Status

All phases have code. The model-free parts (scores, calibration, evaluation, figures, dataset splits) are unit-tested and were run end to end on synthetic score files. The model-based scripts (screening, attacks, mAP, runtime) are first run on the team laptops.

| Phase | Code |
|---|---|
| One-command workflow per member | `scripts/member_a.py`, `member_b.py`, `member_c.py` (guides: `MEMBER_A.md`, `MEMBER_B.md`, `MEMBER_C.md`) |
| 1–2 Environment and baseline | `scripts/smoke_test.py`, `scripts/run_baseline.py` |
| 3 Dataset | `scripts/prepare_dataset.py` → `splits/*.csv`, `dataset/clean/<split>/` |
| 4–5 Views and batched inference | `src/transformations.py`, `src/detector.py`, `src/feature_extractor.py` |
| 6 Consistency scores | `src/box_consistency.py`, `src/confidence_consistency.py`, `src/feature_consistency.py`, `src/score_aggregation.py` |
| 7 Calibration | `src/calibration.py`, `scripts/run_calibration.py`, `scripts/select_params.py` |
| 8 Screening | `src/screening.py`, `scripts/run_screening.py` |
| 9 Attacks | `attacks/` (see `attacks/README.md`), `scripts/run_attacks.py` |
| 10 Evaluation | `src/evaluation.py`, `scripts/run_evaluation.py`, `scripts/run_detection_eval.py`, `scripts/run_runtime.py`, `scripts/make_figures.py` |

## Setup (Windows / Linux / macOS)

Use Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
# PyTorch first: Windows/Linux CPU -> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#                macOS (M-series)  -> pip install torch torchvision
pip install -r requirements.txt
pip install pycocotools                      # only needed for mAP

cd models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov5nu.pt')"
cd ..
python scripts/smoke_test.py                 # all checks should PASS
pytest
```

## Full experiment, in order

Who runs each step is in the team manual. Screening numbers for the paper all come from **one laptop with `--device cpu`**.

```bash
# 1. Dataset (COCO val2017 + annotations unzipped into dataset/raw/)
python scripts/prepare_dataset.py

# 2. 50 dev PGD images for parameter selection (attacks may use --device mps or cuda)
python scripts/run_attacks.py --attack pgd --eps 8 --source dataset/clean/dev --n 50 --name pgd_dev_eps8

# 3. Choose feature layer and mode on dev data, edit configs/default.yaml, commit, tag frozen-config
python scripts/select_params.py --adv dataset/adversarial/pgd_dev_eps8 --device cpu

# 4. Calibration (clean calibration split only) -> results/calibration/calibration.json
python scripts/run_calibration.py --device cpu

# 5. Attack sets on the same 200 test images
python scripts/run_attacks.py --attack pgd --eps 8 --save-ids splits/attack_ids.csv --name pgd_eps8
python scripts/run_attacks.py --attack pgd --eps 4 --ids splits/attack_ids.csv --name pgd_eps4
python scripts/run_attacks.py --attack transfer --eps 8 --ids splits/attack_ids.csv --name transfer_eps8
python scripts/run_attacks.py --attack patch --ids splits/attack_ids.csv --name patch
python scripts/run_attacks.py --attack patch_random --ids splits/attack_ids.csv --name patch_random

# 6. Screening (clean test + every attack set)
python scripts/run_screening.py --source dataset/clean/test --out results/clean_test/scores.csv --device cpu
python scripts/run_screening.py --source dataset/adversarial/pgd_eps8 --out results/pgd_eps8/scores.csv --device cpu
#    ...repeat for pgd_eps4, transfer_eps8, patch, patch_random

# 7. Tables and figures (no model needed)
python scripts/run_evaluation.py
python scripts/make_figures.py

# 8. Detection mAP and runtime
python scripts/run_detection_eval.py
python scripts/run_runtime.py --device cpu
```

Paper tables are written to `results/tables/`; figures (PDF for LaTeX, plus PNG) go to `results/figures/`.

## Design decisions

- **Letterboxing is done by us**, not by Ultralytics, so all views share one geometry and boxes map back exactly.
- **Transforms and attacks touch only the content region**, never the grey padding.
- **One forward pass per batch**: the PyTorch model is called directly, followed by NMS. The feature hook is reset before each pass and checked after it (exactly one call, batch size 3).
- **Scores** (Section 12): Hungarian matching on 1 − IoU, class-aware, IoU ≥ 0.3. U counts unmatched boxes in both views. If neither view has detections, D = 0.
- **Aggregation before standardisation** (Section 13): max over JPEG and blur, then Z = (D − μ) / max(σ, 1e-6) with calibration statistics, then S = mean of Z.
- **Thresholds** are the 95th and 99th percentiles of clean calibration scores. They are never computed on test data. With 600 calibration images, the 1% point rests on about 6 images, so it is reported as indicative (`fpr_notes` in `calibration.json`).
- **All per-view values are saved**, so ablations A–G, the JPEG-only and blur-only baselines, and the no-transform baseline are recomputed from CSV files without new inference.
- **Attacks** are saved as PNG crops of the content region, so loading them reproduces the exact attacked input. The attack success rule is in `attacks/README.md`.

## Known limitations to state in the paper

- Only COCO road-class images. KITTI and BDD100K are future work.
- No adaptive attacker (one who optimises through the JPEG and blur views).
- A high score is not proof of an attack: naturally difficult images can score high too.
