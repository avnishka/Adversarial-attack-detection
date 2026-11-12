# Member A guide (Zenbook, Windows)

You own the dataset, the settings, calibration, every screening run, and runtime. All of it runs through one command, `scripts/member_a.py`. Every step is safe to re-run: finished work is skipped.

Open the repo in VS Code, open a terminal (Ctrl+`), and make sure it shows `(.venv)`. If it doesn't, run `.venv\Scripts\Activate.ps1`.

## Thursday 12 Nov

| Step | Command | Takes | Done when |
|---|---|---|---|
| 1. Download COCO | `python scripts/member_a.py download` | 10–30 min | It prints "Done: 5000 images". If the download stops, run it again: it resumes. |
| 2. Build splits | `python scripts/member_a.py prepare` | 1 min | Every line says PASS, with roughly 400 / 600 / 1,000 images. |
| 3. Commit splits | `git checkout -b a/splits` · `git add splits/` · `git commit -m "Add dataset splits"` · `git push -u origin a/splits` · `gh pr create --fill` | 2 min | The PR is merged. |
| 4. Pack for Drive | `python scripts/member_a.py pack` | 1 min | `dataset/team_pack.zip` is uploaded to Drive and B and C are told. |

## Friday 13 Nov

| Step | Command | Takes | Done when |
|---|---|---|---|
| 5. Get B's dev attacks | Download `pgd_dev_eps8.zip` from Drive into the repo folder and unzip it there (in Explorer: right-click, then Extract All, then choose the repo folder) | 2 min | `dataset/adversarial/pgd_dev_eps8/` holds 50 PNGs. |
| 6. Choose settings | `python scripts/member_a.py select --apply` | 15–30 min | It prints the best layer and mode, and says `configs/default.yaml` was updated. |
| 7. Commit and freeze | The git commands it prints, ending with `git tag frozen-config` and `git push --tags` | 5 min | `check` shows "config frozen: PASS". |
| 8. Calibrate | `python scripts/member_a.py calibrate` | 5–10 min | `results/calibration/calibration.json` is committed. |
| 9. Screen | Unzip each new `<set>.zip` from B in the repo folder, then `python scripts/member_a.py screen` | about 5 min per 1,000 images | Every set in `dataset/adversarial` has a `results/<set>/scores.csv`. Re-run whenever B uploads a new set. |
| 10. Commit results | `git add results/` · commit · push · PR | 2 min | C can `git pull` the CSVs. |

## Saturday 14 Nov

| Step | Command | Takes | Done when |
|---|---|---|---|
| 11. Screen the rest | `python scripts/member_a.py screen` | — | Says "done" for every set. |
| 12. Runtime | `python scripts/member_a.py runtime` (plugged in, other apps closed) | 5 min | `results/tables/runtime.csv` is committed, with the laptop model and RAM added to `runtime_env.csv`. |
| 13. Final check | `python scripts/member_a.py check` | seconds | No FAIL lines. Then at 21:00: `git tag paper-results` and `git push --tags`. |

## What `check` protects you from

- **Mixed settings.** Each `scores.csv` has a `screening_meta.json` beside it holding a fingerprint of the config. A set screened before the config was frozen shows as **FAIL**. Fix: `python scripts/member_a.py screen --only <set> --force`.
- **Mixed machines.** Every number in the paper must come from this laptop's CPU. A set screened on another device shows as **WARN** or **FAIL**.
- **Broken calibration.** The clean test images should be flagged at about the target rate (about 5% at the 5% threshold). Much higher means calibration and test differ somehow; tell the team before using the numbers.

## If something goes wrong

- **Any red error text:** copy all of it into Claude in this Project, together with the command you ran.
- **`select` is very slow:** add `--layers 9 15` to test two layers instead of three.
- **You need to change the config after freezing:** stop and tell the team. Every calibration and screening run must then be redone, and `check` will list exactly which ones.
