# Member B guide (MacBook Air M2)

You own the attacks: the development set A needs for choosing settings, the five test attack sets, their attack success, the mAP numbers and the example figure. All of it runs through one command, `scripts/member_b.py`. Every step is safe to re-run: finished sets are skipped.

Open the repo in VS Code, open a terminal, and make sure it shows `(.venv)`. If it doesn't, run `source .venv/bin/activate`.

Attacks use the Mac GPU (`mps`) automatically. **You never screen images**: A screens every set on one CPU, so the paper's numbers come from one machine.

## Thursday 12 Nov

| Step | Command | Takes | Done when |
|---|---|---|---|
| 1. Get the dataset | Download A's `team_pack.zip` from Drive, put it in the repo folder, run `unzip team_pack.zip` | 5 min | `dataset/clean/dev` and `dataset/clean/test` exist. |
| 2. Setup check | `python scripts/member_b.py setup` | 1–2 min | All PASS; it also downloads `yolov5nu.pt`, needed for the transfer attack. |
| 3. Dev attacks for A | `python scripts/member_b.py dev` | 5–15 min | It prints the attack success rate for 50 images. |
| 4. Send them to A | `python scripts/member_b.py pack pgd_dev_eps8`, then upload `dataset/upload/pgd_dev_eps8.zip` to Drive | 2 min | A has it; this is A's first job tomorrow. |
| 5. Start the test attacks | `python scripts/member_b.py attacks` | 1–3 h in total | Leave it running tonight, plugged in, lid open. Prevent sleep with `caffeinate -i python scripts/member_b.py attacks`. |

`attacks` runs the five sets in order of importance: PGD ε=8, transfer, patch, PGD ε=4, random patch. The first set picks the 200 test images and saves them in `splits/attack_ids.csv`; every later set reuses exactly those images. If it stops, run the same command again and it carries on from the next unfinished set.

## Friday 13 Nov

| Step | Command | Done when |
|---|---|---|
| 6. Check | `python scripts/member_b.py check` | No FAIL lines. Each set shows its attack success rate. |
| 7. Send sets to A | `python scripts/member_b.py pack --all`, then upload `dataset/upload/*.zip` to Drive and tell A which sets are new | A runs `member_a.py screen`. |
| 8. Commit | `git checkout -b b/attacks` · `git add splits/attack_ids.csv results/` · commit · push · `gh pr create --fill` | The PR is merged; C can pull `attack_success.csv`. |

**If the Mac is too slow** (more than 2 hours for a single set): run that set in Google Colab.
1. Choose Runtime, then Change runtime type, then T4 GPU.
2. Upload a zip of the repo plus `team_pack.zip`.
3. Run `pip install ultralytics`, then `python scripts/member_b.py attacks --only patch --device cuda`.
4. Download `dataset/adversarial/patch/` and `results/patch/`.

## Saturday 14 Nov

| Step | Command | Done when |
|---|---|---|
| 9. mAP | `pip install pycocotools`, then `python scripts/member_b.py map` | `results/tables/detection_map.csv` is committed. |
| 10. Example figure | `git pull` (to get A's scores), then `python scripts/member_b.py qualitative` | `results/figures/qualitative_<id>.pdf` shows a clean image, its PGD version and its patch version, each with detections and screening score. To choose a different image, add `--id <image_id>`. |
| 11. Write | Threat model and attacks section, using `attacks/README.md` and `check`'s attack success rates | Section drafted in Overleaf. |

## What `check` protects you from

- **Different images per attack.** Every set must use exactly the 200 images in `splits/attack_ids.csv`. Otherwise the comparisons between attacks are unfair.
- **Budget violations.** No PGD or transfer image may change any pixel by more than its ε (4 or 8 out of 255).
- **Development images leaking into the test.** The development set must come from the development split only.
- **A silent failure.** If white-box PGD ε=8 succeeds on fewer than 30% of images, something is wrong. Tell the team before going further.

## If something goes wrong

- **An error mentioning MPS:** run `export PYTORCH_ENABLE_MPS_FALLBACK=1`, then the same command again. Or add `--device cpu` (slower).
- **Any other error:** copy all of the red text into Claude in this Project, together with the command you ran.
