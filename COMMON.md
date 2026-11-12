# Common guide (all three members)

Release: **final, 12 Nov 2026**. This folder structure and code are the team's starting point. From here on, every change goes through GitHub pull requests; nobody replaces the whole folder again (see "Changing code safely").

Read this file first. Then follow your own guide: `MEMBER_A.md` (Zenbook), `MEMBER_B.md` (MacBook), `MEMBER_C.md` (Vivobook).

## 1. Timeline

| Day | A (Zenbook) | B (MacBook) | C (Vivobook) |
|---|---|---|---|
| Thu 12 Nov | Repo, COCO download, splits, Drive pack | Setup, 50 dev PGD images, start the test attacks overnight | Setup, IEEE template, Introduction, Related Work, Method |
| Fri 13 Nov | Choose settings, freeze config, calibrate, screen | Finish the 5 attack sets, pack them, check | First tables and figures, sanity checks, Experimental Setup |
| Sat 14 Nov | Screen remaining sets, runtime, final check | mAP, example figure, Attacks section | All tables, figures, Results, Discussion, Limitations |
| **Sat 14 Nov, 21:00** | **Results frozen: A tags `paper-results`** | | |
| Sun 15 Nov | Format checks, submit | Proofread, README reproducibility | Final assembly, number check |
| 16–20 Nov | Slides | Demo | Report |

If time runs short, the minimum result set for the paper is: clean test, PGD ε=8/255, transfer, patch, ablation, runtime. Drop PGD ε=4/255 and the random-patch control before anything else.

## 2. Install the tools (first hour, everyone)

Use **Python 3.11** even if a newer version exists.

**Windows (A and C)**
1. Python 3.11 from python.org. On the first installer screen, tick **"Add python.exe to PATH"**.
2. Git for Windows from git-scm.com (keep all defaults).
3. VS Code from code.visualstudio.com, then install the **Python** extension.
4. In PowerShell: `winget install --id GitHub.cli`, then close and reopen PowerShell.
5. In PowerShell: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then answer `Y`.

**Mac (B)**
1. `xcode-select --install`
2. Install Homebrew (the one-line command on brew.sh), then run the two lines it prints at the end.
3. `brew install python@3.11 gh`
4. VS Code, plus the **Python** extension.

**Everyone**
```bash
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
gh auth login        # GitHub.com -> HTTPS -> Yes -> Login with a web browser
```

## 3. Create and join the repo

**A (once):** unzip `RealTime_Adversarial_Detection.zip` into a plain folder (not OneDrive), open it in VS Code, open a terminal, and run:
```bash
git init
git add .
git commit -m "Final starter release (12 Nov)"
git branch -M main
gh repo create realtime-adversarial-detection --private --source=. --remote=origin --push
```
Then on github.com: **Settings**, then **Collaborators**, then **Add people**: add B and C. Post the repo link in the group chat.

**B and C:** accept the invitation (email, or the bell icon on github.com), then:
```bash
gh repo clone <A-github-username>/realtime-adversarial-detection
cd realtime-adversarial-detection
code .
```
**B and C never unzip the release zip.** Your copy comes only from GitHub.

## 4. Python environment (every laptop, inside the repo folder)

**Windows**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Mac**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

**Everyone**
```bash
cd models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov5nu.pt')"
cd ..
python scripts/smoke_test.py      # must end with ALL CHECKS PASSED
pytest                            # must show 0 failed (about 50 passed)
```
In VS Code, press Ctrl+Shift+P (Cmd+Shift+P on Mac), run **Python: Select Interpreter** and choose `.venv`. Every terminal must show `(.venv)` at the start of the prompt.

**Done when** all three laptops show `ALL CHECKS PASSED` and `pytest` reports 0 failed. Post a screenshot in the group chat.

**A only:** after that, run `pip freeze > requirements-lock.txt` and commit it.

## 5. Daily Git routine

```bash
git checkout main
git pull                                   # always start from the latest main
git checkout -b <your-letter>/<task>       # e.g. b/attacks, c/tables
# ... work ...
git status                                 # check what changed before adding
git add <the files you changed>
git commit -m "What changed, in one line"
git push -u origin <your-letter>/<task>
gh pr create --fill --base main
```
- **Review:** post the PR link in the chat. A teammate opens **Files changed**, then **Review changes**, then **Approve**. The author clicks **Squash and merge**.
- **After any merge:** everyone runs `git checkout main` and `git pull`.
- **Standups:** 9:30, 14:00 and 21:00, 10 minutes each: what is done, what is next, what is blocking you.

## 6. Team rules

1. **Each member edits only their own files.** A: dataset, calibration and screening scripts, plus `configs/default.yaml` (before the freeze only). B: `attacks/` and attack scripts. C: evaluation and figure scripts, and the paper.
2. **Never commit** datasets, weights, `.venv` or images. `.gitignore` already blocks them; still run `git status` before every `git add`.
3. **Do commit:** code, `splits/*.csv`, `results/calibration/calibration.json`, every `results/<set>/scores.csv`, `screening_meta.json`, `attack_success.csv`, `attack_config.json`, and everything in `results/tables/` and `results/figures/`.
4. **Images move through Google Drive, never Git:** A's `team_pack.zip` and B's `<set>.zip` files. Unzip them **in the repo folder**; they already contain the right `dataset/...` paths.
5. **Never edit `configs/default.yaml` for your own laptop.** Use `--device` on the command line instead.
6. **After the `frozen-config` tag,** nobody changes the config. If it must change, tell the team: calibration and all screening must then be redone.
7. **Every number in the paper** comes from a CSV at the `paper-results` tag.

## 7. Changing code safely

The code will not be re-issued as a new zip. If something breaks:
1. Copy the full error and the command you ran into Claude in this Project.
2. The fix comes back as one changed file, or a few changed lines.
3. Apply it on a branch (`git checkout -b fix/<what>`), test it, and open a PR. The owner of that file approves it.

That way everyone's history stays intact and nobody loses work.

## 8. Where things live

| Folder | Contents | In Git? |
|---|---|---|
| `src/` | The method: views, detector, scores, calibration, screening, evaluation | yes |
| `attacks/` | PGD, transfer, patch and the threat model (`attacks/README.md`) | yes |
| `scripts/` | Everything you run, including `member_a.py`, `member_b.py`, `member_c.py` | yes |
| `configs/default.yaml` | All settings | yes |
| `splits/` | Image ID lists for dev / calibration / test and the 200 attack images | yes |
| `dataset/raw/`, `dataset/clean/<split>/` | COCO download and the road-scene subset | no (Drive) |
| `dataset/adversarial/<set>/` | Attack images, one folder per set (e.g. `pgd_eps8`) | no (Drive) |
| `results/<set>/` | `scores.csv`, `screening_meta.json`, `attack_success.csv`, `attack_config.json` | yes |
| `results/tables/`, `results/figures/` | Paper tables (CSV + LaTeX) and figures (PDF + PNG) | yes |
| `models/` | Weight files, downloaded on each laptop | no |

The empty folders `dataset/calibration`, `dataset/test`, `dataset/adversarial/{pgd,dpatch,blackbox}` and `results/{pgd,dpatch,blackbox}` come from the proposal's layout. The scripts do not use them; leave them as they are.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized (Windows) | Use `py -3.11`, or reinstall Python with "Add to PATH" ticked |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then reopen the terminal |
| `pip install torch` finds no version | Wrong Python: delete `.venv` and create it again with Python 3.11 |
| Smoke test: weights not found | Run the weights command from inside `models/` |
| `git push` rejected | `git pull`, then `git push` again |
| Merge conflict | VS Code shows Accept Current / Incoming / Both. Choose, save, `git add`, `git commit`. If unsure, ask the file's owner |
| A big file was committed | Do not push. `git rm --cached <file>`, then commit. If already pushed, tell A |
| Error mentioning MPS (Mac) | `export PYTORCH_ENABLE_MPS_FALLBACK=1`, or add `--device cpu` |
| Laptop sleeps during a long run | Windows: Settings, then System, then Power, then Never (plugged in). Mac: start the command with `caffeinate -i` |
| Out of memory | Close Chrome and other apps |

## 10. Final checklist (Sunday)

- [ ] `python scripts/member_a.py check` shows no FAIL
- [ ] `python scripts/member_b.py check` shows no FAIL
- [ ] `python scripts/member_c.py check` shows no FAIL
- [ ] Tag `paper-results` exists, and every number in the paper matches `results/tables/key_numbers.md`
- [ ] `requirements-lock.txt` is committed, and the README steps work on a fresh clone
- [ ] Paper submitted; confirmation posted in the group chat
