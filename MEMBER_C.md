# Member C guide (Vivobook)

You turn A's and B's results into the paper: the tables, figures, LaTeX, key numbers and the written Results. You need no model and no dataset, only the CSV files they commit. All of it runs through one command, `scripts/member_c.py`, and every command only reads files, so you can run it as often as you like.

Before every run: `git checkout main` and `git pull`.

## The one command you will use most

```
python scripts/member_c.py all
```

It checks what has arrived, rebuilds every table and figure, writes the LaTeX tables, and prints the key numbers. Run it whenever A or B says they pushed something.

| Output | Where | Use it for |
|---|---|---|
| Status report | the terminal | What is missing (WAIT = someone else's turn; TODO = yours) and whether anything is inconsistent (FAIL) |
| CSV tables | `results/tables/*.csv` | The source of truth for every number |
| Figures | `results/figures/roc_full.pdf`, `score_hist.pdf`, `ablation_auroc.pdf` | Upload to Overleaf as they are; they are sized for IEEE columns |
| LaTeX tables | `results/tables/paper_tables.tex` | Paste into Overleaf. The preamble needs `\usepackage{booktabs,adjustbox,amssymb}` |
| Key numbers | `results/tables/key_numbers.md` | Every number for the Results text, plus a factual check of hypotheses H1–H4 |

## Day by day

| When | What |
|---|---|
| Thu 12 Nov | Set up Overleaf with the IEEE conference template (`IEEEtran`, conference option). Write the Introduction, Related Work and Method; the README's "Design decisions" lists every choice the Method must state. |
| Fri 13 Nov | As soon as A pushes `calibration.json` and the clean-test scores, run `all`. Check the two sanity lines: **clean FPR** near 5%, and **random-patch control** AUROC near 0.5. Write Experimental Setup. |
| Sat 14 Nov | Run `all` again once every set is in. Paste the LaTeX tables, upload the figures, and write Results, Discussion and Limitations from `key_numbers.md`. |
| Sun 15 Nov | Final assembly. Check every number in the paper against `key_numbers.md` at the `paper-results` tag. |

## Reading the status report

- **WAIT** — another member's step. The report names who to ask.
- **FAIL: screened with a different config than calibration** — A must re-screen that set; do not use its numbers yet.
- **FAIL: scored images differ from B's attacked images** — A and B are using different versions of a set; ask both.
- **WARN: clean FPR much higher than 5%** — calibration and test were run differently. Tell A before writing anything.
- **WARN: random-patch control** with AUROC above 0.7 — the screen reacts to anything covering part of the image. That is a real finding: report it as a limitation, do not hide it.

## Writing honestly from the numbers

- Report AUROC with its 95% interval, and the true-positive rate at the **calibrated** thresholds. Those thresholds come from clean calibration data only, never from test data.
- The 1% false-positive point rests on about 6 calibration images. Call it indicative.
- A weak result is still a result. For example, low detection of the transfer attack shows where consistency screening stops working.
- `key_numbers.md` states facts only. The interpretation in the Discussion is yours.
