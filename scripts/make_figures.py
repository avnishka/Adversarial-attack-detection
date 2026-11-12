"""Paper figures (IEEE single-column width, 3.5 in), saved as PDF (for LaTeX) and PNG (300 dpi).

Usage:  python scripts/make_figures.py
Reads the same files as run_evaluation.py. Writes results/figures/:
  roc_full.pdf/png        ROC curve of the proposed score, one line per attack set
  score_hist.pdf/png      clean vs adversarial score distributions with the calibrated thresholds
  ablation_auroc.pdf/png  AUROC of ablation experiments A-G per attack set
Each set keeps the same colour AND line style in every figure (readable in grayscale print).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_evaluation import DEFAULT_SETS, load_sets  # noqa: E402
from src.calibration import ABLATION_METHODS, load_calibration, method_scores  # noqa: E402
from src.config import resolve_path  # noqa: E402
from src.evaluation import auroc, roc_points  # noqa: E402
from src.io_utils import read_scores  # noqa: E402

# Fixed order: a set always gets the same colour and dash, whichever sets are present (validated palette)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
LABELS = {"pgd_eps4": "PGD ε=4/255", "pgd_eps8": "PGD ε=8/255", "transfer_eps8": "Transfer ε=8/255",
          "patch": "Adversarial patch", "patch_random": "Random patch (control)"}
INK, MUTED, CLEAN_GRAY = "#2b2b29", "#73726c", "#9a9991"


def style(i: int) -> dict:
    return {"color": PALETTE[i % len(PALETTE)], "linestyle": DASHES[i % len(DASHES)]}


def setup() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
        "legend.frameon": False, "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {out / name}.pdf/.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--clean", default="results/clean_test/scores.csv")
    ap.add_argument("--calibration", default="results/calibration/calibration.json")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    setup()
    cal = load_calibration(resolve_path(args.calibration))
    clean = read_scores(resolve_path(args.clean))
    sets = load_sets(args.sets)
    order = [s for s in DEFAULT_SETS + args.sets if s in sets]
    order = list(dict.fromkeys(order))
    idx = {s: DEFAULT_SETS.index(s) if s in DEFAULT_SETS else len(DEFAULT_SETS) + k for k, s in enumerate(order)}
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. ROC curves
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    ax.plot([0, 1], [0, 1], color=CLEAN_GRAY, linewidth=1, linestyle=":", label="Chance")
    for s in order:
        fpr, tpr = roc_points(clean, sets[s][0], cal, "full")
        a = auroc(method_scores(clean, "full", cal["stats"]), method_scores(sets[s][0], "full", cal["stats"]))
        ax.plot(fpr, tpr, linewidth=2, label=f"{LABELS.get(s, s)} ({a:.2f})", **style(idx[s]))
    ax.set_xlabel("False positive rate (clean images flagged)")
    ax.set_ylabel("True positive rate (attacks flagged)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(title="Attack set (AUROC)", loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2,
              title_fontsize=7, columnspacing=1.0, handlelength=2.4)
    save(fig, out, "roc_full")

    # 2. Score distributions: one small panel per set, clean in gray, thresholds marked
    if order:
        n = len(order)
        fig, axes = plt.subplots(n, 1, figsize=(3.5, 1.15 * n + 0.4), sharex=True)
        axes = np.atleast_1d(axes)
        c = method_scores(clean, "full", cal["stats"])
        allv = np.r_[c, *[method_scores(sets[s][0], "full", cal["stats"]) for s in order]]
        lo, hi = np.percentile(allv, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        for ax, s in zip(axes, order):
            a = method_scores(sets[s][0], "full", cal["stats"])
            ax.hist(np.clip(c, lo, hi), bins=bins, density=True, color=CLEAN_GRAY, alpha=0.55,
                    edgecolor="white", linewidth=0.5)
            ax.hist(np.clip(a, lo, hi), bins=bins, density=True, color=style(idx[s])["color"], alpha=0.75,
                    edgecolor="white", linewidth=0.5)
            for f, ls in zip(cal["fprs"], ["--", ":"]):
                ax.axvline(cal["thresholds"]["full"][str(f)], color=INK, linestyle=ls, linewidth=1)
            ax.set_title(LABELS.get(s, s), loc="left", fontsize=7, color=INK, pad=2)
            ax.set_yticks([])
            ax.grid(False)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Patch(color=CLEAN_GRAY, alpha=0.55, label="Clean test images"),
                   Patch(color=MUTED, label="Adversarial images (colour per panel)")]
        handles += [Line2D([], [], color=INK, linestyle=ls, linewidth=1, label=f"Threshold at {f:.0%} FPR")
                    for f, ls in zip(cal["fprs"], ["--", ":"])]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
        fig.tight_layout(h_pad=0.6)
        axes[-1].set_xlabel("Screening score S (standardised; higher = more suspicious)")
        save(fig, out, "score_hist")

    # 3. Ablation: AUROC per experiment, grouped by attack set
    rows = []
    for s in order:
        cs = {m: method_scores(clean, m, cal["stats"]) for m in ABLATION_METHODS}
        for m in ABLATION_METHODS:
            rows.append({"set": s, "method": m, "auroc": auroc(cs[m], method_scores(sets[s][0], m, cal["stats"]))})
    if rows:
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7.16, 2.4))   # IEEE double-column width
        width = 0.8 / len(order)
        x = np.arange(len(ABLATION_METHODS))
        for k, s in enumerate(order):
            vals = df[df["set"] == s].set_index("method").loc[ABLATION_METHODS, "auroc"].to_numpy()
            ax.bar(x - 0.4 + width * (k + 0.5), vals, width=width * 0.9, color=style(idx[s])["color"],
                   label=LABELS.get(s, s), edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color=CLEAN_GRAY, linestyle=":", linewidth=1)
        ax.set_xticks(x, [m.replace("_", "\n", 1) for m in ABLATION_METHODS])
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=min(len(order), 5), loc="upper center", bbox_to_anchor=(0.5, 1.18))
        save(fig, out, "ablation_auroc")


if __name__ == "__main__":
    main()
