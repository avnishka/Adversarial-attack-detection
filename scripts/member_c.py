"""Member C's whole workflow in one command (Vivobook). No model needed: everything works on
the CSV files A and B commit, so run `git pull` first.

    python scripts/member_c.py check     # which results exist, are they consistent, what is missing
    python scripts/member_c.py tables    # all evaluation tables + paper figures
    python scripts/member_c.py latex     # IEEE-ready LaTeX tables -> results/tables/paper_tables.tex
    python scripts/member_c.py numbers   # key numbers + hypothesis check -> results/tables/key_numbers.md
    python scripts/member_c.py all       # pull-free one shot: check, tables, latex, numbers

Re-run `all` every time A or B pushes new results; it only reads files, so it is always safe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import load_calibration  # noqa: E402
from src.run_meta import read_meta  # noqa: E402
from src.team_checks import ATTACK_SETS, LABELS, PLAIN_LABELS, Row, report, run_script  # noqa: E402

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
CAL = RESULTS / "calibration" / "calibration.json"

METHOD_LABELS = {"no_transform": "No transform (1 $-$ mean conf.)", "jpeg_only": "JPEG view only",
                 "blur_only": "Blur view only", "full": "Full: box + conf. + feat. (ours)"}
ABLATION = {"A_box": (1, 0, 0), "B_conf": (0, 1, 0), "C_feat": (0, 0, 1), "D_box+conf": (1, 1, 0),
            "E_box+feat": (1, 0, 1), "F_conf+feat": (0, 1, 1), "G_box+conf+feat": (1, 1, 1)}


def read(name: str) -> pd.DataFrame | None:
    p = TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


# ----------------------------------------------------------------------------- checks
def run_checks() -> list[Row]:
    rows: list[Row] = []
    if not CAL.exists():
        return [("WAIT", "calibration", "waiting for A (results/calibration/calibration.json); run git pull")]
    cal = load_calibration(CAL)
    h = cal.get("config_hash")
    rows.append(("PASS" if h else "WARN", "calibration",
                 f"{cal['n_images']} images" + ("" if h else ", no config fingerprint (A should re-run calibrate)")))
    devices = set()
    for name in ["clean_test"] + ATTACK_SETS:
        sc = RESULTS / name / "scores.csv"
        succ = RESULTS / name / "attack_success.csv"
        if not sc.exists():
            who = "A (screening)" if name == "clean_test" or succ.exists() else "B (attack), then A (screening)"
            rows.append(("WAIT", name, f"waiting for {who}; run git pull"))
            continue
        df = pd.read_csv(sc, dtype={"image_id": str})
        status, notes = "PASS", [f"{len(df)} scored images"]
        meta = read_meta(RESULTS / name / "screening_meta.json")
        if meta:
            devices.add(meta.get("device"))
            if h and meta.get("config_hash") != h:
                status = "FAIL"; notes.append("screened with a different config than calibration (ask A to re-screen)")
        else:
            status = "WARN"; notes.append("no screening_meta.json")
        if name != "clean_test":
            if not succ.exists():
                status = "WARN" if status == "PASS" else status; notes.append("attack_success.csv missing (B)")
            else:
                s_ids = set(pd.read_csv(succ, dtype={"image_id": str})["image_id"])
                if s_ids != set(df["image_id"]):
                    status = "FAIL"; notes.append("scored images differ from B's attacked images")
        rows.append((status, name, ", ".join(notes)))
    if len(devices) > 1:
        rows.append(("FAIL", "one device", f"scores from several devices {sorted(map(str, devices))} (ask A)"))

    main = read("main_results")
    if main is None:
        rows.append(("TODO", "tables", "python scripts/member_c.py tables"))
    else:
        fpr = read("clean_fpr")
        if fpr is not None:
            f5 = float(fpr.loc[fpr["method"] == "full", "fpr@5"].iloc[0])
            rows.append(("PASS" if f5 <= 0.08 else "WARN", "clean FPR (5% target)",
                         f"{f5:.1%}" + ("" if f5 <= 0.08 else ": much higher than 5%, tell A before writing")))
        rp = main[(main["set"] == "patch_random") & (main["subset"] == "all")]
        if len(rp):
            a = float(rp["auroc"].iloc[0])
            rows.append(("PASS" if a < 0.7 else "WARN", "random-patch control",
                         f"AUROC {a:.2f}" + ("" if a < 0.7 else ": the screen reacts to occlusion; the paper must say so")))
        rows.append(("PASS" if (TABLES / "paper_tables.tex").exists() else "TODO", "LaTeX tables",
                     "results/tables/paper_tables.tex" if (TABLES / "paper_tables.tex").exists()
                     else "python scripts/member_c.py latex"))
    for f, who in (("detection_map", "B"), ("runtime", "A")):
        rows.append(("PASS" if read(f) is not None else "WAIT", f"{f} table",
                     "present" if read(f) is not None else f"waiting for {who}"))
    return rows


def cmd_check(args) -> None:
    sys.exit(report(run_checks(), "All results are in and consistent. Write away."))


def cmd_tables(args) -> None:
    if not CAL.exists() or not (RESULTS / "clean_test" / "scores.csv").exists():
        sys.exit("Need calibration.json and results/clean_test/scores.csv from A first (git pull).")
    run_script("run_evaluation.py", "--n-boot", args.n_boot)
    run_script("make_figures.py")


# ----------------------------------------------------------------------------- LaTeX
def f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}\\%"


def with_ci(x, lo, hi, as_pct=False) -> str:
    f = pct if as_pct else f3
    return f"{f(x)} [{f(lo)}, {f(hi)}]"


def table(caption: str, label: str, colspec: str, header: str, body: list[str], wide: bool = False) -> str:
    env = "table*" if wide else "table"
    return "\n".join([f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize",
                      f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      "\\begin{adjustbox}{max width=\\linewidth}",
                      f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule",
                      *[b + " \\\\" for b in body], "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
                      f"\\end{{{env}}}", ""])


def sets_in(df: pd.DataFrame) -> list[str]:
    return [s for s in ATTACK_SETS if s in set(df["set"])]


def bold_best(values: list[float]) -> list[str]:
    """Bold every value that ties for the best AS PRINTED (3 decimals), so equal-looking numbers match."""
    shown = [round(v, 3) if not np.isnan(v) else np.nan for v in values]
    best = np.nanmax(shown) if len(shown) and not np.all(np.isnan(shown)) else None
    return [f"\\textbf{{{f3(v)}}}" if best is not None and sv == best else f3(v) for v, sv in zip(values, shown)]


def latex_main(main: pd.DataFrame, fpr: pd.DataFrame | None) -> str:
    body = []
    for s in sets_in(main):
        for subset, lab in (("all", "All"), ("successful", "Successful")):
            r = main[(main["set"] == s) & (main["subset"] == subset)]
            if r.empty:
                continue
            r = r.iloc[0]
            body.append(f"{LABELS[s] if subset == 'all' else ''} & {lab} & {int(r['n_adv'])} & "
                        f"{with_ci(r['auroc'], r['auroc_lo'], r['auroc_hi'])} & "
                        f"{with_ci(r['tpr@5'], r['tpr@5_lo'], r['tpr@5_hi'], True)} & {pct(r['tpr@1'])}")
    note = ""
    if fpr is not None:
        r = fpr[fpr["method"] == "full"].iloc[0]
        note = (f" On {int(r['n_clean_test'])} clean test images the thresholds flag {pct(r['fpr@5'])} "
                f"(5\\% target) and {pct(r['fpr@1'])} (1\\% target).")
    cap = ("Screening performance of the proposed score. Thresholds are set on clean calibration images only; "
           "95\\% bootstrap confidence intervals in brackets. \\emph{Successful} keeps only images where the "
           "attack removed at least half of the clean detections." + note)
    return table(cap, "tab:main", "llrccc",
                 "Attack & Images & $n$ & AUROC [95\\% CI] & TPR@5\\% FPR [95\\% CI] & TPR@1\\% FPR",
                 body, wide=True)


def latex_by_method(df: pd.DataFrame, methods: list[str], labeler, caption: str, label: str,
                    extra_cols: dict | None = None) -> str:
    sets = sets_in(df)
    grid = {m: [float(df[(df["set"] == s) & (df["method"] == m)]["auroc"].iloc[0])
                if not df[(df["set"] == s) & (df["method"] == m)].empty else np.nan for s in sets] for m in methods}
    # The random patch is a control: a "best" detector there means nothing, so it is never bolded
    cols = {s: (bold_best([grid[m][k] for m in methods]) if s != "patch_random"
                else [f3(grid[m][k]) for m in methods]) for k, s in enumerate(sets)}
    body = []
    for i, m in enumerate(methods):
        cells = [labeler(m)] + ([extra_cols[m]] if extra_cols else []) + [cols[s][i] for s in sets]
        body.append(" & ".join(cells))
    head_extra = " & Box & Conf. & Feat." if extra_cols else ""
    header = ("Method" + head_extra + " & " + " & ".join(LABELS[s] for s in sets))
    colspec = "l" + ("ccc" if extra_cols else "") + "c" * len(sets)
    return table(caption, label, colspec, header, body, wide=True)


def latex_attacks(asr: pd.DataFrame | None, mp: pd.DataFrame | None) -> str:
    rows = []
    if mp is not None and "clean" in set(mp["set"]):
        c = mp[mp["set"] == "clean"].iloc[0]
        rows.append(f"Clean & -- & -- & {f3(c['mAP50'])} & {f3(c['mAP50_95'])}")
    for s in ATTACK_SETS:
        a = asr[asr["set"] == s] if asr is not None else pd.DataFrame()
        m = mp[mp["set"] == s] if mp is not None else pd.DataFrame()
        if a.empty and m.empty:
            continue
        sr = pct(a["attack_success_rate"].iloc[0]) if not a.empty else "--"
        dets = (f"{a['mean_clean_det'].iloc[0]:.1f} $\\rightarrow$ {a['mean_adv_det'].iloc[0]:.1f}"
                if not a.empty else "--")
        rows.append(f"{LABELS[s]} & {sr} & {dets} & {f3(m['mAP50'].iloc[0]) if not m.empty else '--'} & "
                    f"{f3(m['mAP50_95'].iloc[0]) if not m.empty else '--'}")
    cap = ("Effect of each attack on YOLOv8n: attack success rate (at least half of the clean detections lost), "
           "mean detections per image before and after the attack, and COCO mAP on the same images.")
    return table(cap, "tab:attacks", "lcccc", "Images & Success & Detections & mAP@0.50 & mAP@0.50:0.95", rows)


def latex_runtime(rt: pd.DataFrame) -> str:
    body = [f"{r['stage'].replace('screening: ', '')} & {r['median_ms']:.1f} & {r['p95_ms']:.1f} & "
            f"{r['overhead_vs_plain']:.2f}$\\times$" for _, r in rt.iterrows()]
    cap = "Latency on one laptop CPU (median and 95th percentile over 200 runs after 20 warm-up runs)."
    return table(cap, "tab:runtime", "lrrr", "Stage & Median (ms) & p95 (ms) & vs.\\ plain", body)


def latex_params(ps: pd.DataFrame) -> str:
    ps = ps.sort_values(["layer", "feat_mode"])
    body = [f"{int(r['layer'])} & {r['feat_mode']} & {f3(r['auroc_C_feat'])} & {f3(r['auroc_full'])}"
            for _, r in ps.iterrows()]
    cap = "Feature-layer and pooling selection on the development split only (AUROC against development PGD images)."
    return table(cap, "tab:params", "lccc", "Layer & Pooling & Feature only & Full score", body)


def cmd_latex(args) -> None:
    main = read("main_results")
    if main is None:
        sys.exit("No results yet. Run: python scripts/member_c.py tables")
    parts = ["% Generated by scripts/member_c.py latex. Preamble needs: \\usepackage{booktabs,adjustbox,amssymb}", ""]
    parts.append(latex_main(main, read("clean_fpr")))
    base = read("baselines")
    if base is not None:
        parts.append(latex_by_method(base, list(METHOD_LABELS), METHOD_LABELS.get,
                                     "AUROC of the proposed score against simpler alternatives on the same images. "
                                     "Best per attack in bold (not for the random-patch control).", "tab:baselines"))
    abl = read("ablation")
    if abl is not None:
        ticks = {m: " & ".join("\\checkmark" if v else "" for v in flags) for m, flags in ABLATION.items()}
        parts.append(latex_by_method(abl, list(ABLATION), lambda m: m.split("_")[0], "Ablation (Section 20): "
                                     "AUROC when combining subsets of the three consistency signals. Best per attack in bold "
                                     "(not for the random-patch control).",
                                     "tab:ablation", extra_cols=ticks))
    if read("attack_success") is not None or read("detection_map") is not None:
        parts.append(latex_attacks(read("attack_success"), read("detection_map")))
    if read("runtime") is not None:
        parts.append(latex_runtime(read("runtime")))
    if read("param_selection") is not None:
        parts.append(latex_params(read("param_selection")))
    out = TABLES / "paper_tables.tex"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)} ({len(parts) - 2} tables). Paste into Overleaf; preamble needs \\usepackage{{booktabs,adjustbox,amssymb}}.")


# ----------------------------------------------------------------------------- key numbers
def key_numbers() -> str:
    main, base, abl = read("main_results"), read("baselines"), read("ablation")
    if main is None:
        raise FileNotFoundError("Run: python scripts/member_c.py tables")
    lines = ["# Key numbers for the paper", "",
             "Generated from results/tables/*.csv. On Sunday, check every number in the paper against this file.", ""]
    fpr = read("clean_fpr")
    if fpr is not None:
        r = fpr[fpr["method"] == "full"].iloc[0]
        lines.append(f"- Clean test false-positive rate: {r['fpr@5']:.1%} at the 5% threshold, {r['fpr@1']:.1%} "
                     f"at the 1% threshold ({int(r['n_clean_test'])} images).")
    asr = read("attack_success")
    mp = read("detection_map")
    lines += ["", "## Per attack (all attempts)", ""]
    for s in sets_in(main):
        r = main[(main["set"] == s) & (main["subset"] == "all")].iloc[0]
        txt = (f"- **{PLAIN_LABELS[s]}**: AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}], "
               f"TPR {r['tpr@5']:.1%} at 5% FPR, {r['tpr@1']:.1%} at 1% FPR")
        if asr is not None and s in set(asr["set"]):
            txt += f"; attack success {asr[asr['set'] == s]['attack_success_rate'].iloc[0]:.1%}"
        if mp is not None and s in set(mp["set"]) and "clean" in set(mp["set"]):
            c = mp[mp["set"] == "clean"]["mAP50"].iloc[0]
            txt += f"; mAP@0.50 {c:.3f} -> {mp[mp['set'] == s]['mAP50'].iloc[0]:.3f}"
        lines.append(txt)

    lines += ["", "## Hypothesis check (facts only; the interpretation is yours)", ""]
    real = [s for s in sets_in(main) if s != "patch_random"]
    aur = {s: float(main[(main["set"] == s) & (main["subset"] == "all")]["auroc"].iloc[0]) for s in sets_in(main)}
    if fpr is not None:
        lines.append(f"- H1 (clean images stable): {fpr[fpr['method'] == 'full']['fpr@5'].iloc[0]:.1%} of clean test "
                     "images exceed the 5% threshold.")
    lines.append("- H2 (attacks cause inconsistency): AUROC above 0.75 for "
                 f"{sum(aur[s] > 0.75 for s in real)} of {len(real)} attack sets ("
                 + ", ".join(f"{PLAIN_LABELS[s]} {aur[s]:.2f}" for s in real) + ").")
    if abl is not None:
        wins = []
        for s in real:
            sub = abl[abl["set"] == s].set_index("method")["auroc"]
            single = max(sub.get("A_box", np.nan), sub.get("B_conf", np.nan), sub.get("C_feat", np.nan))
            wins.append(sub.get("G_box+conf+feat", np.nan) >= single - 1e-9)
        lines.append(f"- H3 (fusion helps): the three-signal score matches or beats the best single signal on "
                     f"{sum(wins)} of {len(wins)} attack sets.")
    rt = read("runtime")
    if rt is not None:
        e2e = rt[rt["stage"].str.contains("end to end")].iloc[0]
        plain = rt.iloc[0]
        lines.append(f"- H4 (overhead): end-to-end screening takes {e2e['median_ms']:.0f} ms median "
                     f"({e2e['overhead_vs_plain']:.1f}x the plain detector's {plain['median_ms']:.0f} ms).")
    if base is not None:
        lines += ["", "## Against the no-transform baseline", ""]
        for s in real:
            b = base[base["set"] == s].set_index("method")["auroc"]
            if "no_transform" in b and "full" in b:
                lines.append(f"- {PLAIN_LABELS[s]}: full {b['full']:.3f} vs no-transform {b['no_transform']:.3f}")
    if "patch_random" in aur:
        lines += ["", f"Control: random patch AUROC {aur['patch_random']:.3f} (near 0.5 means the screen does not "
                  "simply react to occlusion)."]
    return "\n".join(lines) + "\n"


def cmd_numbers(args) -> None:
    out = TABLES / "key_numbers.md"
    out.write_text(key_numbers(), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(f"Saved {out.relative_to(ROOT)}")


def cmd_all(args) -> None:
    code = report(run_checks(), "All results are in and consistent.")
    if code:
        print("\nContinuing anyway; fix the FAIL lines before trusting the numbers.\n")
    cmd_tables(args)
    cmd_latex(args)
    cmd_numbers(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="which results exist and whether they are consistent")
    p = sub.add_parser("tables", help="evaluation tables and figures")
    p.add_argument("--n-boot", default="1000")
    sub.add_parser("latex", help="LaTeX tables for the IEEE paper")
    sub.add_parser("numbers", help="key numbers and hypothesis check")
    p = sub.add_parser("all", help="check + tables + latex + numbers")
    p.add_argument("--n-boot", default="1000")
    args = ap.parse_args()
    {"check": cmd_check, "tables": cmd_tables, "latex": cmd_latex, "numbers": cmd_numbers, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
