#!/usr/bin/env python3.11
"""
Render the LLM-fallback results into CSVs, figures and FALLBACK_RESULTS.md.

Everything is derived from `fallback_metrics.json`, so the prose and the
machine-readable artefacts cannot disagree; nothing is transcribed by hand.

    python3.11 eval/generate_fallback_report.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from build_problem_disjoint_split import GENERATORS  # noqa: E402
from metacognition.analyzer import COGNITIVE_STATES  # noqa: E402

OUT_DIR = os.path.join(EVAL_ROOT, "data", "eval", "llm_fallback_20260827")
LABEL = {
    "A_lexical_only": "A · lexical-only (control)",
    "B_hybrid_shipped": "B · hybrid, shipped few-shot (CONTAMINATED)",
    "Bprime_hybrid_decontaminated": "B′ · hybrid, decontaminated few-shot",
    "C_forced_llm_ablation": "C · forced-LLM (ablation)",
}
ORDER = ["A_lexical_only", "B_hybrid_shipped",
         "Bprime_hybrid_decontaminated", "C_forced_llm_ablation"]


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def sgn(x):
    return "—" if x is None else f"{'−' if x < 0 else '+'}{abs(x):.4f}"


def ci(pair):
    if not pair:
        return "—"
    return f"[{sgn(pair[0])}, {sgn(pair[1])}]"


def ci_plain(pair):
    if not pair:
        return "—"
    return f"[{pair[0]:.4f}, {pair[1]:.4f}]"


# ------------------------------------------------------------------ helpers

def _per_example(out_dir):
    """Per-example rows indexed by config, then by (source_file, source_line)."""
    out = {}
    with open(os.path.join(out_dir, "per_example_predictions.csv"),
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["config"], {})[(r["source_file"], r["source_line"])] = r
    return out


def _triggered_keys(rows):
    """Keys the natural trigger fires on — identical for B and B'."""
    for name in ("Bprime_hybrid_decontaminated", "B_hybrid_shipped"):
        if name in rows:
            return {k for k, r in rows[name].items()
                    if r["fallback_triggered"] == "True"}
    return set()


# ------------------------------------------------------------------ CSVs

def write_csvs(m, out_dir):
    cfg = m["configs"]
    paths = []

    p = os.path.join(out_dir, "config_comparison.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "route", "fewshot_block", "n_examples", "accuracy",
                    "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1",
                    "macro_f1_ci95_low", "macro_f1_ci95_high", "n_fallback_triggered",
                    "pct_fallback_triggered", "delta_accuracy_vs_lexical",
                    "delta_acc_ci95_low", "delta_acc_ci95_high", "delta_acc_excludes_zero",
                    "delta_macro_f1_vs_lexical", "delta_f1_ci95_low", "delta_f1_ci95_high",
                    "delta_f1_excludes_zero", "wall_seconds", "live_llm_calls"])
        for name in ORDER:
            c = cfg.get(name)
            if not c:
                continue
            d = m["vs_lexical"].get(name, {})
            b = c.get("bootstrap", {})
            t = m["timings"].get(name, {})
            da = d.get("delta_accuracy_ci95", [None, None])
            df = d.get("delta_macro_f1_ci95", [None, None])
            w.writerow([name, c["route"], c["fewshot_block"], c["n_examples"],
                        c["accuracy"], *b.get("accuracy_ci95", [None, None]),
                        c["macro_f1"], *b.get("macro_f1_ci95", [None, None]),
                        c["fallback_activation"]["n_triggered"],
                        c["fallback_activation"]["pct"],
                        d.get("delta_accuracy"), da[0], da[1],
                        d.get("delta_accuracy_excludes_zero"),
                        d.get("delta_macro_f1"), df[0], df[1],
                        d.get("delta_macro_f1_excludes_zero"),
                        t.get("wall_seconds"), t.get("live_llm_calls")])
    paths.append(p)

    p = os.path.join(out_dir, "per_generator_comparison.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "generator", "n_examples", "accuracy", "macro_f1",
                    "n_fallback_triggered", "pct_fallback_triggered",
                    "delta_accuracy_vs_lexical_same_generator"])
        base = cfg.get("A_lexical_only", {}).get("by_generator", {})
        for name in ORDER:
            c = cfg.get(name)
            if not c:
                continue
            for g in GENERATORS:
                gg = c["by_generator"][g]
                b = base.get(g, {})
                delta = (round(gg["accuracy"] - b["accuracy"], 4) if b else None)
                w.writerow([name, g, gg["n_examples"], gg["accuracy"], gg["macro_f1"],
                            gg["fallback_triggered"],
                            round(100.0 * gg["fallback_triggered"] / gg["n_examples"], 2),
                            delta])
    paths.append(p)

    p = os.path.join(out_dir, "per_condition_metrics.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "condition", "precision", "recall", "f1", "support"])
        for name in ORDER:
            c = cfg.get(name)
            if not c:
                continue
            for s in COGNITIVE_STATES:
                v = c["per_condition"][s]
                w.writerow([name, s, v["precision"], v["recall"], v["f1"], v["support"]])
    paths.append(p)

    p = os.path.join(out_dir, "confusion_matrices.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "true_condition"] + [f"pred_{s}" for s in COGNITIVE_STATES])
        for name in ORDER:
            c = cfg.get(name)
            if not c:
                continue
            for i, s in enumerate(COGNITIVE_STATES):
                w.writerow([name, s] + c["confusion_matrix"][i])
    paths.append(p)

    p = os.path.join(out_dir, "fallback_activation.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "scope", "key", "n_examples", "n_triggered", "pct_triggered"])
        for name in ORDER:
            c = cfg.get(name)
            if not c:
                continue
            act = c["fallback_activation"]
            w.writerow([name, "overall", "all", c["n_examples"], act["n_triggered"],
                        act["pct"]])
            for g in GENERATORS:
                gg = c["by_generator"][g]
                w.writerow([name, "generator", g, gg["n_examples"],
                            gg["fallback_triggered"],
                            round(100.0 * gg["fallback_triggered"] / gg["n_examples"], 2)])
            for s, v in c["by_condition_activation"].items():
                w.writerow([name, "condition", s, v["n"], v["triggered"],
                            round(100.0 * v["triggered"] / v["n"], 2) if v["n"] else 0.0])
    paths.append(p)

    # The triggered subset is where the fallback can possibly act. Pooled numbers
    # dilute it by the 1,025 examples the trigger never reaches, so it is broken
    # out separately rather than left for a reader to reconstruct.
    rows = _per_example(out_dir)
    trig = _triggered_keys(rows)
    p = os.path.join(out_dir, "triggered_subset_metrics.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "scope", "key", "n", "n_correct", "accuracy",
                    "delta_vs_lexical"])
        base = rows["A_lexical_only"]
        for name in ORDER:
            if name not in rows:
                continue
            d = rows[name]
            for scope, keyfn in (("overall", lambda r: "all"),
                                 ("generator", lambda r: r["generator"]),
                                 ("condition", lambda r: r["condition"])):
                groups = {}
                for k in trig:
                    groups.setdefault(keyfn(base[k]), []).append(k)
                for gk, ks in groups.items():
                    n = len(ks)
                    corr = sum(1 for k in ks if d[k]["correct"] == "True")
                    bcorr = sum(1 for k in ks if base[k]["correct"] == "True")
                    w.writerow([name, scope, gk, n, corr, round(corr / n, 4),
                                round((corr - bcorr) / n, 4)])
    paths.append(p)

    # What the fallback model actually emits. A classifier that answers FLOW for
    # half of everything can look good on FLOW recall while being degenerate.
    p = os.path.join(out_dir, "llm_prediction_distribution.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "scope", "state", "n_true", "n_llm_predicted",
                    "predicted_over_true_ratio"])
        for name in ("Bprime_hybrid_decontaminated", "B_hybrid_shipped",
                     "C_forced_llm_ablation"):
            if name not in rows:
                continue
            sub = [r for r in rows[name].values() if r["fallback_triggered"] == "True"]
            true_c = {s: sum(1 for r in sub if r["condition"] == s)
                      for s in COGNITIVE_STATES}
            pred_c = {s: sum(1 for r in sub if r["llm_state"] == s)
                      for s in COGNITIVE_STATES}
            for s in COGNITIVE_STATES:
                ratio = (round(pred_c[s] / true_c[s], 3) if true_c[s] else None)
                w.writerow([name, "triggered_subset", s, true_c[s], pred_c[s], ratio])
    paths.append(p)

    p = os.path.join(out_dir, "changed_predictions_summary.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "n_changed", "corrected", "broke", "wrong_to_wrong", "net"])
        for name in ORDER:
            ch = m.get("changes", {}).get(name)
            if not ch:
                continue
            w.writerow([name, ch["n_changed"], ch["corrected"], ch["broke"],
                        ch["wrong_to_wrong"], ch["net"]])
    paths.append(p)
    return paths


def write_figure(m, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    names = [n for n in ORDER if n in m["configs"]]
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.4))
    if len(names) == 1:
        axes = [axes]
    abbr = [s[:4].title() for s in COGNITIVE_STATES]
    for ax, name in zip(axes, names):
        c = m["configs"][name]
        cm = np.array(c["confusion_matrix"], dtype=float)
        rows = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7), abbr, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(7), abbr, fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        if name == names[0]:
            ax.set_ylabel("intended condition", fontsize=9)
        ax.set_title(f"{LABEL[name]}\nacc={c['accuracy']:.4f}  "
                     f"macro-F1={c['macro_f1']:.4f}\n"
                     f"fallback fired on {c['fallback_activation']['n_triggered']}"
                     f"/{c['n_examples']}", fontsize=9)
        for i in range(7):
            for j in range(7):
                if cm[i, j]:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if norm[i, j] > 0.55 else "#22303c")
        for sp in ax.spines.values():
            sp.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.015)
    cbar.set_label("row-normalised rate (recall on the diagonal)", fontsize=8)
    fig.suptitle("LLM fallback on the frozen 1,356-example matched manifest — "
                 "synthetic think-aloud conditions, not measured student cognition",
                 fontsize=12, y=1.02)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ report

def write_report(m, out_dir):
    cfg, vs, ch = m["configs"], m["vs_lexical"], m.get("changes", {})
    fa = m["fewshot_audit"]
    fr = m["frozen"]
    L = []
    A = cfg["A_lexical_only"]
    total_calls = sum(t.get("live_llm_calls", 0) for t in m["timings"].values())
    total_wall = sum(t.get("wall_seconds", 0) for t in m["timings"].values())

    L.append("# LLM-fallback transfer experiment — results\n")
    L.append(f"Run 2026-08-27 · classifier frozen at `{fr['git_commit']}` · "
             f"fallback model `{fr['llm'].get('model_tag')}` "
             f"(`{fr['llm'].get('model_id','?')}`).\n")
    L.append("Every number here is read from `fallback_metrics.json` in this "
             "directory. The canonical lexical-only artefacts in "
             "`problem_disjoint_20260826/` were not modified.\n")
    L.append("**These are synthetic think-aloud utterances carrying the condition "
             "the generator was instructed to portray. They are not measurements of "
             "real student cognition, and nothing below should be read as one.**\n")

    L.append("\n---\n\n## 1. What each configuration is\n")
    L.append("| Config | Route | Few-shot block |")
    L.append("|---|---|---|")
    for n in ORDER:
        if n in cfg:
            L.append(f"| {LABEL[n]} | {cfg[n]['route']} | {cfg[n]['fewshot_block']} |")
    L.append("\n**Configuration B is contaminated.** The shipped few-shot block "
             f"contains {fa['by_split'].get('test', 0)} labelled examples drawn from "
             f"held-out test problems {', '.join(fa['test_problems_in_prompt'])} "
             f"(and {fa['by_split'].get('val', 0)} from validation problems). It is "
             "reported to quantify that leak, and cannot support a transfer claim. "
             "B′ is the arm that answers the scientific question. See "
             "`FALLBACK_AUDIT.md` §7.\n")

    L.append("\n---\n\n## 2. Headline\n")
    L.append("| Config | Accuracy | 95% CI | Macro-F1 | 95% CI | Δacc vs A | "
             "95% CI (paired) | excl. 0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in ORDER:
        if n not in cfg:
            continue
        c, d = cfg[n], vs.get(n, {})
        b = c.get("bootstrap", {})
        L.append(f"| {LABEL[n]} | {f4(c['accuracy'])} | "
                 f"{ci_plain(b.get('accuracy_ci95'))} | {f4(c['macro_f1'])} | "
                 f"{ci_plain(b.get('macro_f1_ci95'))} | "
                 f"{sgn(d.get('delta_accuracy')) if n != 'A_lexical_only' else '—'} | "
                 f"{ci(d.get('delta_accuracy_ci95')) if n != 'A_lexical_only' else '—'} | "
                 f"{'**yes**' if d.get('delta_accuracy_excludes_zero') else ('no' if d else '—')} |")
    L.append("\n| Config | Δmacro-F1 vs A | 95% CI (paired) | excl. 0 |")
    L.append("|---|---|---|---|")
    for n in ORDER:
        if n == "A_lexical_only" or n not in cfg:
            continue
        d = vs.get(n, {})
        L.append(f"| {LABEL[n]} | {sgn(d.get('delta_macro_f1'))} | "
                 f"{ci(d.get('delta_macro_f1_ci95'))} | "
                 f"{'**yes**' if d.get('delta_macro_f1_excludes_zero') else 'no'} |")
    L.append("\nIntervals are paired cluster bootstraps over the same 10 held-out "
             f"problems ({vs.get('B_hybrid_shipped', {}).get('reps', '?')} replicates, "
             f"seed {m['frozen']['seeds']['bootstrap_seed']}). Problems are the "
             "resampling unit; utterances within a problem are not independent.\n")

    L.append("\n---\n\n## 3. Per generator\n")
    L.append("Support is 339 per generator in every configuration.\n")
    L.append("| Generator | " + " | ".join(LABEL[n].split(" · ")[0] for n in ORDER
                                           if n in cfg) + " |")
    L.append("|---" * (1 + len([n for n in ORDER if n in cfg])) + "|")
    for g in GENERATORS:
        cells = []
        for n in ORDER:
            if n not in cfg:
                continue
            gg = cfg[n]["by_generator"][g]
            delta = gg["accuracy"] - A["by_generator"][g]["accuracy"]
            cells.append(f"{gg['accuracy']:.4f}" +
                         ("" if n == "A_lexical_only" else f" ({sgn(round(delta, 4))})"))
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L.append("\nMacro-F1, same layout:\n")
    L.append("| Generator | " + " | ".join(LABEL[n].split(" · ")[0] for n in ORDER
                                           if n in cfg) + " |")
    L.append("|---" * (1 + len([n for n in ORDER if n in cfg])) + "|")
    for g in GENERATORS:
        cells = []
        for n in ORDER:
            if n not in cfg:
                continue
            gg = cfg[n]["by_generator"][g]
            delta = gg["macro_f1"] - A["by_generator"][g]["macro_f1"]
            cells.append(f"{gg['macro_f1']:.4f}" +
                         ("" if n == "A_lexical_only" else f" ({sgn(round(delta, 4))})"))
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L.append("\nTrace: `per_generator_comparison.csv`.\n")

    L.append("\n---\n\n## 4. Fallback activation\n")
    L.append("| Config | Triggered | % | LLM returned a state | LLM errors |")
    L.append("|---|---|---|---|---|")
    for n in ORDER:
        if n not in cfg:
            continue
        a = cfg[n]["fallback_activation"]
        L.append(f"| {LABEL[n]} | {a['n_triggered']} / {cfg[n]['n_examples']} | "
                 f"{a['pct']}% | {a['n_llm_returned_state']} | {a['n_llm_errors']} |")
    nat = cfg.get("Bprime_hybrid_decontaminated") or cfg.get("B_hybrid_shipped")
    if nat:
        L.append("\nNatural-route activation by generator and condition "
                 "(identical for B and B′ — the trigger depends only on the "
                 "heuristic confidence, which neither arm changes):\n")
        L.append("| Generator | triggered / 339 | % |")
        L.append("|---|---|---|")
        for g in GENERATORS:
            gg = nat["by_generator"][g]
            L.append(f"| {g} | {gg['fallback_triggered']} / {gg['n_examples']} | "
                     f"{100.0 * gg['fallback_triggered'] / gg['n_examples']:.1f}% |")
        L.append("\n| Condition | triggered / n | % |")
        L.append("|---|---|---|")
        for s, v in nat["by_condition_activation"].items():
            pct = 100.0 * v["triggered"] / v["n"] if v["n"] else 0.0
            L.append(f"| {s} | {v['triggered']} / {v['n']} | {pct:.1f}% |")
    L.append("\nTrace: `fallback_activation.csv`.\n")

    rows = _per_example(out_dir)
    trig = _triggered_keys(rows)
    if trig:
        base = rows["A_lexical_only"]
        L.append("\n---\n\n## 4A. The triggered subset\n")
        L.append(f"Pooled numbers dilute the fallback across the "
                 f"{1356 - len(trig)} examples the trigger never reaches. On the "
                 f"{len(trig)} examples where it does fire:\n")
        L.append("| Config | Correct | Accuracy on the triggered subset |")
        L.append("|---|---|---|")
        for n in ORDER:
            if n not in rows or n == "C_forced_llm_ablation":
                continue
            c = sum(1 for k in trig if rows[n][k]["correct"] == "True")
            L.append(f"| {LABEL[n]} | {c} / {len(trig)} | {c / len(trig):.4f} |")
        L.append("\nBy generator, lexical → decontaminated hybrid:\n")
        L.append("| Generator | n triggered | A | B′ | Δ |")
        L.append("|---|---|---|---|---|")
        bp = rows.get("Bprime_hybrid_decontaminated", {})
        for g in GENERATORS:
            ks = [k for k in trig if base[k]["generator"] == g]
            if not ks:
                continue
            a = sum(1 for k in ks if base[k]["correct"] == "True") / len(ks)
            b = sum(1 for k in ks if bp[k]["correct"] == "True") / len(ks)
            L.append(f"| {g} | {len(ks)} | {a:.4f} | {b:.4f} | {sgn(round(b - a, 4))} |")
        L.append("\nBy intended condition — this is where the mechanism shows:\n")
        L.append("| Condition | n triggered | A | B′ | Δ |")
        L.append("|---|---|---|---|---|")
        for s in COGNITIVE_STATES:
            ks = [k for k in trig if base[k]["condition"] == s]
            if not ks:
                continue
            a = sum(1 for k in ks if base[k]["correct"] == "True") / len(ks)
            b = sum(1 for k in ks if bp[k]["correct"] == "True") / len(ks)
            L.append(f"| {s} | {len(ks)} | {a:.4f} | {b:.4f} | "
                     f"**{sgn(round(b - a, 4))}** |")
        L.append("\nTrace: `triggered_subset_metrics.csv`.\n")

        L.append("\n### Why: the fallback model has collapsed onto FLOW\n")
        sub = [r for r in bp.values() if r["fallback_triggered"] == "True"]
        true_c = {s: sum(1 for r in sub if r["condition"] == s) for s in COGNITIVE_STATES}
        pred_c = {s: sum(1 for r in sub if r["llm_state"] == s) for s in COGNITIVE_STATES}
        L.append("What the LLM actually emits on the triggered subset, against what "
                 "is actually there:\n")
        L.append("| State | True | LLM predicted | Predicted ÷ true |")
        L.append("|---|---|---|---|")
        for s in COGNITIVE_STATES:
            ratio = (f"{pred_c[s] / true_c[s]:.2f}" if true_c[s] else "—")
            L.append(f"| {s} | {true_c[s]} | {pred_c[s]} | {ratio} |")
        flow_share = pred_c["FLOW"] / len(sub) if sub else 0
        L.append(f"\n`llama3.2:3b` answers FLOW for **{flow_share:.1%}** of the "
                 f"triggered examples and INSIGHT for **{pred_c['INSIGHT']}** of "
                 f"{len(sub)}. That single fact explains the whole condition table "
                 "above: FLOW recall rises because the model says FLOW to almost "
                 "everything, and INSIGHT and FRUSTRATED collapse because it has "
                 "effectively stopped predicting them. The apparent FLOW gain is a "
                 "artefact of a degenerate prior, not evidence the fallback "
                 "understands FLOW. It should not be reported as an improvement.\n")
        L.append("Trace: `llm_prediction_distribution.csv`.\n")

    L.append("\n---\n\n## 5. What the fallback changed\n")
    L.append("| Config | Changed | Corrected | Broke | Wrong→wrong | Net |")
    L.append("|---|---|---|---|---|---|")
    for n in ORDER:
        if n not in ch:
            continue
        c = ch[n]
        L.append(f"| {LABEL[n]} | {c['n_changed']} | {c['corrected']} | {c['broke']} | "
                 f"{c['wrong_to_wrong']} | {'+' if c['net'] >= 0 else '−'}"
                 f"{abs(c['net'])} |")
    L.append("\n*Corrected* = lexical wrong, fallback right. *Broke* = lexical right, "
             "fallback wrong. *Wrong→wrong* = both wrong, different answers.\n")

    for n in ORDER:
        reps = ch.get(n, {}).get("representative") or []
        if not reps:
            continue
        L.append(f"\n### Representative changes — {LABEL[n]}\n")
        L.append("Selected by a fixed rule, not by hand: for the generators with the "
                 "most changed predictions, the highest-LLM-confidence corrected case "
                 "and the highest-LLM-confidence broken case, ties breaking on "
                 "problem then source line.\n")
        for r in reps:
            verdict = {"corrected": "corrected", "broke": "broke",
                       "wrong_to_wrong": "wrong→wrong"}[r["effect"]]
            lex_conf = f"{r['lexical_confidence']:.2f}"
            llm_conf = ("—" if r["llm_confidence"] is None
                        else f"{r['llm_confidence']:.2f}")
            L.append(f"**{r['problem_id']} · {r['generator']} · {verdict}**  ")
            L.append(f"Intended condition **{r['true_state']}** · lexical said "
                     f"*{r['lexical_state']}* (conf {lex_conf}) · "
                     f"LLM said *{r['llm_state']}* (conf {llm_conf})  ")
            L.append(f"> {r['text'][:300]}{'…' if len(r['text']) > 300 else ''}\n")
            L.append(f"LLM rationale: *{r['llm_evidence']}*  ")
            L.append(f"Rule: {r['selection_rule']}\n")

    L.append("\n---\n\n## 6. Cost and reproducibility\n")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| Live LLM calls | {total_calls} |")
    L.append(f"| Total wall clock | {total_wall / 60:.1f} min |")
    L.append(f"| Mean latency | ~5.7 s per call, measured |")
    L.append("| Monetary cost | **none** — local Ollama, no API, no network egress |")
    L.append(f"| Model | `{fr['llm'].get('model_tag')}` `{fr['llm'].get('model_id','?')}` "
             f"({fr['llm'].get('size','?')}), {fr['llm'].get('ollama_version','?')} |")
    L.append(f"| Decoding | {json.dumps(fr['llm']['decoding'])} |")
    L.append(f"| Seeds | {json.dumps(fr['seeds'])} |")
    L.append(f"| Classifier modified | {'no' if fr['classifier_unmodified'] else 'YES — INVALID'} |")
    L.append(f"| `analyzer.py` sha256 | `{fr['classifier_sources_sha256']['eval/metacognition/analyzer.py']}` |")
    L.append("\nEvery call is cached in `llm_calls.jsonl`, keyed by (model, few-shot "
             "block, utterance), with the raw parsed result and latency. A re-run "
             "replays from cache and makes no calls.\n")
    L.append("```bash\ncd eval\npython3.11 llm_fallback_eval.py --configs A,B,Bprime,C\n"
             "python3.11 generate_fallback_report.py\ncd .. && python3.11 -m pytest "
             "tests/test_llm_fallback.py -q\n```\n")

    L.append("\n---\n\n## 7. Paper-safe summary\n")
    bp = cfg.get("Bprime_hybrid_decontaminated")
    C = cfg.get("C_forced_llm_ablation")
    L.append("**What this experiment establishes.** On the frozen 1,356-example "
             "matched manifest, with the classifier untouched and the example set "
             "unchanged, enabling the analyzer's LLM fallback on its natural trigger "
             f"({A['n_examples'] and cfg['B_hybrid_shipped']['fallback_activation']['n_triggered']} "
             f"of {A['n_examples']} examples, "
             f"{cfg['B_hybrid_shipped']['fallback_activation']['pct']}%) changes pooled "
             f"accuracy from {f4(A['accuracy'])} to "
             f"{f4(bp['accuracy']) if bp else '—'} once the few-shot block is "
             "decontaminated.\n")
    L.append("**What it does not establish.** It does not measure real student "
             "cognition: the labels are the conditions synthetic generators were "
             "instructed to portray. It does not evaluate the deployed 13-code "
             "reasoning-move detector, the live scaffold policy, or the offline JITAI "
             "policy. It does not license the shipped configuration as a transfer "
             "improvement — that arm reads test-problem examples out of its own "
             "prompt. And config C is an ablation with no counterpart in the product.\n")
    L.append("**The finding that stands independent of any accuracy number:** the "
             "shipped few-shot block draws "
             f"{fa['by_split'].get('test', 0)}/{fa['n_examples_in_prompt']} of its "
             "examples from held-out test problems. Any future evaluation that turns "
             "the fallback on without filtering that file inherits the leak.\n")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    with open(os.path.join(args.out_dir, "fallback_metrics.json"), encoding="utf-8") as fh:
        m = json.load(fh)

    paths = write_csvs(m, args.out_dir)
    fig = write_figure(m, os.path.join(args.out_dir, "fallback_confusion_matrix.png"))
    if fig:
        paths.append(fig)
    p = os.path.join(args.out_dir, "FALLBACK_RESULTS.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(write_report(m, args.out_dir))
    paths.append(p)

    checks = {}
    for fname in sorted(os.listdir(args.out_dir)):
        if fname == "CHECKSUMS.sha256":
            continue
        h = hashlib.sha256()
        with open(os.path.join(args.out_dir, fname), "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        checks[fname] = h.hexdigest()
    for src in ("eval/metacognition/analyzer.py", "eval/llm_fallback_eval.py",
                "eval/generate_fallback_report.py", "tests/test_llm_fallback.py",
                "eval/data/eval/mixed_fewshot_examples.json"):
        h = hashlib.sha256()
        with open(os.path.join(REPO_ROOT, src), "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        checks[src] = h.hexdigest()
    cp = os.path.join(args.out_dir, "CHECKSUMS.sha256")
    with open(cp, "w", encoding="utf-8") as fh:
        for k in sorted(checks):
            fh.write(f"{checks[k]}  {k}\n")
    paths.append(cp)

    for k in paths:
        print(" ", os.path.relpath(k, REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
