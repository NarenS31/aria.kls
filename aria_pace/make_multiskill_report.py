"""docs/aria_pace/MULTISKILL.md from results/multiskill_val.json (no recomputation)."""
import json

from .constants import RESULTS, ROOT

DOCS = ROOT.parent / "docs" / "aria_pace"


def main():
    d = json.load(open(RESULTS / "multiskill_val.json"))
    sd, ss = json.load(open(RESULTS / "dkt_val.json")), json.load(open(RESULTS / "sakt_val.json"))
    sing = {"ms_dkt": sd["paired_dkt_minus_bkt"], "ms_sakt": ss["paired_sakt_minus_bkt"]}
    L = ["# Multi-skill DKT and SAKT (Amendment A5)", "",
         f"**Status: {d['status']}.** Source: `aria_pace/results/multiskill_val.json`, produced by `python -m aria_pace.run_multiskill`. 2009 train and validation only; no locked split was read. "
         "Variant A (three correct in a row, T = 10). The models see the student's earlier attempts on **other skills** (most recent 90 rows up to the sequence's k-th opportunity); "
         "the target is still the current sequence's wheel-spinning label.", "",
         f"**Pre-registered rule (A5.4):** {d['decision']['rule']}. **Outcome: {'a model was promoted: STOP (A5.5)' if d['decision']['promoted'] else 'no multi-skill model is promoted; BKT stays the primary risk score.'}**", "",
         "## Wheel-spinning AUC on validation vs BKT", "",
         "| Model | k | N | AUC BKT [95% CI] | AUC model [95% CI] | Paired difference [95% CI] | Single-skill version (A3.5 / A3a), same k |", "|---|---|---|---|---|---|---|"]
    for m, name in (("ms_dkt", "multi-skill DKT"), ("ms_sakt", "multi-skill SAKT")):
        for k in (3, 5, 8):
            a = d["auc_variantA_val"][f"k{k}"]; b, s = a["bkt"], a[m]; p = d["paired_minus_bkt_variantA"][m][f"k{k}"]; o = sing[m][f"variantA_k{k}"]
            ci = lambda x: f"{x['auc']:.4f} [{x['auc_ci95_student_bootstrap'][0]:.4f}, {x['auc_ci95_student_bootstrap'][1]:.4f}]"  # noqa: E731
            L.append(f"| {name} | {k}{' (does not decide)' if k == 8 else ''} | {b['n']} | {ci(b)} | {ci(s)} | {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {o['diff']:+.4f} [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}] |")
    L += ["", "Decision (mechanical): " + "; ".join(
        f"{m}: k=3 diff {v['per_k']['k3']['diff']:+.4f} (>= +0.01: {v['per_k']['k3']['diff_ge_0.01']}, CI lower bound > 0: {v['per_k']['k3']['ci_excludes_0_above']}), "
        f"k=5 diff {v['per_k']['k5']['diff']:+.4f} (>= +0.01: {v['per_k']['k5']['diff_ge_0.01']}, CI lower bound > 0: {v['per_k']['k5']['ci_excludes_0_above']}) -> promote: {v['promote']}"
        for m, v in d["decision"].items() if m.startswith("ms_")), "",
         "## Next-response log-loss (matched set: every non-first opportunity of every validation sequence)", "",
         "| Model | Chosen size | Log-loss model | Log-loss BKT | Difference [95% paired student-bootstrap CI] | Steps / students |", "|---|---|---|---|---|---|"]
    for m, name in (("dkt", "multi-skill DKT"), ("sakt", "multi-skill SAKT")):
        o = d["models"][m]["matched_next_response_logloss"]
        L.append(f"| {name} | {d['models'][m]['chosen_size']} | {o['model']:.4f} | {o['bkt']:.4f} | {o['diff_model_minus_bkt']:+.4f} [{o['diff_ci95'][0]:+.4f}, {o['diff_ci95'][1]:+.4f}] | {o['n_steps']} / {o['n_students']} |")
    L += ["", "Reading: the multi-skill LSTM predicts the **next response** better than BKT (the interval excludes 0), but that does not carry over to the wheel-spinning risk, "
          "where every k = 3 and k = 5 difference is below the +0.01 margin and every interval includes 0. Next-response accuracy and wheel-spinning discrimination are different questions.", "",
          "Approximations (A5.7): the outcome tree assumes the rest of the sequence follows immediately; the models see token order, not timestamps; windows reset state. "
          "Selection of the size used validation next-response log-loss only; the label was never used to choose anything.", ""]
    (DOCS / "MULTISKILL.md").write_text("\n".join(L))
    print("wrote docs/aria_pace/MULTISKILL.md")


if __name__ == "__main__":
    main()
