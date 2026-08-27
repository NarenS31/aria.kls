# VERIFIED RESULTS — paper-safe numbers

Leakage-free evaluation of the ARIA **offline seven-state `CognitiveStateAnalyzer`**.
Run 2026-08-26. Classifier frozen at commit `ef5b7d6c50bd05c4b6797e0eedf220c8f81b7183`.

Every number on this page was produced by executing the evaluation and is traceable to a named
field in a machine-readable artefact in this directory. Nothing here is estimated, rounded from
memory, or carried over from an earlier run. Numbers that did **not** survive verification are
listed in §7 so they are not cited by mistake.

---

## 1. Scope — what these numbers describe

| Component | Evaluated here? |
|---|---|
| Offline seven-state `CognitiveStateAnalyzer` (`eval/metacognition/analyzer.py`) | **Yes — this and only this.** Offline, `use_llm=False`, 100% heuristic route in every condition. |
| Deployed 13-code reasoning-move detector (`agent/reasoning_moves.py`) | No. No human-labelled 13-code corpus exists. |
| Live hand-written scaffold policy (`agent/nudge.py`, `agent/intervention_pipeline.py`) | No. Not run, not modified. |
| Offline JITAI policy (`eval/metacognition/jitai.py`) | No. Not part of this experiment. |

States: PLANNING, FLOW, CONFUSED, RUSHING, FRUSTRATED, STUCK, INSIGHT.
Data: synthetic LLM-generated think-aloud text with LLM-assigned labels. **No human-labelled ground
truth is involved anywhere in this evaluation.**

---

## 2. Split — citable as leakage-free

50 problems partitioned **at the problem level**, seed `20260826`, into 30 train / 10 validation /
10 test. Every utterance derived from a problem — all seven states, all personas, all paraphrases,
all four generators — stays with its problem.

| Split | Problems | Examples |
|---|---|---|
| train | 30 | 5,332 |
| val | 10 | 1,702 |
| test | 10 | 1,716 |

All seven states are present in every split and in every (split × generator) cell.
All pairwise overlaps between splits are **0** — for problem ID, normalized-problem SHA-256,
normalized problem text, and utterance text. Independently, no utterance in the 8,750-utterance pool
occurs under more than one problem.

*Trace:* `split_manifest.json → distributions`, `→ leakage_check`; asserted by 36 tests in
`tests/test_problem_disjoint_split.py`, including a negative control that plants a leak and confirms
the assertion fires.

---

## 3. Primary result — problem leakage did not inflate the previous score

| Condition | n | Problems | Accuracy | 95% CI | Macro-F1 | 95% CI |
|---|---|---|---|---|---|---|
| Seen-problem / utterance-disjoint (previous) | 350 | 50 | 0.8457 | [0.8042, 0.8838] | 0.8373 | [0.7926, 0.8747] |
| **Problem-disjoint** | 695 | 10 | **0.8619** | [0.8317, 0.8881] | **0.8567** | [0.8271, 0.8836] |

Difference: **+0.0162** accuracy, **+0.0194** macro-F1 — the estimate moves **up**, and the
intervals overlap heavily.

**Citable claim.** Removing problem-level leakage did not reduce measured performance. The
previously reported 0.8457 / 0.8373 is **not** shown to be inflated by problem overlap.

**Do not write** that leakage was found and removed, or that the earlier number was inflated.
Neither is supported.

Controlled version, holding the utterance pool and classifier fixed and varying only the problem
group:

| llama3.1:8b pool | n | Problems | Accuracy | 95% CI |
|---|---|---|---|---|
| 30 train problems (seen) | 2,114 | 30 | 0.8477 | [0.8328, 0.8617] |
| 10 test problems (unseen) | 695 | 10 | 0.8619 | [0.8317, 0.8881] |

Unseen problems score **+0.0142** higher than seen ones. The interpretation that survives scrutiny:
a fixed bank of hand-written lexical markers has no per-problem parameter to overfit, so there was
no problem-level leakage for this classifier family to exploit. This is evidence about the
classifier family, not a clean bill of health for the corpus (§6, limitation 2).

Majority-class baseline (chosen on train only): 0.1429 utterance-level, 0.1396 problem-disjoint,
against a near-uniform seven-class corpus.

*Trace:* `metrics.json → conditions.A_utterance_level`, `→ conditions.B_problem_disjoint`,
`→ conditions.B_train_problems`, `→ utterance_level_vs_problem_disjoint`,
`→ seen_vs_unseen_problems_same_pool`; `comparison_utterance_vs_problem_disjoint.csv`.

---

## 4. Per-state results, problem-disjoint (llama3.1:8b, n=695)

| State | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PLANNING | 0.828 | 0.960 | 0.889 | 100 |
| **FLOW** | 0.895 | **0.505** | **0.646** | 101 |
| CONFUSED | 0.863 | 0.880 | 0.871 | 100 |
| RUSHING | 0.929 | 0.948 | 0.939 | 97 |
| FRUSTRATED | 0.748 | 0.980 | 0.848 | 100 |
| STUCK | 1.000 | 0.895 | 0.944 | 95 |
| INSIGHT | 0.848 | 0.873 | 0.860 | 102 |

**Citable claim.** FLOW is the weakest class by a wide margin in every condition, and removing
problem leakage does not fix it (recall 0.460 → 0.505). Dominant confusions: FLOW→PLANNING (18),
FLOW→INSIGHT (14), CONFUSED→FRUSTRATED (11).

Mechanism, stated as interpretation rather than measurement: FLOW has no distinctive lexical marker,
while the states it loses to carry high-weight ones, so a single competing marker anywhere in an
utterance takes the argmax. Part of the deficit is label ambiguity — a fluent student checking their
work is not cleanly separable from a confused one on surface lexis — but **no inter-rater study
exists to quantify that split**, so it must be presented as a hypothesis.

*Trace:* `per_class_metrics.csv`; `metrics.json → conditions.B_problem_disjoint.per_state`,
`→ .confusion_matrix`; figure `confusion_matrix.png`.

---

## 5. Cross-generator results — the primary finding

Two analyses, both on the same 10 held-out problems. Differences use a **paired problem-clustered
bootstrap** (2,000 replicates, seed 4242): each replicate resamples the 10 problems once and scores
both arms on that same resample. Utterances are never resampled — they are not independent within a
problem.

### 5.1 Matched analysis — cite this one

Support equalised at the (problem × state) cell level: for each of the 70 cells, *k* = the smallest
count any generator has, and exactly *k* examples are drawn from each. **All 70 cells were usable,
0 dropped**, giving **339 examples per generator** (1,356 total). Support is identical across all
four generators per state *and* cell by cell — asserted in code and in tests, not assumed.

Generator × state support, identical for all four generators:

| PLANNING | FLOW | CONFUSED | RUSHING | FRUSTRATED | STUCK | INSIGHT | Total |
|---|---|---|---|---|---|---|---|
| 52 | 51 | 53 | 44 | 44 | 41 | 54 | 339 |

| Generator | n | Accuracy | 95% CI | Macro-F1 | 95% CI |
|---|---|---|---|---|---|
| llama3.1:8b | 339 | 0.8820 | [0.8387, 0.9224] | 0.8796 | [0.8337, 0.9199] |
| gemma2:9b | 339 | 0.8378 | [0.8234, 0.8537] | 0.8317 | [0.8136, 0.8457] |
| mistral:7b | 339 | 0.8142 | [0.7706, 0.8547] | 0.8104 | [0.7584, 0.8521] |
| **phi3:medium** | 339 | **0.6313** | [0.5812, 0.6824] | **0.6226** | [0.5619, 0.6777] |

| vs llama3.1:8b | Δ accuracy | 95% CI | Δ macro-F1 | 95% CI | Interval excludes 0 |
|---|---|---|---|---|---|
| gemma2:9b | −0.0442 | [−0.0888, +0.0030] | −0.0479 | [−0.1010, +0.0046] | **No — not a finding** |
| mistral:7b | −0.0678 | [−0.1032, −0.0340] | −0.0692 | [−0.1177, −0.0317] | **Yes** |
| **phi3:medium** | **−0.2507** | [−0.3141, −0.1785] | **−0.2570** | [−0.3293, −0.1803] | **Yes** |

Matched per-state F1:

| State | llama3.1:8b | mistral:7b | gemma2:9b | phi3:medium |
|---|---|---|---|---|
| PLANNING | 0.901 | 0.925 | 0.943 | 0.741 |
| FLOW | 0.714 | 0.519 | 0.608 | **0.420** |
| CONFUSED | 0.909 | 0.870 | 0.855 | 0.555 |
| RUSHING | 0.923 | 0.902 | 0.925 | 0.738 |
| FRUSTRATED | 0.878 | 0.881 | 0.846 | 0.639 |
| STUCK | 0.935 | 0.811 | 0.849 | **0.576** |
| INSIGHT | 0.897 | 0.765 | 0.796 | 0.689 |

**Citable claims.**
1. Under exactly matched support, **phi3:medium loses ~25 accuracy points** relative to llama3.1:8b
   on identical problems and identical label counts (Δ −0.2507, 95% CI [−0.3141, −0.1785]). It is
   worst on **all seven** states.
2. **mistral:7b loses ~7 points** (Δ −0.0678, 95% CI [−0.1032, −0.0340]), on both accuracy and
   macro-F1.
3. **gemma2:9b is inconclusive** — the interval crosses zero on both metrics. Report it as no
   detected difference, not as a small gap.
4. llama3.1:8b is **not a neutral reference**: it generated the original corpus and the markers were
   developed against it. Describe it as the in-domain generator, not as a control.

### 5.2 Unmatched analysis — consistent, reported for completeness

| Generator | n | Accuracy | Δ vs llama | 95% CI | Excludes 0 |
|---|---|---|---|---|---|
| llama3.1:8b | 695 | 0.8619 | — | — | — |
| gemma2:9b | 341 | 0.8387 | −0.0232 | [−0.0558, +0.0113] | No |
| mistral:7b | 340 | 0.8147 | −0.0472 | [−0.0893, −0.0108] | Yes |
| phi3:medium | 340 | 0.6294 | −0.2325 | [−0.2831, −0.1792] | Yes |

Matching moves the three non-llama arms by at most 0.002 (they were already near-fully sampled,
339 of ~340). Essentially the entire change comes from llama, subsampled 695 → 339, whose accuracy
rises +0.0201. **Matching therefore removes a sampling artefact that had been understating the
llama baseline; the unmatched gaps in §5.2 were conservative.**

*Trace:* `matched_metrics.json`, `matched_generator_comparison.csv`, `matched_support.csv`,
`matched_per_class_metrics.csv`, `matched_confusion_matrices.csv`, `matched_manifest.{json,csv}`,
figure `matched_confusion_matrix.png`; unmatched in `metrics.json → conditions."C_generator::*"`,
`→ generator_vs_llama`, `generator_comparison.csv`.

---

## 6. Limitations that must accompany any citation

1. **Entirely synthetic data.** LLM-generated think-alouds with LLM-assigned labels. No human
   annotation, no real students, no inter-rater agreement. These numbers measure agreement with a
   generator's stated intent, not with human judgement of a real cognitive state.
2. **Problem-disjointness does not remove developer exposure.** The heuristic markers were
   hand-written with all 50 problems visible. A problem-level split removes leakage between the
   evaluation set and any data-driven fitting; it cannot remove the authors' prior exposure to the
   test problems. This is the strongest caveat on §3.
3. **Ten test problems is a small cluster count.** All intervals bootstrap 10 clusters. They are
   percentile intervals with no small-sample correction. Matching (§5.1) fixes the support confound
   but not this.
4. **Narrow subject coverage in test.** The 10 test problems span 5 of the corpus's 10 subjects
   (ACT English, ACT Reading, ACT Science, Algebra, US History). Chemistry, Biology, Geometry, ACT
   Math and Python Programming are absent from test. The split was fixed before any result was seen
   and was never reselected.
5. **The corpus is 50 academic problems across 10 subjects, not 50 mathematics problems.** Only a
   minority are mathematics.
6. **No calibrated confidence, and no ECE.** The analyzer's confidence is a hand-set affine function
   of keyword-marker weights, capped at 0.97, never fitted against correctness; 73% of predictions
   sit at the ceiling. ECE is deliberately omitted rather than computed on a non-probability. Any
   downstream policy gating on this confidence gates on an uncalibrated margin.
7. **The previous-vs-new comparison is not a significance test.** The 50 problems of the previous
   condition include the 10 of the new one, so the arms are neither paired nor independent.
8. **Nothing here speaks to the deployed system** (§1).

---

## 7. Numbers that did NOT survive verification — do not cite

* **No ECE / calibration error for the classifier.** Not computed, by design. If a calibration
  number is needed, it must come from new work. The repository's `calibration.py` measures the
  *student's* metacognitive calibration and is a different quantity entirely.
* **No gemma2:9b gap.** Inconclusive in both the matched and unmatched analyses.
* **No claim that the earlier 0.8457 was inflated.** The comparison shows the opposite direction.
* **No human-agreement or inter-rater number** for any of the seven states.
* **No accuracy figure for the 13-code reasoning-move detector**, the scaffold policy, or the JITAI
  policy.
* **The adversarial FLOW hard-negative accuracy (0.4722, n=36)** is from a deliberately constructed
  stress set, not a sample of the corpus. Cite only as an adversarial bound, never pooled with the
  headline numbers.

---

## 8. Reproducibility

All paths are relative to the repository root. Every default matches the values used for this run.

```bash
cd eval
python3.11 problem_disjoint_eval.py       # split, all conditions, bootstraps, artefacts
python3.11 matched_generator_eval.py      # matched cross-generator robustness check
cd .. && python3.11 -m pytest tests/test_problem_disjoint_split.py -q
```

**Determinism.** The heuristic route is pure string matching — no RNG, no network call, no model
weights — and took 100% of predictions in every condition. The only randomness is the problem split
(seed 20260826), the matched draw (seed 20260826), and the bootstrap (seed 4242). Verified
empirically: the pipeline was run twice end to end and the two `metrics.json` files are identical in
every field except the git-status provenance string.

**Frozen classifier.** SHA-256 recorded before any test result was computed:

| File | SHA-256 |
|---|---|
| `eval/metacognition/analyzer.py` | `0edb199802f9eebc49fbbfd402022c5eb17fbf743929e57d9027c82526ff96a2` |
| `eval/metacognition/behavioral.py` | `15ec47fd7438d8ce27ec60aea8855eadd8fee79d0dbb9fad25ce460846a0a25f` |
| `eval/metacognition/transfer.py` | `1ee8bcbd101f438d556516e6add53e6cef8accfb7db3fca3ec1e343fc11ef165` |

No rule, prompt, threshold or mapping was changed at any point. `git diff` against `ef5b7d6` touches
no tracked file. Configuration: `use_llm=False`, `HEURISTIC_TRUST=0.65`, `ARIA_META_MODEL` unset, no
audio features, no behavioural features.

**Environment.** Python 3.11.15 (CPython), macOS arm64, numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.9.0, matplotlib 3.11.0, pytest 9.1.1. Ollama installed but never called.
Corpus generator models (not invoked in this run): llama3.1:8b, mistral:7b, gemma2:9b, phi3:medium.

**Tests.** 36 tests in `tests/test_problem_disjoint_split.py` (leakage, split invariants, matching
balance, determinism, artefact agreement, checksum integrity) plus the full existing suite —
**all pass**. Checksums for every artefact are in `CHECKSUMS.sha256`.

**Full detail, including failure analysis and the complete methodology:** `REPORT.md` in this
directory.
