# Problem-disjoint evaluation of the ARIA offline seven-state cognitive-state classifier

**Run date** 2026-08-26 · **Frozen at commit** `ef5b7d6c50bd05c4b6797e0eedf220c8f81b7183` (branch `main`)
**Every number below is copied from `metrics.json` in this directory.** The trace path for each
table is given beneath it.

---

## 0. What is and is not evaluated here

The ARIA repository contains four separate components that are easy to conflate. Exactly one is
measured in this report.

| Component | Source | Status in this report |
|---|---|---|
| **Offline seven-state `CognitiveStateAnalyzer`** | `eval/metacognition/analyzer.py` | **This is what is evaluated.** Seven states: PLANNING, FLOW, CONFUSED, RUSHING, FRUSTRATED, STUCK, INSIGHT. Run offline, `use_llm=False`, heuristic route only. |
| **Deployed 13-code rule-based reasoning-move detector** | `agent/reasoning_moves.py` (`MOVE_DEFINITIONS`, 13 codes) | **Not evaluated.** No human-labelled 13-code corpus exists, so there is no ground truth to score it against. Nothing in this report speaks to its accuracy. |
| **Live hand-written scaffold policy** | `agent/nudge.py`, `agent/intervention_pipeline.py` | **Not evaluated and not modified.** This is what actually decides what a student sees. |
| **Offline JITAI policy** | `eval/metacognition/jitai.py` | **Not part of this experiment.** It was not run, not tuned, and not wired into the product. |

The seven-state analyzer is an *offline research instrument*. It is not the detector that runs in
the product. A result here is a statement about that instrument on synthetic think-aloud text, and
about nothing else.

---

## 1. Headline result

**Removing problem-level leakage did not reduce measured performance.** The previously reported
0.8457 accuracy / 0.8373 macro-F1 is *not* shown to be inflated by problem overlap. The real
generalization failure in this system is across **generators**, not across problems.

| | Accuracy | 95% CI | Macro-F1 | 95% CI | n | problems |
|---|---|---|---|---|---|---|
| **A. Seen-problem / utterance-disjoint** (previous result) | 0.8457 | [0.8042, 0.8838] | 0.8373 | [0.7926, 0.8747] | 350 | 50 |
| **B. Problem-disjoint** (new) | **0.8619** | [0.8317, 0.8881] | **0.8567** | [0.8271, 0.8836] | 695 | 10 |
| Difference (B − A) | **+0.0162** | — | **+0.0194** | — | | |

Both intervals overlap heavily and the point estimate moves *up*, not down. Per the pre-registered
rule for this analysis, the earlier number is therefore **not** described as inflated.

Trace: `metrics.json → conditions.A_utterance_level`, `conditions.B_problem_disjoint`,
`utterance_level_vs_problem_disjoint`. Also `comparison_utterance_vs_problem_disjoint.csv`.

### The controlled version of the same comparison

A and B differ in two ways at once: seen vs unseen problems, *and* a different utterance pool
(A is the 350-row `test.jsonl`; B is every llama3.1:8b utterance for the held-out problems). The
cleaner contrast holds the pool fixed and varies only the problem group:

| llama3.1:8b pool, same classifier | Accuracy | 95% CI | Macro-F1 | n | problems |
|---|---|---|---|---|---|
| 30 **train** problems (seen) | 0.8477 | [0.8328, 0.8617] | 0.8414 | 2114 | 30 |
| 10 **validation** problems (unseen) | 0.8539 | [0.8336, 0.8764] | 0.8426 | 698 | 10 |
| 10 **test** problems (unseen) | 0.8619 | [0.8317, 0.8881] | 0.8567 | 695 | 10 |
| Difference (test − train) | **+0.0142** | — | **+0.0153** | | |

Unseen problems score *slightly higher* than seen ones. This is the expected result for a
classifier whose decision rule is a fixed bank of hand-written lexical markers: it has no capacity
to memorise a problem, so there is no per-problem quantity for it to overfit. The seen/unseen gap
being ≈ 0 is evidence about the *classifier family*, not a clean bill of health for the corpus —
see §7.

Trace: `metrics.json → seen_vs_unseen_problems_same_pool`, `conditions.B_train_problems`,
`conditions.B_val_problems`.

---

## 2. The split

50 unique problems, partitioned **at the problem level** into 30 train / 10 validation / 10 test.
Every utterance derived from a problem — all seven cognitive states, all three ADHD personas, all
subject-confidence levels, all paraphrases, all four generators — stays with its problem.

* **Seed** 20260826, recorded in `split_manifest.json → seed`.
* **Procedure** `seeded_shuffle_30_10_10`. The plain seeded shuffle was used because it left every
  one of the seven states present in validation and test *for every generator*. The deterministic
  group-stratified fallback (`_stratified_partition`, subject-blocked round-robin) is implemented
  and documented in `eval/build_problem_disjoint_split.py` but **was not needed and was not used**.
  Test problems were never selected by looking at performance.
* **Problem IDs** `P01`–`P50`, assigned in ascending SHA-256 order of the normalized problem text
  (NFKC → lowercase → collapse whitespace → strip). The ID is a function of the problem text alone,
  so it is stable across seeds, file order, and reruns.
* **Test problems** P03, P05, P10, P14, P20, P31, P33, P35, P39, P41.
* **Validation problems** P07, P08, P09, P12, P26, P34, P45, P46, P47, P48.

### Pool

8,750 utterances, verified mutually disjoint at the utterance level:

| Generator | Files | n |
|---|---|---|
| llama3.1:8b | `dataset.jsonl` | 3507 |
| mistral:7b | `holdout_mistral.jsonl` + `augment_mistral.jsonl` | 1746 |
| gemma2:9b | `holdout_gemma2.jsonl` + `augment_gemma2.jsonl` | 1750 |
| phi3:medium | `holdout_phi3.jsonl` + `augment_phi3.jsonl` | 1747 |

**Excluded, and why** (recorded in `split_manifest.json → excluded_from_pool`): `train/val/test.jsonl`
and `dataset_mixed.jsonl` / `mixed_*.jsonl` are resamples or subsets of the above and would
double-count; `test.jsonl` is instead scored separately as condition A. `flow_hard_negatives.jsonl`
(160 adversarially written FLOW look-alikes over 28 problems) is a targeted stress set rather than a
sample of the corpus distribution, so folding it in would bias accuracy downward by construction; it
is scored separately in §6 and excluded from every headline number.

### Split distributions

| Split | Problems | Examples | PLAN | FLOW | CONF | RUSH | FRUS | STUCK | INSI |
|---|---|---|---|---|---|---|---|---|---|
| train | 30 | 5332 | 731 | 753 | 762 | 782 | 791 | 770 | 743 |
| val | 10 | 1702 | 264 | 244 | 226 | 238 | 228 | 260 | 242 |
| test | 10 | 1716 | 256 | 254 | 261 | 229 | 232 | 220 | 264 |

| Split | llama3.1:8b | mistral:7b | gemma2:9b | phi3:medium |
|---|---|---|---|---|
| train | 2114 | 1072 | 1074 | 1072 |
| val | 698 | 334 | 335 | 335 |
| test | 695 | 340 | 341 | 340 |

All seven states are present in every split, and in every (split × generator) cell.

Trace: `split_manifest.json → distributions`, `split_manifest.csv`.

### Leakage check

Every pairwise overlap is **0**, asserted at build time and re-asserted by the test suite:

| Quantity | train∩val | train∩test | val∩test |
|---|---|---|---|
| problem ID | 0 | 0 | 0 |
| normalized-problem SHA-256 | 0 | 0 | 0 |
| normalized problem text | 0 | 0 | 0 |
| utterance text | 0 | 0 | 0 |

Independently verified: across all 8,750 pooled utterances, **no utterance text occurs under more
than one problem**, so partitioning by problem partitions the utterances exactly.

Trace: `split_manifest.json → leakage_check`; `tests/test_problem_disjoint_split.py`.

---

## 3. Per-state metrics

Condition A (seen-problem, n=350) vs condition B (problem-disjoint, n=695), llama3.1:8b:

| State | A prec | A rec | A F1 | A sup | B prec | B rec | B F1 | B sup |
|---|---|---|---|---|---|---|---|---|
| PLANNING | 0.793 | 0.920 | 0.852 | 50 | 0.828 | 0.960 | 0.889 | 100 |
| **FLOW** | 0.852 | **0.460** | **0.597** | 50 | 0.895 | **0.505** | **0.646** | 101 |
| CONFUSED | 0.840 | 0.840 | 0.840 | 50 | 0.863 | 0.880 | 0.871 | 100 |
| RUSHING | 0.922 | 0.940 | 0.931 | 50 | 0.929 | 0.948 | 0.939 | 97 |
| FRUSTRATED | 0.769 | 1.000 | 0.870 | 50 | 0.748 | 0.980 | 0.848 | 100 |
| STUCK | 1.000 | 0.820 | 0.901 | 50 | 1.000 | 0.895 | 0.944 | 95 |
| INSIGHT | 0.810 | 0.940 | 0.870 | 50 | 0.848 | 0.873 | 0.860 | 102 |

FLOW is the weakest class in both conditions by a wide margin, and its recall (0.46 → 0.505) is the
single largest contributor to the macro-F1 shortfall. Removing problem leakage does not fix it.

Trace: `per_class_metrics.csv`; `metrics.json → conditions.*.per_state`.

### Confusion matrices

Seven-class counts for every condition are in `confusion_matrices.csv`
(`metrics.json → conditions.*.confusion_matrix`). The publication figure comparing A and B,
row-normalised with raw counts overlaid, is `confusion_matrix.png`.

The dominant off-diagonal cells in condition B are FLOW→PLANNING (18), FLOW→INSIGHT (14),
CONFUSED→FRUSTRATED (11), STUCK→FRUSTRATED (8), FLOW→CONFUSED (7).

### Majority-class baseline

The majority class is chosen on the **training** split only (never on the evaluation set):

| Condition | Majority class | Baseline accuracy | Classifier accuracy |
|---|---|---|---|
| A. utterance-level | RUSHING | 0.1429 | 0.8457 |
| B. problem-disjoint | RUSHING | 0.1396 | 0.8619 |
| C. all generators pooled | FRUSTRATED | 0.1352 | 0.8019 |

The corpus is near-uniform over seven classes, so the baseline sits close to 1/7 ≈ 0.1429
everywhere. The classifier is far above it in every condition.

Trace: `metrics.json → conditions.*.majority_baseline`.

---

## 4. Generator-stratified results

All four arms are scored on **the same 10 problem-disjoint test problems**, verified by
`tests/test_problem_disjoint_split.py::test_metrics_use_the_manifest_split`. Differences against the
llama baseline use a **paired** cluster bootstrap: each replicate resamples the 10 problems once and
scores both arms on that same resample.

| Generator | n | problems | Accuracy | 95% CI | Macro-F1 | 95% CI | Δacc vs llama | 95% CI of Δ |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | 695 | 10 | 0.8619 | [0.8317, 0.8881] | 0.8567 | [0.8271, 0.8836] | — (baseline) | — |
| gemma2:9b | 341 | 10 | 0.8387 | [0.8243, 0.8546] | 0.8326 | [0.8150, 0.8466] | −0.0232 | [−0.0558, +0.0113] |
| mistral:7b | 340 | 10 | 0.8147 | [0.7716, 0.8555] | 0.8107 | [0.7594, 0.8524] | −0.0472 | [−0.0893, −0.0108] |
| **phi3:medium** | 340 | 10 | **0.6294** | [0.5808, 0.6760] | **0.6206** | [0.5613, 0.6714] | **−0.2325** | [−0.2831, −0.1792] |

The gemma2 interval includes zero; the mistral and phi3 intervals do not. **phi3:medium is the
finding**: a 23-point accuracy drop on identical problems, five times the mistral gap, with an
interval nowhere near zero.

Note the corpus was generated by llama3.1:8b and the classifier's markers were developed against it,
so llama3.1:8b is a home-field arm, not a neutral reference.

Per-state F1 across generators, showing the drop is broad rather than one bad class:

| State | llama3.1:8b | mistral:7b | gemma2:9b | phi3:medium |
|---|---|---|---|---|
| PLANNING | 0.89 | 0.93 | 0.94 | 0.74 |
| FLOW | 0.65 | 0.52 | 0.61 | **0.42** |
| CONFUSED | 0.87 | 0.87 | 0.86 | 0.55 |
| RUSHING | 0.94 | 0.90 | 0.93 | 0.74 |
| FRUSTRATED | 0.85 | 0.88 | 0.85 | 0.64 |
| STUCK | 0.94 | 0.81 | 0.85 | **0.57** |
| INSIGHT | 0.86 | 0.77 | 0.80 | 0.69 |

phi3 loses on six of seven states. Its worst are STUCK (F1 0.57, recall 0.41) and FLOW (F1 0.42,
recall 0.33).

Trace: `generator_comparison.csv`; `metrics.json → conditions."C_generator::*"`,
`generator_vs_llama`.

---

## 5. Calibration — ECE is deliberately omitted

**No ECE is reported, because the analyzer does not produce a probability.** Its `confidence` is

```
confidence = clip(0.45 + 0.09 · evidence_score + 0.12 · (evidence_score − runner_up_score), 0, 0.97)
```

a hand-set affine function of keyword-marker weights (`analyzer.py`, `per_state_confidence`). It is
not normalised over the seven states, cannot fall below 0.45 whenever any marker fires, saturates at
0.97, and was never fitted against observed correctness. Binning it and calling the residual an
expected calibration error would attach a calibration number to a quantity that is not a
probability, so it is omitted rather than guessed.

For the avoidance of doubt: `eval/metacognition/calibration.py` measures something else entirely —
the **student's** metacognitive calibration (`confidence_before` vs `correct`). It is not a
classifier confidence calibration and is not reported here.

Descriptive only — accuracy by confidence bin on the problem-disjoint test set, all generators.
**This is not a reliability diagram and not an ECE:**

| Confidence bin | n | Mean confidence | Accuracy |
|---|---|---|---|
| [0.50, 0.60) | 1 | 0.585 | 0.000 |
| [0.60, 0.70) | 410 | 0.604 | 0.493 |
| [0.70, 0.80) | 20 | 0.758 | 0.300 |
| [0.80, 0.90) | 30 | 0.847 | 0.500 |
| [0.90, 1.01) | 1255 | 0.969 | 0.919 |

The distribution is bimodal — 73% of predictions sit in the top bin at the 0.97 ceiling — which is
itself the reason the score cannot be read as a probability.

Trace: `metrics.json → calibration`.

---

## 6. Supplementary: adversarial FLOW hard negatives

Excluded from every headline number above. Restricted to the 10 test problems, the adversarial FLOW
hard-negative set gives n=36 over 5 problems, accuracy **0.4722**, macro-F1 0.4690. It is
adversarially constructed rather than sampled, so it must not be pooled with the other conditions; it
is reported because it bounds how far FLOW detection degrades under deliberate pressure.

Trace: `metrics.json → conditions.S_flow_hard_negatives`.

---

## 7. Failure cases

Selected by fixed rules, not by hand (rules recorded in `failure_cases.json → selection_rules`).
Each selector takes the most-confident qualifying error, preferring a problem not already reported —
93 of the 340 pooled test errors (27.4%) sit at the 0.97 confidence ceiling, so without that
preference the tie-break would send every case to the same problem. Full text and marker scores in `failure_cases.json`.

### 7.1 FLOW false negative — P03, mistral:7b

* **Problem** "Results contradict a hypothesis. What is the most reasonable next step?" (ACT Science / hypothesis testing, persona `inattentive`)
* **Ground truth** FLOW · **Predicted** CONFUSED · **Confidence** 0.97 (ceiling)
* **Marker scores** CONFUSED 4.0, INSIGHT 2.5, FLOW 1.5, RUSHING 1.2 · **matched** `wait, what, circular repetition`
* **Think-aloud** *"[Excited mutter] Alright, let's do this! Results don't make sense with my initial idea, but I know I can figure it out. Just gotta stay focused. [Whispers to himself] Hypothesis vs. results... Hmm, maybe I messed up the calculations or missed something obvious. Let's go back and check each step again. Alright, that seems good. But wait, what if it's something else? Okay, let me look at the data from another angle. …"*
* **Why it fails.** FLOW has no lexical marker of its own. Sustained productive work is narrated in
  ordinary procedural language, while the states it loses to carry high-weight distinctive markers
  (PLANNING's *"first I'll"*, INSIGHT's *"oh wait"*, CONFUSED's *"wait, what"*). FLOW's evidence is
  diffuse and low-weight, so a single competing marker anywhere in a long utterance takes the argmax
  away from it — here `wait, what` scores CONFUSED at 4.0 against FLOW's 1.5. Note this utterance
  *does* contain a genuine self-check, so part of the FLOW gap is label ambiguity rather than pure
  classifier error: a fluent student checking their work is not cleanly separable from a confused one
  on surface lexis alone. FLOW is missed on 50 of 101 true-FLOW llama test utterances.

### 7.2 phi3-specific failure — P20, phi3:medium

* **Problem** "Which revision best fixes the run-on: 'It was raining we stayed inside.'" (ACT English / sentence structure, persona `hyperactive`)
* **Ground truth** STUCK · **Predicted** CONFUSED · **Confidence** 0.97 (ceiling)
* **Marker scores** CONFUSED 7.0, STUCK 3.3, FRUSTRATED 2.5, INSIGHT 2.5 · **matched** `confusing, is confusing, circular repetition`
* **Think-aloud** *"Man, this run-on sentence thing is confusing. I don't know where to even start. Okay… it was raining - that's the first part. And we stayed inside. Uh... maybe adding a period? But then what about 'we stayed?' Wait, could a comma work too? Or... Nope, can't remember clearly. This is so frustrating. I think I got it! A period after raini…"*
* **Why it fails.** phi3:medium narrates in long, formal, fully-punctuated prose that walks the whole
  solution path in one utterance. That style trips markers belonging to several states at once —
  here CONFUSED, STUCK, FRUSTRATED and INSIGHT all fire — so the winner is decided by which marker
  bank carries more weight rather than by the student's actual state. The marker banks were audited
  for generator-agnosticism on *token identity* (`eval/mine_signals.py`), which does not control for
  the **utterance length and register** phi3 changes: more sentences means more chances to fire an
  off-state marker, and the additive scoring has **no length normalisation**. This is the mechanism
  behind the 23-point phi3 gap in §4.

### 7.3 Highest-confidence error — P05, gemma2:9b

* **Problem** "A passage describes a town's decline after a factory closes. What is the main idea?" (ACT Reading / main idea, persona `combined`)
* **Ground truth** FLOW · **Predicted** INSIGHT · **Confidence** 0.97 (ceiling)
* **Marker scores** INSIGHT 4.0, CONFUSED 2.0, PLANNING 1.5, RUSHING 1.5 · **matched** `got it, makes sense`
* **Think-aloud** *"Okay, so it's about a town and how things went downhill after the factory shut down. I bet it's something about how important factories are to towns or something. Yeah, probably about how places depend on factories for jobs. Hmm, but wait...the passage is mostly about people losing their homes and businesses too. So maybe it's more about…"*
* **Why it fails.** Not a near-tie the analyzer had no basis to call — the evidence actively points
  the wrong way (INSIGHT 4.0 vs FLOW 0.0). This is the **structurally important** case: the reported
  confidence rises with the *winning margin* and contains no term for whether the winning evidence is
  *correct*, so a maximally-confident wrong answer is indistinguishable from a maximally-confident
  right one to any downstream consumer. Any policy that gates on `confidence > τ` inherits this
  failure silently. It is also the concrete reason no ECE is reported (§5).

### 7.4 Most frequent non-FLOW confusion — P14, phi3:medium

Included so the failure set is not dominated by the one class already known to be weakest.

* **Problem** "What was a central economic difference between the North and South before the war?" (US History / civil war, persona `inattentive`)
* **Ground truth** CONFUSED · **Predicted** FRUSTRATED · **Confidence** 0.97 (ceiling)
* **Marker scores** FRUSTRATED 4.2, CONFUSED 2.0, RUSHING 1.2, FLOW 0.0 · **matched** `why is, finally, so hard`
* **Think-aloud** *"Okay, I gotta remember something about the North and South having different ways of making money before all that war stuff. But man, what was it again? The textbook's like a maze. Wait... wasn't it like the South did cotton or something 'cause they had those big farms? Nope, wait, wrong topic. Okay, maybe they were into... manufacturing?…"*
* **Why it fails.** CONFUSED→FRUSTRATED is the most frequent non-FLOW confusion in the pooled
  problem-disjoint test set. The two states are adjacent in the ARIA state model and share most of
  their vocabulary of difficulty, but they call for different scaffolds — a content clarification
  versus an affect-protective acknowledgement — so this confusion carries the clearest downstream
  cost of any in the matrix.

---

## 7A. Matched cross-generator robustness check

The §4 arms are equal in *problems* but not in *support*: llama3.1:8b contributes 695 utterances
against roughly 340 for each other generator, and the per-(problem × state) mix differs between
arms. A gap measured that way is partly a gap in what was sampled. This check removes that confound.

**Design.** The matching unit is the (held-out problem × cognitive state) cell. For each of the 70
cells, *k* = the smallest count any of the four generators has in that cell, and exactly *k*
utterances are drawn from each generator. Each cell/generator draw is seeded from
`(20260826, problem_id, state, generator)` over a `(source_file, source_line)`-sorted pool, so a
change in one cell cannot perturb the draws in any other.

**All 70 cells were usable — no cell was dropped**, giving **339 examples per generator** (1,356
total). Every generator's support is identical not only in total and per state, but cell by cell;
this is asserted, not assumed (`matched_generator_eval.py` raises on mismatch, and
`tests/test_problem_disjoint_split.py::MatchedGeneratorTest` re-checks it).

### Generator × state support (identical for all four generators)

| Generator | PLANNING | FLOW | CONFUSED | RUSHING | FRUSTRATED | STUCK | INSIGHT | Total |
|---|---|---|---|---|---|---|---|---|
| llama3.1:8b | 52 | 51 | 53 | 44 | 44 | 41 | 54 | 339 |
| mistral:7b | 52 | 51 | 53 | 44 | 44 | 41 | 54 | 339 |
| gemma2:9b | 52 | 51 | 53 | 44 | 44 | 41 | 54 | 339 |
| phi3:medium | 52 | 51 | 53 | 44 | 44 | 41 | 54 | 339 |

Trace: `matched_support.csv`; `matched_metrics.json → support`.

### Matched results

Differences use a **paired problem-clustered bootstrap** (2000 replicates, seed 4242): each
replicate resamples the 10 held-out problems once and scores both arms on that same resample.

| Generator | n | Accuracy | 95% CI | Macro-F1 | 95% CI |
|---|---|---|---|---|---|
| llama3.1:8b | 339 | 0.8820 | [0.8387, 0.9224] | 0.8796 | [0.8337, 0.9199] |
| gemma2:9b | 339 | 0.8378 | [0.8234, 0.8537] | 0.8317 | [0.8136, 0.8457] |
| mistral:7b | 339 | 0.8142 | [0.7706, 0.8547] | 0.8104 | [0.7584, 0.8521] |
| **phi3:medium** | 339 | **0.6313** | [0.5812, 0.6824] | **0.6226** | [0.5619, 0.6777] |

| Generator vs llama | Δ accuracy | 95% CI (paired) | excludes 0 | Δ macro-F1 | 95% CI (paired) | excludes 0 |
|---|---|---|---|---|---|---|
| gemma2:9b | −0.0442 | [−0.0888, +0.0030] | **no** | −0.0479 | [−0.1010, +0.0046] | **no** |
| mistral:7b | −0.0678 | [−0.1032, −0.0340] | **yes** | −0.0692 | [−0.1177, −0.0317] | **yes** |
| **phi3:medium** | **−0.2507** | [−0.3141, −0.1785] | **yes** | **−0.2570** | [−0.3293, −0.1803] | **yes** |

### Per-state F1 under matched support

| State | llama3.1:8b | mistral:7b | gemma2:9b | phi3:medium |
|---|---|---|---|---|
| PLANNING | 0.901 | 0.925 | 0.943 | 0.741 |
| FLOW | 0.714 | 0.519 | 0.608 | **0.420** |
| CONFUSED | 0.909 | 0.870 | 0.855 | 0.555 |
| RUSHING | 0.923 | 0.902 | 0.925 | 0.738 |
| FRUSTRATED | 0.878 | 0.881 | 0.846 | 0.639 |
| STUCK | 0.935 | 0.811 | 0.849 | **0.576** |
| INSIGHT | 0.897 | 0.765 | 0.796 | 0.689 |

phi3:medium is worst on all seven states. Its weakest remain STUCK (F1 0.576, recall 0.415) and FLOW
(F1 0.420, recall 0.333).

### Do the findings survive?

**Yes — both hold, and both gaps widen.**

| Generator | Unmatched Δacc (§4) | Matched Δacc | Verdict |
|---|---|---|---|
| gemma2:9b | −0.0232 (CI includes 0) | −0.0442 (CI includes 0) | **Inconclusive in both.** Larger point estimate, but the interval still crosses zero (upper bound +0.0030). Not a finding. |
| mistral:7b | −0.0472 (CI excludes 0) | −0.0678 (CI excludes 0) | **Holds and strengthens**, now on macro-F1 as well as accuracy. |
| phi3:medium | −0.2325 (CI excludes 0) | −0.2507 (CI excludes 0) | **Holds and strengthens.** ~25 points, interval nowhere near zero. |

Why the gaps grew is visible in the per-arm shift:

| Generator | Unmatched acc | Matched acc | Change | n: unmatched → matched |
|---|---|---|---|---|
| llama3.1:8b | 0.8619 | 0.8820 | **+0.0201** | 695 → 339 |
| mistral:7b | 0.8147 | 0.8142 | −0.0005 | 340 → 339 |
| gemma2:9b | 0.8387 | 0.8378 | −0.0009 | 341 → 339 |
| phi3:medium | 0.6294 | 0.6313 | +0.0019 | 340 → 339 |

The three non-llama arms move by at most 0.002 — they were already near-fully sampled (339 of ~340).
Essentially all of the change comes from llama, whose 695 utterances are subsampled to 339. So
matching does not rescue the weaker generators; it removes a sampling advantage that had been
*understating* the llama baseline. The gaps in §4 were, if anything, conservative.

Trace: `matched_metrics.json`, `matched_generator_comparison.csv`, `matched_per_class_metrics.csv`,
`matched_confusion_matrices.csv`, `matched_confusion_matrix.png`, `matched_manifest.{json,csv}`.
The classifier was not modified: `matched_generator_eval.py` reports the same
`analyzer.py` SHA-256 as every other run in this directory.

---

## 8. Reproducibility

### Exact commands

All paths are relative to the repository root.

```bash
cd eval

# 1. Build the split, assert no leakage, write the manifest (also run by step 2).
python3.11 build_problem_disjoint_split.py \
    --out-dir data/eval/problem_disjoint_20260826

# 2. Full evaluation: all conditions, 2000-replicate cluster bootstraps, all artefacts.
python3.11 problem_disjoint_eval.py \
    --seed 20260826 \
    --bootstrap-reps 2000 \
    --out-dir data/eval/problem_disjoint_20260826

# 3. Matched cross-generator robustness check (§11): equal support per generator.
python3.11 matched_generator_eval.py \
    --seed 20260826 \
    --match-seed 20260826 \
    --bootstrap-reps 2000 \
    --out-dir data/eval/problem_disjoint_20260826

# 4. Leakage, split and matching assertions (36 tests).
cd .. && python3.11 -m pytest tests/test_problem_disjoint_split.py -q
```

Every default (`--seed 20260826`, `--match-seed 20260826`, `--bootstrap-reps 2000`, the output
directory) matches the values used for this run, so bare `python3.11 eval/problem_disjoint_eval.py`
followed by `python3.11 eval/matched_generator_eval.py` from the repository root reproduces it
exactly. Total runtime ≈ 45 s.

### Determinism

The heuristic route is pure string matching: no RNG, no network call, no model weights, and
100% of predictions in every condition took the `heuristic` route (`metrics.json →
conditions.*.routing`). The Ollama fallback was never consulted. The only randomness in the pipeline
is the split shuffle (seed 20260826) and the bootstrap resampling (seed 4242), both recorded.

Verified empirically: the pipeline was executed twice end to end, and the two `metrics.json` files
are identical in every field except `frozen.git_status_porcelain`, which changes only because the
first run created this results directory.

### Frozen classifier

Hashed **before** any test result was computed:

| File | SHA-256 |
|---|---|
| `eval/metacognition/analyzer.py` | `0edb199802f9eebc49fbbfd402022c5eb17fbf743929e57d9027c82526ff96a2` |
| `eval/metacognition/behavioral.py` | `15ec47fd7438d8ce27ec60aea8855eadd8fee79d0dbb9fad25ce460846a0a25f` |
| `eval/metacognition/transfer.py` | `1ee8bcbd101f438d556516e6add53e6cef8accfb7db3fca3ec1e343fc11ef165` |

Configuration: `use_llm=False`, `HEURISTIC_TRUST=0.65`, `ARIA_META_MODEL` unset, no audio features,
no behavioural features, empty `aria_previous_prompt`. **No rule, prompt, threshold or mapping was
changed at any point during this work.** `git diff` against `ef5b7d6` touches no tracked file.

### Versions

| | |
|---|---|
| Python | 3.11.15 (CPython) |
| Platform | macOS-26.4.1-arm64-arm-64bit |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.0 |
| ollama | installed, **never called** |
| pytest | 9.1.1 |

Generator models (used to *produce* the corpus previously, not invoked in this run): `llama3.1:8b`,
`mistral:7b`, `gemma2:9b`, `phi3:medium`. LLM fallback model, not invoked: `llama3.2:3b`.

### Checksums

`CHECKSUMS.sha256` in this directory covers every artefact here plus the three frozen classifier
sources and the three evaluation scripts. It is regenerated by whichever evaluation script runs
last, so it always reflects the full directory.

### Source files and artefacts used

**Read:** `eval/data/synthetic_thinkaloud/{dataset,test,holdout_mistral,holdout_gemma2,holdout_phi3,augment_mistral,augment_gemma2,augment_phi3,flow_hard_negatives}.jsonl`,
`eval/metacognition/{analyzer,behavioral,transfer}.py`.
**Written (all new):** `eval/build_problem_disjoint_split.py`, `eval/problem_disjoint_eval.py`,
`eval/matched_generator_eval.py`, `tests/test_problem_disjoint_split.py`, and this directory.
**No pre-existing result file was modified or overwritten.**

---

## 9. Methodological limitations

1. **The corpus is entirely synthetic.** These are LLM-generated think-alouds with LLM-assigned
   state labels, not transcripts of real students. There is no human-labelled ground truth anywhere
   in this evaluation. Everything here measures agreement with a generator's intent, not with human
   judgement of a real cognitive state.
2. **The "50 mathematical problems" are 50 academic problems across 10 subjects** — Algebra,
   Geometry, ACT Math, ACT Science, ACT Reading, ACT English, Chemistry, Biology, US History and
   Python Programming. Only a minority are mathematics.
3. **Ten test problems is a small cluster count.** All CIs bootstrap 10 clusters, which is few;
   the intervals are correspondingly wide and are percentile intervals with no small-sample
   correction. The 10 test problems happen to span only 5 of the 10 subjects (ACT English, ACT
   Reading, ACT Science, Algebra, US History) — Chemistry, Biology, Geometry, ACT Math and Python
   Programming are absent from test. The split was fixed before any result was seen and was not
   reselected, but subject coverage is a real limitation on generalization claims.
4. **Problem-disjointness does not remove developer exposure.** The heuristic markers were
   hand-written by developers who had the *entire* 50-problem corpus visible. A problem-level split
   removes leakage between the evaluation set and any data-driven fitting; it cannot remove the
   author's prior exposure to the test problems. This is the single most important caveat on the
   §1 headline: the near-zero seen/unseen gap is consistent with "a fixed marker bank cannot
   memorise problems", but it is *not* proof that no test-problem information reached the rules.
5. **The A vs B comparison is not a significance test.** A's 50 problems include B's 10, so the two
   arms are neither paired nor independent; the reported bounds combine two separate bootstraps and
   are indicative only. The controlled train-vs-test contrast in §1 is the sounder comparison.
6. **llama3.1:8b is not a neutral baseline.** It generated the original corpus and the markers were
   developed against it, so generator gaps are measured from a home-field reference.
7. **The FLOW deficit may be partly label noise.** As §7.1 shows, "fluent student checking their
   work" and "confused student" are not cleanly separable on surface lexis, and some FLOW labels are
   arguable. No inter-rater study exists to quantify how much of the FLOW gap is classifier error
   versus label ambiguity.
8. **Condition B draws on utterances that appear in `train.jsonl`.** This is not leakage for a
   classifier with no fitted parameters, but a future trained model evaluated on this split must
   restrict its training pool by problem, not merely reuse this manifest's problem assignment.
9. **No ECE, and no calibrated confidence at all** (§5). Any downstream policy gating on this
   confidence is gating on an uncalibrated margin.
10. **These results say nothing about the deployed system.** The 13-code detector, the live scaffold
    policy, and the JITAI policy remain unevaluated (§0).
11. **The matched analysis (§7A) trades power for balance.** Equalising support costs the llama arm
    just over half its examples (695 → 339), which widens its interval. The gemma2 comparison is
    inconclusive in both the unmatched and matched analyses and must not be reported as a gap in
    either direction. The matched sample is still clustered in the same 10 problems, so matching
    fixes the support confound but not the small cluster count of limitation 3.

---

## 10. Artefact index

| File | Contents |
|---|---|
| `split_manifest.json` | Full split: seed, procedure, per-problem ID / hash / split / generator / state counts, distributions, leakage report |
| `split_manifest.csv` | Same, one row per problem |
| `metrics.json` | Every metric in this report, plus the freeze record and environment |
| `per_class_metrics.csv` | Per-condition, per-state precision / recall / F1 / support |
| `confusion_matrices.csv` | Seven-class confusion matrix for every condition |
| `comparison_utterance_vs_problem_disjoint.csv` | Condition A vs B vs train/val problems |
| `generator_comparison.csv` | Per-generator metrics and paired deltas vs llama |
| `confusion_matrix.png` | Publication figure: A vs B, row-normalised with counts |
| `failure_cases.json` | Four failure cases with full text, marker scores, selection rules |
| `matched_metrics.json` | §7A matched cross-generator metrics, matching audit, freeze record |
| `matched_manifest.json` / `.csv` | Exactly which 1,356 examples were drawn (IDs and labels only, no utterance text) |
| `matched_support.csv` | Generator × state support counts for the matched sample |
| `matched_per_class_metrics.csv` | Matched per-generator, per-state precision / recall / F1 / support |
| `matched_generator_comparison.csv` | Matched per-generator metrics, paired deltas vs llama, matched-vs-unmatched shift |
| `matched_confusion_matrices.csv` | Matched seven-class confusion matrix per generator |
| `matched_confusion_matrix.png` | Publication figure: four matched generator panels |
| `CHECKSUMS.sha256` | SHA-256 of every artefact, source and corpus file |
| `REPORT.md` | This document |
