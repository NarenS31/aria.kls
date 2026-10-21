# Affect paragraph (writing only; Step 3 of aria-pace-v3)

Status: text for the paper / site. **No new analysis was run for it.** Every number is read from the already-published, locked 2012 result (`aria_pace/results/test_results_2012.json`, produced once under `TEST_LOCK_2012`); nothing was re-evaluated.

## Paragraph

**Detector-predicted affect adds little decision-relevant information here.** The 2012–13 ASSISTments release carries four numeric columns, `Average_confidence(FRUSTRATED)`, `(CONFUSED)`, `(CONCENTRATING)` and `(BORED)`, which are per-row outputs of ASSISTments' own sensor-free affect detectors (`loader2012.py:15`, read raw at `loader2012.py:48`). They are computed from interaction-log features such as errors, correct answers, hints and timing, not from anything a student wrote (`DATA_v2.md:62`; no text column is loaded, `loader2012.py:16`). They are predictions, not observations, the detector version that produced them is unverified, and the file contains no human labels (`DATA_v2.md:58`, `:61`). We asked whether adding these four signals to the response-history risk model improves wheel-spinning prediction. For each signal we added three features at the decision point, the last value, the running mean and the running maximum over the rows seen so far, after a rank transform fitted on the 2012 training rows (`affect.py:5-18`, `affect.py:27-30`). On the locked 2012 test students, the mean AUC difference over k = 3, 5, 8 was +0.0022 (95% CI +0.0013 to +0.0031; `test_results_2012.json`, `b_within_dataset_affect`, `with_primary|variantA`, line 25415-25421; lagged version +0.0024, line 25399-25405). That is below the pre-registered bar of 0.005 (`SPEC.md:199`), the lagged version agrees, and choosing the threshold separately for each model changed no policy outcome measurably (`RESULTS_v2.md:130`). A small positive difference is therefore detectable but smaller than the margin we fixed in advance, and **we find no evidence that this detector adds decision-relevant information beyond the response history in this data**. This is a statement about this detector, this data and this decision; it is not a claim that affect is irrelevant to tutoring. Because the detectors read the same interaction logs the risk model already uses, overlap with the response history is expected (`DATA_v2.md:62`).

**Future work.** Other signals, such as keystroke dynamics, response latency used directly rather than through the detector, or gaze, could be tested against the same pre-registered gate (lower confidence bound above 0 and a point estimate of at least 0.005) on data held out in advance.

## Claims checked against the code and documents

| Claim | Source | Result |
|---|---|---|
| The four columns are `Average_confidence(...)` for FRUSTRATED, CONFUSED, CONCENTRATING, BORED | `aria_pace/loader2012.py:15` | verified |
| They are read raw into `Seq12.affect` | `aria_pace/loader2012.py:48`, `:21` | verified |
| No text column is loaded | `aria_pace/loader2012.py:16` (`USECOLS` lists ids, times, correctness, first_action, type, skill_id and the four affect columns only) | verified |
| Features: last value, running mean, running maximum; rank transform fitted on training rows | `aria_pace/affect.py:5-18`, `:21-24`, `:27-30` | verified |
| Computed from interaction logs (errors, correct answers, hints, timing) | `docs/aria_pace/DATA_v2.md:62` (cites Ocumpaugh et al., 69 features); `:76` (detectors "developed using data collected by conducting field observations") | verified in the data note; the primary papers are only partly read (`DATA_v2.md:69-74`) |
| Predictions, not observations; version unverified; no human labels in the file | `docs/aria_pace/DATA_v2.md:58`, `:61`, `:80` | verified |
| +0.0022 [+0.0013, +0.0031]; lagged +0.0024; bar 0.005 not met | `aria_pace/results/test_results_2012.json` (`b_within_dataset_affect`, lines 25386-25421); `docs/aria_pace/RESULTS_v2.md:49-52`; rule at `aria_pace/SPEC.md:199` | verified |
| Policy outcomes unchanged | `docs/aria_pace/RESULTS_v2.md:130` | verified |
| "Sensor-free" / "not text" | the detectors use interaction-log features only (`DATA_v2.md:62`); whether any detector input was text is not stated in the sources I read: **not found** beyond that | the claim rests on the feature list in the data note |

Note on the future-work sentence: response latency is already one of the detectors' inputs (`DATA_v2.md:62`), so a latency feature added directly would partly overlap with what the detector already summarises; it is still a legitimate test, but a clean one needs the detector columns left out of that comparison.
