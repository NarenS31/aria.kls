# ARIA — Evidence Table

_Generated 2026-08-23 13:41Z from the result JSONs. Every row's tier is applied strictly; nothing here is aspirational._

## Tier legend

- **A** — validated against real human labels
- **B** — validated against real behavior (proxy / measured labels)
- **C** — validated across independent generators
- **D** — synthetic only, circular
- **E** — asserted, not validated

**Tier distribution:** A=0, B=4, C=3, D=3, E=2

| Claim | Evidence | Tier | n | Limitation |
|---|---|:--:|--:|---|
| Cognitive-state detection from think-aloud text (accuracy 82.0%, macro-F1 0.814) | metacognition_eval + cross_generator_eval | C | 100 | partial generator overfitting (mean gap 9.05 pts); some signal is genuine, some is llama3.1-specific. |
| Classifier accuracy transfers to unseen generators (mean gap 9.05 pts: {'mistral': 4.86, 'gemma2': -1.72, 'phi3': 24.0}) | cross_generator_eval (mistral/gemma2/phi3 vs llama3.1) | C | 1050 | gap 5-15 pts: partial generator overfitting; some signal is genuine, some is generator-specific. |
| Interventions are state-appropriate (mean 1.74/2 by LLM judge; FRUSTRATED 2.00, RUSHING 1.50) | metacognition_eval Metric 2 (llama3.1 judge) | E | 70 | LLM-as-judge with NO human agreement study; the judge shares a model family with the data generator. |
| Behavioral incorrectness aligns with real (gaze-measured) confusion (agreement 0.663, macro-F1 0.409) | external_validation Exp D (EduAgent310, real gaze labels) | B | 3584 | EduAgent310 has no think-aloud text, so ARIA's TEXT classifier is not tested; only FLOW/CONFUSED are recoverable and the classes are highly imbalanced. |
| Classifier survives contact with real human (non-LLM) text | external_validation Exp B (NCTE) | E | — | NCTE transcripts require a data-request form; run pending. |
| Self-initiated-metacognition (transfer) detection (F1 0.886) | metacognition_eval Metric 4 | D | 100 | Only PLANNING/INSIGHT are separable in the synthetic corpus; labels are deterministic by construction (circular). |
| Calibration measurement is valid (sim error 0.290) | metacognition_eval Metric 5 | D | 50 | Confidence ratings are simulated, not from real students. |
| Optimal-intervention-timing detection is valid (match rate 100%) | metacognition_eval Metric 6 | D | 32 | Timing optima are hypotheses tested on simulated scenarios, not measured on real recovery outcomes. |
| Longitudinal metacognitive-growth tracking (real usage) | data/metacognition/longitudinal_naren.json | B | 1 | n=1 real user; no statistical power, not generalizable. |
| Behavioral proxy mapping runs on 7 independent real datasets across 5 platforms (ASSISTments, EdNet / Santa (Riiid), Eedi, Fractions Lab (DataShop), OULAD (Open University)), 2004-2020 (2,068,294 transactions, 285,222 students) | external_validation_multi_dataset.json | B | 2068294 | Mapping alone is not validation. Only 3/7 carry the hint, attempt and timing fields the proxy rules read; the other 4 (assistments_2015, oulad, ednet_riiid, eedi_neurips2020) yield no state and are reported as uncomputable, not as null results. |
| Real behavioral state prevalence differs from the distribution ARIA's corpus assumes (3/3 testable datasets reject, p < 1e-300 each) | external_validation_multi_dataset.json chi-square goodness-of-fit, per dataset | C | 404150 | Weak as stated: the expectation is the synthetic corpus, which is uniform BY CONSTRUCTION (50 items per state), so this rejects uniformity rather than a considered prevalence model — a near-foregone result. What is usable is the shape of the departure per dataset (observed vs expected counts in the JSON), and the fact that it is consistent across two independent ASSISTments years. Do not cite the synthetic distribution as behaviourally calibrated. |
| ARIA's absolute timing thresholds track percentile-based real behaviour (agreement assistments 2004 73.7%, assistments 2005 64.7%, aleven fractions 18.4%) | external_validation_multi_dataset.json timing agreement, per dataset | B | 66138 | Agreement is not stable across sources: 18.4% to 73.7%. The two ASSISTments logs agree far better than the step-level Fractions Lab data, so the thresholds are calibrated to problem-grain tutor logs and should not be claimed as platform-independent. |

### How to read this

The strongest ARIA claims are Tier B/C. **No claim is Tier A** (real human think-aloud labels) until the EDM 2024 think-aloud dataset is obtained (see `datasets/REQUEST_EMAIL.md`). Anything still at Tier D is validated only against the generator that produced its training data; Tier E is asserted from design rationale, not measured.
