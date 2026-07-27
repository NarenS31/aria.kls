# ARIA — Evidence Table

_Generated 2026-07-24 17:16Z from the result JSONs. Every row's tier is applied strictly; nothing here is aspirational._

## Tier legend

- **A** — validated against real human labels
- **B** — validated against real behavior (proxy / measured labels)
- **C** — validated across independent generators
- **D** — synthetic only, circular
- **E** — asserted, not validated

**Tier distribution:** A=0, B=2, C=1, D=4, E=2

| Claim | Evidence | Tier | n | Limitation |
|---|---|:--:|--:|---|
| Cognitive-state detection from think-aloud text (accuracy 82.0%, macro-F1 0.814) | metacognition_eval + cross_generator_eval | D | 100 | largely detects llama3.1's fingerprint (mean gap 18.96 pts > 15); accuracy does not transfer to other generators. |
| Classifier accuracy transfers to unseen generators (mean gap 18.96 pts: {'mistral': 17.43, 'gemma2': 4.29, 'phi3': 35.15}) | cross_generator_eval (mistral/gemma2/phi3 vs llama3.1) | C | 1050 | gap > 15 pts: classifier is largely detecting llama3.1's stylistic fingerprint, NOT cognitive state. Major limitation. |
| Interventions are state-appropriate (mean 1.74/2 by LLM judge; FRUSTRATED 2.00, RUSHING 1.50) | metacognition_eval Metric 2 (llama3.1 judge) | E | 70 | LLM-as-judge with NO human agreement study; the judge shares a model family with the data generator. |
| Behavioral incorrectness aligns with real (gaze-measured) confusion (agreement 0.663, macro-F1 0.409) | external_validation Exp D (EduAgent310, real gaze labels) | B | 3584 | EduAgent310 has no think-aloud text, so ARIA's TEXT classifier is not tested; only FLOW/CONFUSED are recoverable and the classes are highly imbalanced. |
| Classifier survives contact with real human (non-LLM) text | external_validation Exp B (NCTE) | E | — | NCTE transcripts require a data-request form; run pending. |
| Self-initiated-metacognition (transfer) detection (F1 0.886) | metacognition_eval Metric 4 | D | 100 | Only PLANNING/INSIGHT are separable in the synthetic corpus; labels are deterministic by construction (circular). |
| Calibration measurement is valid (sim error 0.290) | metacognition_eval Metric 5 | D | 50 | Confidence ratings are simulated, not from real students. |
| Optimal-intervention-timing detection is valid (match rate 100%) | metacognition_eval Metric 6 | D | 32 | Timing optima are hypotheses tested on simulated scenarios, not measured on real recovery outcomes. |
| Longitudinal metacognitive-growth tracking (real usage) | data/metacognition/longitudinal_naren.json | B | 1 | n=1 real user; no statistical power, not generalizable. |

### How to read this

The strongest ARIA claims are Tier B/C. **No claim is Tier A** (real human think-aloud labels) until the EDM 2024 think-aloud dataset is obtained (see `datasets/REQUEST_EMAIL.md`). Anything still at Tier D is validated only against the generator that produced its training data; Tier E is asserted from design rationale, not measured.
