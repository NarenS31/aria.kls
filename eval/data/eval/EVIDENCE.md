# ARIA — Evidence Table

_Generated 2026-07-24 17:16Z from the result JSONs; updated 2026-07-27 after the generator-agnostic heuristic repair. Every row's tier is applied strictly; nothing here is aspirational._

## Tier legend

- **A** — validated against real human labels
- **B** — validated against real behavior (proxy / measured labels)
- **C** — validated across independent generators
- **D** — synthetic only, circular
- **E** — asserted, not validated

**Tier distribution:** A=0, B=2, C=1, D=4, E=2

| Claim | Evidence | Tier | n | Limitation |
|---|---|:--:|--:|---|
| Cognitive-state detection from think-aloud text (same-generator accuracy 84.6% heuristic-only, up from 80.3%; macro-F1 0.837) | metacognition_eval Metric 1 + cross_generator_eval | D | 350 | still synthetic-only (no human labels); residual cross-generator gap 9.05 pts is concentrated in phi3's formal style. |
| Classifier accuracy transfers to unseen generators (mean gap 9.05 pts heuristic-only: {'mistral': +4.86, 'gemma2': -1.72, 'phi3': +24.00}) | cross_generator_eval (mistral/gemma2/phi3 vs llama3.1) | C | 1050 | gap 5-15 pts: partial overfitting. Markers repaired to be generator-agnostic (was 18.96 pts); phi3's verbose/formal style is the remaining weak spot. |
| Interventions are state-appropriate (mean 1.74/2 by LLM judge; FRUSTRATED 2.00, RUSHING 1.50) | metacognition_eval Metric 2 (llama3.1 judge) | E | 70 | LLM-as-judge with NO human agreement study; the judge shares a model family with the data generator. |
| Behavioral incorrectness aligns with real (gaze-measured) confusion (agreement 0.663, macro-F1 0.409) | external_validation Exp D (EduAgent310, real gaze labels) | B | 3584 | EduAgent310 has no think-aloud text, so ARIA's TEXT classifier is not tested; only FLOW/CONFUSED are recoverable and the classes are highly imbalanced. |
| Classifier survives contact with real human (non-LLM) text | external_validation Exp B (NCTE) | E | — | NCTE transcripts require a data-request form; run pending. |
| Self-initiated-metacognition (transfer) detection (F1 0.886) | metacognition_eval Metric 4 | D | 100 | Only PLANNING/INSIGHT are separable in the synthetic corpus; labels are deterministic by construction (circular). |
| Calibration measurement is valid (sim error 0.290) | metacognition_eval Metric 5 | D | 50 | Confidence ratings are simulated, not from real students. |
| Optimal-intervention-timing detection is valid (match rate 100%) | metacognition_eval Metric 6 | D | 32 | Timing optima are hypotheses tested on simulated scenarios, not measured on real recovery outcomes. |
| Longitudinal metacognitive-growth tracking (real usage) | data/metacognition/longitudinal_naren.json | B | 1 | n=1 real user; no statistical power, not generalizable. |
| Closed-loop selector enforces lexical grounding, question, length, and answer-leakage invariants | stratified `closed_loop_benchmark`, 20 episodes × 5 conditions across 10 topics | D | 100 responses/model | Full selector reached the automated ceiling for both models because verified fallbacks are constructed to satisfy these checks; requires blinded human ratings. |
| Current distilled checkpoint improves intervention quality | paired full-pipeline comparison against llama3.2:3b | E | 20 paired episodes | No automated quality difference; both hit the selector ceiling. Descriptive latency was 9.8s vs 4.7s. Existing saved tutoring score also does not favor the distilled model. |

### Update 2026-07-27 — generator-agnostic heuristic repair

Keyword markers were re-tuned to signals shared across ≥3 generators (audited with `eval/mine_signals.py`); the `use_llm` default is now `False` (heuristic-only). Measured with `cross_generator_eval` (heuristic-only) and `metacognition_eval` Metric 1 (n=350):

| Result | Value | Tier |
|---|---|:--:|
| Cross-generator gap (heuristic-only) | 9.05 pts | C |
| vs original heuristic gap | 18.96 pts | C |
| LLM routing degrades cross-gen performance | confirmed | C |
| Generator-agnostic heuristics outperform LLM | confirmed | C |

Detail: routing the low-confidence minority to the `llama3.2:3b` fallback at threshold 0.65 raised the mean gap to 11.14 pts and lowered accuracy on every generator (llama 0.846→0.814, gemma2 0.863→0.826, mistral 0.797→0.769, phi3 0.606→0.514). Same-generator (llama) accuracy rose 80.3% → 84.6% with the repaired heuristics — no regression.

### How to read this

The strongest ARIA claims are Tier B/C. **No claim is Tier A** (real human think-aloud labels) until the EDM 2024 think-aloud dataset is obtained (see `datasets/REQUEST_EMAIL.md`). Anything still at Tier D is validated only against the generator that produced its training data; Tier E is asserted from design rationale, not measured.
