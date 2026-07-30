# ARIA language and intervention model card

Version date: 2026-07-29.

## Components

1. `student_intent.joblib`: TF-IDF character/word n-gram logistic regression.
2. `student_understanding.py`: transparent semantic routing plus an optional,
   time-bounded local language-model parser.
3. `reasoning_moves.py`: transparent multi-label observable-move baseline.
4. `ClosedLoopInterventionPipeline`: candidate filtering and selection.
5. Optional local Ollama models for candidate generation.

## Intended use

Research support for selecting brief, answer-preserving questions on
educator-approved learning tasks. Outputs are hypotheses and prompts, not
grades, diagnoses, accommodations, or clinical advice.

## Training and development data

- The intent model uses real Eedi tutoring dialogue under CC BY-NC-SA 4.0,
  noncommercial terms.
- Intent labels are weak supervision and are not human ground truth.
- The observable-move baseline is manually specified from an evidence-informed
  prospective taxonomy.
- Synthetic data are used for software and robustness development.

Exact counts, class balance, seed, source, license, and weak-label performance
are recorded in `models/student_intent_metadata.json`.

## Current evaluation

- Weak-label held-out metrics measure reproduction of weak rules.
- A 72-case human-written development challenge was used to repair the router;
  its final score is not independent validation.
- Synthetic state metrics do not validate real student language.
- Independent human-labeled observable-move evaluation is pending.
- Educator-rated intervention comparison is pending.
- Student learning outcomes are pending.

## Known risks

- Misinterpreting slang, fragments, indirect references, or speech transcripts.
- Overconfidence on out-of-distribution language.
- Mistaking emotion for reasoning or reasoning for a hidden state.
- Incorrect problem feedback when the task model is wrong.
- Answer leakage, repeated prompts, invented student actions, and differential
  performance across language and accessibility groups.
- Local model latency or timeout.

## Mitigations

- Exact evidence spans and observable codes.
- Abstention/clarification for unsupported turns.
- Answer-keyed task models and educator review gate.
- Deterministic leakage, grounding, repetition, and one-question checks.
- Human supervision in prospective studies.
- Versioned logs, failure retention, and no diagnosis claims.

## Prohibited use

High-stakes grading, discipline, disability inference, clinical diagnosis or
treatment, unsupervised crisis handling, or replacing legally required
educational services.

