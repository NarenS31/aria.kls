# Prospective validation of observable reasoning moves

Status: protocol and tooling ready; independent labels pending.

## Question

How accurately and confidently does ARIA identify observable reasoning and
self-regulation moves in real student language?

This study does not validate hidden cognitive states, diagnoses, or learning
effectiveness.

## Data sources

1. **Primary prospective corpus:** consented ARIA think-aloud sessions collected
   under institutional approval.
2. **External development corpus:** appropriately licensed real tutoring
   dialogue such as Eedi, used according to its noncommercial
   CC BY-NC-SA 4.0 terms.
3. **External validation corpus:** an independently collected or licensed
   corpus from a different setting, held untouched until the system and
   annotation manual are frozen.

Synthetic language can be used for software tests but is never counted as human
validation.

## Sampling

- Sample complete dialogue windows, not isolated phrases.
- Include the current student turn and up to two preceding turns.
- Sample by intervention/session so one conversation cannot appear across
  development and test sets.
- Prespecify strata for domain, message length, transcript type, and code
  prevalence when known.
- Remove or replace incidental direct identifiers before annotators receive
  text.

## Annotation

- Two trained independent annotators label every confirmatory item using
  `OBSERVABLE_REASONING_TAXONOMY.md`.
- Every positive code requires an exact evidence span.
- A calibration set is discussed before the manual is locked.
- Confirmatory disagreements remain stored; adjudication is separate.
- Annotators do not see ARIA predictions.

## Metrics

For each code:

- prevalence;
- precision, recall, F1, and support;
- Cohen's kappa and positive/negative agreement between annotators;
- probability calibration and expected calibration error;
- abstention coverage versus accuracy;
- confusion or co-occurrence analysis where relevant.

Macro averages are reported alongside per-code results. Micro accuracy alone is
not an acceptable summary for rare moves.

## Generalization tests

- Split by student when IDs exist; otherwise by complete intervention/session.
- Evaluate typed and speech-transcribed text separately.
- Report results by subject and source.
- Test on misspellings, fragments, indirect requests, and ASR errors without
  adding final-test phrases to training.
- Freeze a final external test set and record its content hash before model
  tuning.

## Claim threshold

The wording “ARIA recognizes observable reasoning moves” requires:

- independent real-human labels;
- acceptable inter-rater reliability;
- a student/session-disjoint test;
- prespecified per-code minimum performance and calibration;
- transparent abstention for unsupported turns.

Until then, outputs are “automatic hypotheses from a development classifier.”

