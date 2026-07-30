# ARIA research protocol

## Research question

Does a learner-conditioned, problem-grounded, closed-loop intervention policy
produce more specific, non-repetitive, answer-safe tutoring questions than
generic or partially conditioned prompting?

This is the current defensible systems question. It is separate from the later
causal question of whether ARIA improves learning.

## Confirmatory conditions

Every condition receives the same task and is evaluated on the same episode.

1. Generic fixed Socratic prompt.
2. Language model with the problem and answer key only.
3. Problem plus the student's current utterance.
4. Problem, utterance, learner preference, and prior response pattern.
5. Full ARIA candidate generation, grounding gate, answer-leakage gate,
   semantic repetition filter, and history-aware selection.

## Primary outcomes

- Student-utterance grounding.
- Problem grounding.
- Misconception targeting.
- Answer leakage.
- One-question contract.
- Semantic repetition across a session.

Automated lexical checks are regression tests, not a substitute for human
ratings. The confirmatory human outcome is a blinded educator rating using the
annotation protocol.

## Statistical analysis

- Episode is the unit of inference.
- All comparisons are paired on episode.
- Mean differences use an episode-level nonparametric bootstrap confidence
  interval.
- P values use paired sign-flip randomization tests.
- Holm correction controls family-wise error across primary outcomes.
- Cohen's dz reports standardized paired effects.
- Results are broken down by subject, task topic, and learner profile.
- Seeds, prompts, model identifiers, latency, raw responses, and selector
  audits are retained.

## Required external validation

The synthetic benchmark cannot establish learning effectiveness or accurate
cognitive-state inference. A publishable study still needs:

1. Real think-aloud transcripts with human self-regulated-learning labels.
2. Blinded ratings from at least two educators, with adjudication and
   inter-rater reliability.
3. A prospective student study comparing learning and unprompted transfer
   against a strong tutoring baseline.
4. A preregistered analysis plan, power analysis, and participant protections.
5. Evaluation on models and generators not used to create training data.

## Claim policy

- Synthetic response-property results must say `simulated`.
- Behavioral proxy results must say `proxy` and name the measured behavior.
- LLM-judge results must say `LLM-rated` until human agreement is measured.
- A nonsignificant or negative comparison cannot be described as an
  improvement.
- No result may be generalized to neurodivergent students without data from
  that population and an appropriate study design.
