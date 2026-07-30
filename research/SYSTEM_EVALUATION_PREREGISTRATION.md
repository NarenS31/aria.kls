# Prospective registration: blinded educator evaluation

Status: **draft, not registered, no ratings collected**  
Version: 1.0  
Registration venue: [OSF project/registration URL pending]  
Investigators and independent mentor: [pending]

## Research question

On the same answer-keyed tutoring episodes, do qualified educators rate full
ARIA interventions as more grounded and pedagogically useful than a generic
Socratic prompt and partially conditioned language-model baselines?

This study evaluates response properties. It does not evaluate student
learning.

## Conditions

1. `generic`: fixed Socratic question.
2. `problem_only`: task model, without the student's current reasoning.
3. `turn_grounded`: task model plus current utterance.
4. `profile_history`: task, utterance, declared preference, and prior pattern.
5. `full_closed_loop`: candidate generation plus grounding, leakage,
   repetition, and history-aware selection.

All conditions use the same model version and generation budget except the
fixed generic condition. Prompts, model hashes, decoding parameters, latency,
failures, and raw outputs are retained.

## Sample

- Target: all 100 version-locked tasks, with one prespecified misconception
  episode per task and all five paired conditions.
- At least two independent qualified educators rate every response.
- Subject-specific responses are rated only by reviewers who attest relevant
  expertise.
- A separate 10-item calibration packet is discussed before confirmatory
  ratings. Calibration items are excluded from confirmatory inference.

## Blinding and randomization

- Responses are labeled with opaque IDs.
- Condition, system, model, task author, and selector outcome are hidden.
- Order is randomized separately for each rater.
- Responses from the same episode are separated where feasible.
- The condition key remains inaccessible until both raters submit locked files.

## Outcomes

Ordinal, 1 to 5:

- problem grounding
- student grounding
- diagnostic usefulness
- actionability
- learner ownership
- fit to supplied learner context

Binary:

- reveals an answer
- invents a student action
- generic across unrelated problems
- safety or fairness concern

### Primary family

Student grounding, diagnostic usefulness, and actionability. Holm correction
controls family-wise error across these three outcomes.

### Safety outcomes

Answer revelation and invented student action are reported for every condition
with exact counts and confidence intervals. They are not hidden by a composite.

## Analysis

- Episode is the unit of inference.
- Rater scores are averaged within response only after reporting reliability.
- Full ARIA is compared separately with `generic`, `problem_only`, and
  `turn_grounded`.
- Paired episode-level mean differences receive nonparametric bootstrap 95%
  confidence intervals.
- Two-sided paired sign-flip tests are reported with Holm-adjusted p values.
- Cohen's dz is reported as a descriptive standardized effect.
- Ordinal inter-rater agreement uses quadratic-weighted Cohen's kappa; binary
  flags use ordinary Cohen's kappa. Raw agreement and prevalence are also
  reported.
- Results are stratified descriptively by subject and topic. No underpowered
  subgroup significance claims are made.

## Missingness and failures

- Empty, timed-out, or malformed model responses remain in the sample and
  receive the minimum score unless the failure occurred before condition
  assignment.
- Missing rater cells are reported. Primary analysis requires two ratings;
  sensitivity analysis retains single-rated items.
- No response is excluded for poor quality.

## Decision rule

The system-property claim advances only if full ARIA has a positive adjusted
effect on the primary family without a higher answer-revelation or invention
rate. Statistical significance alone is insufficient; effect sizes,
confidence intervals, and representative failures are reported.

## Deviations

Any change after registration is timestamped, justified, and labeled
exploratory. The original plan remains available.

