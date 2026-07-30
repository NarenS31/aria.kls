# Blinded intervention annotation protocol

## Rater packet

For each randomized item, show the task, the student's current utterance, and
one anonymous tutor response. Do not expose system name, model, condition, or
student identity. Responses from the same episode should not appear adjacent.

## Ratings

Rate each dimension from 1 (poor) to 5 (excellent).

1. **Problem grounding:** The response reflects the actual task or its required
   reasoning.
2. **Student grounding:** The response addresses the student's specific words
   or decision, rather than giving generic encouragement.
3. **Diagnostic usefulness:** The question targets the likely reasoning gap
   without claiming certainty about the student's internal state.
4. **Actionability:** The student can take a clear next reasoning action.
5. **Learner ownership:** The response supports thinking without revealing the
   answer or completing the work.
6. **Fit to learner context:** The form and length fit the supplied preference
   and prior interaction.

Also mark:

- Reveals all or part of the answer: yes/no.
- Invents something the student did or said: yes/no.
- Could be pasted unchanged into an unrelated problem: yes/no.
- Safety or fairness concern: yes/no, with a short explanation.

## Reliability and adjudication

Two educators independently rate every confirmatory item. Report weighted
Cohen's kappa for ordinal ratings, ordinary kappa for binary flags, and raw
agreement. Resolve disagreements only after independent ratings are locked.
Keep both original ratings and the adjudicated label.

## Exclusion policy

Exclude an item only for a prespecified technical failure such as an empty or
truncated response. Never exclude a low-quality response. Report exclusions by
condition and repeat the analysis with failures scored at the minimum.
