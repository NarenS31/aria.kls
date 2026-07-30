# Observable student reasoning-move taxonomy

Version 1.0, for prospective human validation.

## Unit and principle

The unit is a meaning-bearing student utterance segment. Codes describe what is
observable in the words, not what the student secretly feels or knows. Multiple
codes may apply. Every code requires an exact text span. Coders must not use
correct-answer knowledge when assigning discourse moves.

| Code | Operational definition | Positive example | Important exclusion |
|---|---|---|---|
| `TASK_ORIENTATION` | Analyzes or restates what the task requires. | “I have to find the slope.” | Merely rereading the entire prompt. |
| `TASK_META` | Comments on beginning, attempting, or participating without proposing an academic step. | “This is my first attempt.” | An actual planned operation. |
| `PLAN` | Names an intended strategy or future step. | “First I’ll isolate the variable.” | “This is my first attempt.” |
| `STRATEGY_STEP` | States a concrete operation, claim, or evidence choice. | “Subtract 2x from both sides.” | A bare answer. |
| `JUSTIFICATION` | Gives a reason linking a step or claim to the task. | “because that keeps the equation balanced” | “because” with no intelligible relation. |
| `MONITORING` | Checks present understanding, progress, or correctness. | “Does that step keep both sides equal?” | A retrospective judgment of a finished strategy. |
| `EVALUATION` | Judges a result or completed strategy against a criterion. | “Substitution gives the same value, so it checks.” | Uncertain prediction before acting. |
| `SELF_CORRECTION` | Explicitly retracts or revises earlier reasoning. | “Wait, I meant subtract, not add.” | A new idea with no signaled revision. |
| `UNCERTAINTY` | Explicitly marks uncertainty about academic content. | “Maybe the slope is 2.” | General frustration without a content proposition. |
| `HELP_SEEKING` | Requests a hint, explanation, start, or verification. | “Can you explain what coefficient means?” | A rhetorical complaint without a request. |
| `ANSWER_ONLY` | Supplies only an answer or option. | “x = 6” | An answer with a reason. |
| `AFFECT` | Expresses task-related emotion or disengagement. | “This is making me frustrated.” | Clinical interpretation such as diagnosing anxiety. |
| `OFF_TASK` | Contains no codable task reasoning or regulation. | “What time is lunch?” | A brief but valid answer. |

## Annotation procedure

1. Segment only when one utterance contains independently meaningful moves.
2. Assign every applicable code.
3. Highlight the smallest sufficient evidence span for each code.
4. Mark `UNCLEAR` outside the taxonomy when audio/text is unintelligible.
5. Coders independently label the calibration set.
6. Discuss disagreements and revise this manual before the confirmatory set.
7. Lock the manual, then independently label the confirmatory set.
8. Retain both original labels; adjudication creates a separate field.

## Reliability

- Report per-code prevalence, positive agreement, negative agreement, and
  Cohen's kappa for each binary code.
- Report macro-F1 between coders because rare codes can make raw agreement look
  misleadingly high.
- Do not merge codes solely to increase agreement without documenting the
  revision and re-annotating the calibration set.

## Automatic model evaluation

- Split by student or complete session, never by utterance.
- Keep development and final test participants disjoint.
- Report per-code precision, recall, F1, prevalence, and calibration.
- Measure performance on typed text and speech transcripts separately.
- Report an abstention curve: coverage versus accuracy as the confidence
  threshold changes.
- Report performance by task domain, transcript source, message length, and
  prespecified accessibility groups only when sample sizes and approvals permit.

`agent/reasoning_moves.py` is a transparent software baseline. Its pattern
outputs are not human ground truth and must not be used to train and evaluate
on the same examples.
