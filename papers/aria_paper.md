# ARIA: A Closed-Loop, Learner-Conditioned Intervention Policy for Metacognitive Tutoring

**Authors:** ARIA Research Team

**Date:** July 2026

**Status:** Research draft; not peer reviewed

## Abstract

ARIA is a research prototype for selecting brief metacognitive questions from
the current problem, a student's current reasoning, a learner profile, and the
longitudinal interaction trace. The system generates multiple candidate
interventions, rejects candidates that are ungrounded, repetitive, or reveal an
answer, and ranks the remainder using problem, learner, and prior-outcome
signals. We introduce a paired, episode-level evaluation protocol with
progressive conditioning ablations and deterministic response-property checks.
Existing cognitive-state results are based mainly on synthetic think-aloud
examples; the only locally available external validation uses behavioral and
gaze-derived signals that do not test the text classifier directly. Existing
tutoring scores are also produced by an LLM judge without a human agreement
study. The current contribution is therefore a testable closed-loop systems
design and reproducible evaluation protocol, not evidence that ARIA improves
learning or is effective for students with ADHD. We specify the human
annotation, real think-aloud validation, and prospective student study required
to test those claims.

## 1. Motivation and research question

Many tutoring systems optimize the immediate answer. ARIA instead asks whether
a system can use evidence from the student's reasoning to choose one question
that helps the student plan, check, or recover while preserving ownership of
the solution. The design was motivated partly by lived experience with ADHD,
but this motivation is not evidence of population-specific effectiveness.

The present research question is:

> Does a learner-conditioned, problem-grounded, closed-loop intervention policy
> produce more specific, non-repetitive, answer-safe tutoring questions than
> generic or partially conditioned prompting?

The later causal question, which this paper does not yet answer, is whether such
interventions improve learning and unprompted metacognitive transfer.

## 2. System

### 2.1 Context representation

For each turn, ARIA constructs an intervention signature containing:

- student and task identifiers;
- topic and expected problem step;
- a short anchor from the student's current words;
- a candidate misconception;
- a fallible cognitive-state estimate;
- learner response preferences; and
- the outcome of the prior intervention.

A hash of this representation supports auditability without serving as an
evaluation label.

### 2.2 Candidate generation and selection

One language-model call requests several interventions using different
strategies. Verified fallback candidates are added so model failure does not
stop the learning session. A deterministic selector then evaluates:

- grounding in the student's current turn;
- grounding in the problem's key ideas;
- semantic similarity to recent interventions;
- answer leakage;
- one-question and length constraints;
- state-strategy fit; and
- smoothed historical effectiveness by student, topic, and strategy.

The selector records the chosen strategy, candidate source, rejection reasons,
and closest prior-response similarity.

### 2.3 Outcome update

On the next turn, ARIA records observable indicators such as a self-correction
phrase or a transition out of an estimated negative state. These are weak
online proxies, not proof of learning. Strategy-effectiveness estimates use
Beta smoothing so a single early event cannot dominate later selection.

## 3. Evaluation protocol

### 3.1 Task bank

The local task bank contains 100 answer-keyed tasks:

- 50 mathematics tasks;
- 30 English language arts tasks; and
- 20 science and coding-reasoning tasks.

Every task includes a solution or rubric target, expected reasoning steps, key
ideas, and multiple anticipated misconceptions. The bank is researcher-created
and is not a standardized achievement test.

### 3.2 Progressive-conditioning ablations

Every condition sees the same episode:

1. a fixed generic Socratic question;
2. model prompting with the problem and answer key;
3. problem plus the current student utterance;
4. problem, utterance, learner preference, and prior response pattern; and
5. the full candidate-generation and deterministic-selection pipeline.

This isolates the contribution of current-turn grounding, learner history, and
the selector. Additional confirmatory ablations should remove the grounding
gate, repetition gate, and outcome-learning term individually.

### 3.3 Automated response-property measures

The executable benchmark reports problem grounding, student-utterance
grounding, misconception targeting, answer leakage, one-question compliance,
conciseness, and latency. Lexical overlap checks are useful regression tests but
cannot establish pedagogical quality.

### 3.4 Human evaluation

The confirmatory evaluation requires two blinded educator raters. They score
problem grounding, student grounding, diagnostic usefulness, actionability,
learner ownership, and fit to learner context, and flag answer leakage,
invented student actions, generic responses, and safety concerns. Rater
agreement and adjudication are specified in
`eval/HUMAN_ANNOTATION_PROTOCOL.md`.

### 3.5 Statistics

Episode, not turn, is the unit of inference. Comparisons are paired by episode.
The analysis uses:

- episode-level bootstrap confidence intervals;
- paired sign-flip randomization tests;
- Holm correction across primary outcomes; and
- Cohen's dz for standardized paired effects.

All seeds, prompts, model identifiers, raw responses, latency, and selector
audits are retained.

## 4. Current evidence

### 4.1 Cognitive-state estimation

Same-generator synthetic think-aloud evaluation reports 84.6% accuracy and
0.837 macro-F1 for the heuristic classifier. Cross-generator results are lower
and vary substantially by generator. These results measure transfer between
synthetic generators, not accuracy on real student language.

EduAgent310 provides real gaze-derived cognitive labels, but it contains no
think-aloud text. A correctness-based behavioral proxy agreed with its
two-state labels at 0.663 accuracy and 0.409 macro-F1. This result illustrates
that observable incorrectness is a weak proxy for confusion; it does not
validate ARIA's text classifier.

### 4.2 Intervention quality

Earlier intervention-appropriateness results were rated by a language model
from the same family used elsewhere in the synthetic pipeline. No educator
agreement study exists. Those values are reported only as LLM-rated exploratory
measurements.

A stratified 20-episode smoke test compared five conditioning conditions across
all ten task-bank topics. The full selector reached the ceiling on the current
lexical regression checks for both `llama3.2:3b` and
`aria_distilled:latest`. This does not demonstrate perfect intervention quality:
verified fallbacks are constructed to contain student and problem anchors, so
the checks partly measure a design invariant. Blinded educator ratings are
required to distinguish substantive specificity from superficial overlap.

### 4.3 Distillation

The current stored distilled-model comparison does not demonstrate an
improvement over the base model. The distilled score is lower in the saved
comparison, the difference is not significant, and the live launcher does not
load a LoRA adapter. In the new 20-episode response-property smoke test, both
models tied at the selector's automated ceiling, while the distilled model's
descriptive full-pipeline latency was 9.8 seconds per response compared with
4.7 seconds for the 3B base model. This is a negative result, not an improvement
claim. No paper or demo should claim that training improved ARIA until a
human-rated controlled evaluation shows otherwise.

### 4.4 Learning and transfer

Current recovery, calibration, timing, and self-initiated-metacognition
measurements are simulations or limited traces. They validate software behavior
and generate hypotheses. They do not establish that ARIA changes student
learning, metacognitive skill, or classroom outcomes.

## 5. Limitations

1. No real human think-aloud corpus with compatible labels is locally available.
2. Synthetic cases can share the language patterns of their generators.
3. Automated grounding measures reward lexical overlap and can miss semantic
   quality or reward superficial quotation.
4. LLM judges are not substitutes for educator ratings.
5. Cognitive states are uncertain hypotheses and are not diagnoses.
6. There is no prospective student comparison, power analysis, or causal
   learning outcome.
7. There is no basis for generalizing effectiveness to students with ADHD or
   any other neurodivergent population.
8. Some external datasets are noncommercial or access controlled and cannot be
   redistributed.

## 6. Required studies

The following steps are required before a strong archival submission:

1. Obtain authorized real think-aloud data with human self-regulated-learning
   labels and freeze a held-out test split.
2. Complete blinded educator annotation with inter-rater reliability.
3. Evaluate independent model families and generators, including adversarial
   paraphrases and out-of-domain writing.
4. Preregister a powered prospective study with participant protections.
5. Compare against a strong adaptive tutoring baseline and a problem-aware
   direct-prompt baseline.
6. Measure immediate accuracy, delayed retention, unprompted transfer,
   intervention burden, and subgroup uncertainty.
7. Release prompts, seeds, raw outputs, exclusions, code, and a model/data card
   where licensing permits.

## 7. Conclusion

ARIA currently contributes an auditable architecture and a falsifiable
evaluation plan for closed-loop metacognitive intervention selection. The
prototype should be described as promising only in the narrow sense that its
mechanisms can now be tested. Claims about learning, neurodivergent students,
or classroom effectiveness remain open research questions.

## Artifact map

- Claim policy: `eval/CLAIM_LEDGER.md`
- Confirmatory design: `eval/RESEARCH_PROTOCOL.md`
- Human rating protocol: `eval/HUMAN_ANNOTATION_PROTOCOL.md`
- Paired statistics: `eval/research_stats.py`
- Executable benchmark: `eval/closed_loop_benchmark.py`
- Evidence table: `eval/data/eval/EVIDENCE.md`
- Limitations: `eval/data/eval/LIMITATIONS.md`

## Research benchmark references

- *Using Large Language Models to Detect Self-Regulated Learning in
  Think-Aloud Protocols.* Educational Data Mining 2024.
  <https://educationaldatamining.org/edm2024/proceedings/2024.EDM-long-papers.13/index.html>
- *MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical
  Capabilities of LLM Tutors.* <https://arxiv.org/abs/2502.18940>
- *TutorBench: A Benchmark for Evaluating AI Tutors.*
  <https://arxiv.org/abs/2510.02663>
- *KMP-Bench: A Multi-Turn Benchmark for K-8 Mathematics Pedagogy.*
  <https://arxiv.org/abs/2603.02775>
