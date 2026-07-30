# ARIA claim ledger

This ledger is the source of truth for papers, websites, talks, and demos.

| Claim | Current evidence | Status | Allowed wording |
|---|---|---|---|
| ARIA uses the problem, current reasoning, learner context, and interaction history when constructing an intervention. | Product code, prompt audit, and reliability tests. | Supported as a system property. | “ARIA conditions interventions on four sources of context.” |
| ARIA filters semantically repetitive, ungrounded, and answer-revealing candidates. | Deterministic selector tests and selector audit logs. | Supported as an implementation property. | “ARIA applies deterministic safety and novelty gates.” |
| ARIA records observable reasoning moves with exact evidence spans. | Transparent pattern baseline and software tests. No independent human-label validation. | Supported only as an implementation property. | “ARIA logs automatic hypotheses about observable reasoning moves.” |
| ARIA recognizes observable reasoning moves in real students. | Real Eedi dialogue is available for private annotation, but labels are pending. | Unsupported pending independent labels. | Do not claim accuracy until student/session-disjoint human validation is complete. |
| The closed-loop selector produces more grounded responses than weaker prompting conditions. | Paired synthetic benchmark, pending adequate sample and human validation. | Preliminary only. | “In a simulated offline benchmark…” followed by sample size, effect, CI, and corrected p value. |
| ARIA identifies seven cognitive states from student think-aloud text. | Same-generator synthetic accuracy and cross-generator synthetic transfer. No real human text labels locally. | Not externally validated. | “ARIA estimates one of seven states; synthetic evaluation suggests…, but real-text validation is pending.” |
| ARIA recognizes confusion in real students. | EduAgent gaze/correctness proxy supports only a weak two-state behavioral comparison and does not test text. | Unsupported for the product classifier. | Do not claim. |
| ARIA improves student learning or metacognitive transfer. | Simulated outcomes and one-user trace only. | Unsupported. | “Designed to test whether…” |
| ARIA is effective for students with ADHD or neurodivergent students. | Synthetic personas; no population study. | Unsupported. | “Motivated partly by lived experience with ADHD; population-specific efficacy is untested.” |
| The distilled model improves over its base model. | Stored score is lower than the base comparison and not significant; live launcher does not load a LoRA adapter. | Contradicted by current evidence. | “The current distillation attempt did not demonstrate improvement.” |
| LLM-rated tutoring scores represent educator judgment. | Single-family LLM judge; no educator agreement study. | Unsupported. | “LLM-rated” only. |
| The system is diagnostic. | No clinical design or validation. | Prohibited. | “A learning-support research prototype, not a diagnostic tool.” |

## Promotion rule

A claim moves to “supported” only when its named result artifact, sample,
analysis unit, confidence interval, and limitations are available. Statistical
significance without a practically meaningful effect is insufficient.

## Research program

The prospective protocols, evidence traceability, observable-move taxonomy,
task review packet, ethics materials, and student-study plan are in
`research/`. Their existence is preparation, not completed evidence.
