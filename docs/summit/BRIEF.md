# ARIA: When Should an AI Math Tutor Stay Quiet?

*One-page brief (draft). A student project; a replay of old logs, not a classroom study.*

**Question.** When a student is working on a math problem, when should an AI tutor speak, and when is it better to say nothing?

**How ARIA works.** A symbolic math engine (SymPy) checks each step the student types. If the step is fine, ARIA stays quiet. Only after a *verified wrong step* does a timing model decide whether to offer a hint. If it decides to stay quiet, the student sees a notice that a flagged step was held back, so a wrong step is never met with silence.

**Three profiles.** Quiet, Balanced and Supportive are three *cost weights* on the trade-off between speaking too often and missing a student who is stuck (3:1, 1:1, 1:3). They are not types of learner.

**Headline result.** The timing model is a 1995 knowledge-tracing method (Bayesian knowledge tracing), fitted on ASSISTments skill-builder math logs (grade level not stated), not calculus. In a replay of 597 held-out students, the Quiet profile intervened **52% less often** than helping after every wrong step and missed **3.7%** (95% CI 2.2-5.3) of students who went on to get stuck. "Mastery" here means **three correct answers in a row**.

**Sensitivity.** The saving depends on the definition. Cross-validated on training and validation students (exploratory, BKT only), the Quiet saving is **50.1%** [48.5, 51.6] under 3 in a row, with 6.2% missed help, and **7.1% to 41.6%** under the four other definitions. Missed help is about 4 to 6% across our checks (3.7% on the locked test).

**Limits.**
- Replay of logged data. It says nothing about whether students learn more.
- One simple model; neural alternatives and a per-turn model with response time did not clear the pre-registered bar in validation.
- The demo approximates the evaluated policy: it uses one pooled set of model parameters, while the evaluated policy used per-skill parameters. On validation this changes the Quiet saving from 49.8% to 43.9%, and 14.4% of Quiet wrong-step decisions flip.
- The answer-leak and math-error checks (0 of 160 turns per profile) come from scripted inputs on a 30-problem bank (algebra / trigonometry / calculus), not free-form students.
- The concept map in the Teacher summary is exploratory and not validated.
- No classroom pilot has taken place. A pilot protocol exists only as an unapproved draft.
- Full limits (assumed cost weights, definition-dependent 52%, math-log calibration, analyzer on synthetic text only): `docs/aria_pace/LIMITS.md`.

**Try it.** Do the "Would you step in?" activity and tell us where we are wrong: **aria-learning.org/activity**

**Contact.** [Contact details to be added by the author]

> QR code placeholder: [insert QR code for https://aria-learning.org here before printing]
