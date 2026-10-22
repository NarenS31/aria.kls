# The 12 hardest questions, with short honest answers

Rules for every answer: say "under the 3-correct-in-a-row mastery definition" when you say 52%; the three settings are cost weights, not types of learner; nothing here shows that the tutor improves learning.

## 1. Why a 1995 model (BKT) and not a neural model?
Because the neural models did not do better at the thing the decision uses: ranking who will get stuck. We fixed a rule in advance (a neural model must beat BKT by at least 0.01 in AUC at both 3 and 5 attempts, with the confidence interval above zero) and tested four models on validation students. Differences from BKT at 3 and 5 attempts: single-skill LSTM-style -0.0027 and -0.0027, single-skill attention-style -0.0078 and -0.0134 (worse, intervals exclude zero), multi-skill LSTM-style -0.0042 and -0.0097, multi-skill attention-style -0.0111 and -0.0077 (all four multi-skill intervals include zero). The multi-skill LSTM does predict the next answer slightly better (log-loss lower by 0.0174, interval 0.0133 to 0.0218), but that did not improve the ranking. BKT is a 1995 model; our contribution is the decision policy, the protocol and these negative results, not a new model.

## 2. Why 2009 data, and what are those students?
It was the only candidate we verified to contain every signal we needed, every row comes from a "skill builder" set where three correct in a row is the system's own mastery rule, the file could be downloaded and checksummed, and the terms allow research use with citation. The weaknesses: the grade level is not stated by the sources, there are no timestamps (time is counted in attempts), and it is math practice, not calculus. We do not know how well it carries over to algebra-through-calculus problems.

## 3. Does this help students?
Not shown. We have run no study with students. The logs contain no tutor interventions, so replay cannot say what a student would have done if interrupted or helped. Fewer interruptions is a count in old data, not a learning result.

## 4. What is the 52% under other definitions?
The 52% is under the 3-correct-in-a-row definition, on a test evaluated once. We re-ran the whole pipeline on validation students under four other rules. On validation the quiet saving is 49.8% for the original rule, 8.6% for four in a row, 32.2% for five in a row within 15 attempts, 45.4% within 20, and 9.0% for a moving-average rule. The share of students counted as stuck ranges from 10.7% to 75.9%, and the best quiet threshold from 0.27 to 0.69. This check is exploratory, validation numbers are optimistic, it covers BKT only, and no alternative rule has a test number.

## 5. What does the closed-loop simulation show, and not show?
It shows how much the replay conclusion depends on an assumption the logs cannot measure: how costly it is to leave a stuck student alone. Over 100 assumed settings, the quiet rule beat always helping in 40, was worse in 55 and was indistinguishable in 5; the balanced rule beat it in 20 and was worse in 80; the supportive rule made the same decisions as always helping. The sign flips at roughly 9 to 10% chance of giving up per unaided wrong step for quiet, and roughly 2% for balanced. It does not show anything about real students: it is an assumption-dependent sensitivity analysis, not evidence of effect.

## 6. Why not just always help?
Because it depends on what each mistake costs. When interruptions are expensive (the quiet setting), always helping made 2.54 interventions per student against 1.22. When waiting is expensive (supportive), the rule saved only 3% and the difference was not distinguishable from zero. And the simulation says that if leaving a student stuck is costly enough, always helping wins. The settings make that trade-off explicit instead of hiding it.

## 7. What happens at attempts 1 and 2, and after attempt 9?
The policy was fitted and tested only at attempts 3 to 9, so that is where the gate applies. In the live demo's pace build, a wrong step on attempt 1 or 2, or from attempt 10 on, always gets a hint. Here an attempt means an answer the math engine marked right or wrong; "I don't know" and restating the problem do not count.

## 8. How does privacy work in the "Would you step in?" activity?
Eight moments, about two minutes. Your answers stay in your browser tab; nothing is uploaded or sent. A CSV is created only if you click download. It asks no names or emails (role, years teaching and grade band are optional). The consent text is a draft that an adult has not yet reviewed. The tutor itself is a separate program, and its storage and network behaviour must be verified before any classroom use: we do not yet promise that no text leaves the device. The website has no analytics and no backend; the forms prepare an email you send yourself.

## 9. What would you do with a teacher pilot?
Nothing is open yet. The proposal is 2 to 3 sessions of 30 to 45 minutes on typed math, with format agreed with the school, after offline checks, consent review and school approval. First step, though, is smaller: go through the tutor's replies with a teacher, compare their "would you step in" calls with ARIA's decisions, and take their criticism of the method. We would measure nothing about learning until a reviewed study exists.

## 10. Are quiet, balanced and supportive types of student?
No. They are cost weights: how much an unnecessary interruption costs compared with a delayed hint (3 to 1, 1 to 1, and 1 to 3). We chose them. Nothing we did shows that any group of learners corresponds to them, and we make no claim about any clinical or other group.

## 11. Does it ever give the answer, and is the math right?
Every reply is checked for the student's answer and for false math claims before it is shown. In scripted runs on the 8 featured problems (104 turns) there were 0 leaks, 0 math errors and 0 crashes. Those are scripted inputs, not students, and the replay itself did not evaluate the math checker.

## 12. What is the weakest point, and what would change your mind?
Three things. The headline saving depends on how "stuck" is defined (question 4). Replay cannot tell us what helping does (question 3), and the cost of leaving a student alone is unmeasured (question 5). Also, counting students who left early as not stuck moves the best quiet threshold from 0.57 to 0.85 and raises missed help from 3.7% to 12.4% (95% CI 9.4% to 15.5%). We would change our view if a reviewed pilot showed students staying stuck longer under the quiet setting, or if teachers consistently disagreed with its calls.

## Limits
Full text: [`docs/aria_pace/LIMITS.md`](../aria_pace/LIMITS.md) (on branch `aria-pace`).

- **a. Interruption cost is an assumed weight, not a measured harm.** J = w_I x unnecessary interventions + w_D x delayed help is an offline simulation, and "unnecessary" is a proxy. Whether the Quiet setting reduces frustration or preserves working memory has not been measured; that needs a live comparison with students, which needs approval.
- **b. The 52% depends on the mastery definition.** 3-correct-in-a-row is the ASSISTments skill-builder's own completion rule. Under four other definitions the Quiet saving was 8.6%-45.4% (validation only, exploratory, BKT only).
- **c. The risk model is an offline baseline calibration.** It was fitted on ASSISTments skill-builder math logs (grade level not stated), not calculus, and needs re-calibration before deployment; the demo applies it unchanged.
- **d. The language analyzer was evaluated only on synthetic messages.** The ARIA-Pace gate and its published results do not depend on it. In the demo its flags (they need an explicit feeling word) can add an affect-based reply or prompt a reply to a words-only message, but never override correctness.
