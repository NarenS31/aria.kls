# Limits of ARIA-Pace (read this before quoting a number)

Each item says what the evidence does and does not support. Sources are file:line on branch `aria-pace` unless a branch is named.

## a. Interruption cost is an assumed weight, not a measured harm

The cost J = w_I x (unnecessary interventions) + w_D x (delayed help) is an offline policy simulation (`aria_pace/SPEC.md:135`; weights at `:129-133`). "Unnecessary interruption" is a proxy defined from logs (`SPEC.md:218`), not an observed harm. Whether the Quiet weighting reduces frustration or preserves working memory in students has not been measured; that needs a live human comparison, such as the pilot in `PILOT_PROTOCOL.md` (branch `learning-graph`), which needs approval first. The three profiles are cost weights, not learner types, and the A6 sweep varies the assumed costs (`docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md:5,76`).

## b. The 52% depends on the mastery definition

"3 correct in a row" is the ASSISTments skill-builder's own completion rule for the logged problem sets (`docs/aria_pace/DATA.md:15`), and the headline 52% is under that definition. Under the four other definitions tested (4 in a row; 5 in a row with T=15 and T=20; an EMA rule), the Quiet saving was 8.6%, 32.2%, 45.4% and 9.0% (`docs/aria_pace/MASTERY_SENSITIVITY.md:20,23,26,29`), against 49.8% for the 3-in-a-row control on the same validation students (`:17`). These checks are validation only, exploratory and BKT only, so in our setup the headline saving is fragile to the definition; nothing is claimed about other knowledge-tracing benchmarks, which were not tested.

## c. The risk model is an offline baseline calibration

The parameters were fitted on ASSISTments skill-builder math logs (grade level not stated: `aria_pace/SPEC.md:94`, `docs/aria_pace/RESULTS.md:205`, `docs/aria_pace/DATA.md:140`), not on calculus. The live demo applies them unchanged to algebra, trigonometry and calculus problems (branch `summit-pace-mac`, `agent/demo_pace.py:91-107`, which loads the frozen parameters). Re-calibration on the target domain is required before any deployment.

## d. Status of the language analyzer

The analyzer was evaluated only on synthetic messages, with no independent human ground truth, and accuracy dropped by about 9 points when a different model generated the messages (`docs/AUDIT_20260826.md:163`). The ARIA-Pace gate and its published results do not depend on it: they use the SymPy verdict and the BKT risk only. In the demo (branch `summit-pace-mac`) the analyzer's flags, which need an explicit feeling word in the student's own words (`agent/demo_tutor.py:103-104`), can set the state to FRUSTRATED or CONFUSED (`:137-140`) and so add an affect-based reply (`encouragement` or `break_suggestion` after repeated frustration, `eval/metacognition/jitai.py:189-197`) or a hint or task chunk after repeated confusion (`:206-213`), including in reply to a words-only message with no math (`agent/demo_tutor.py:123-129`). They never override correctness (`:136`), and on a verified wrong step below the risk threshold the reply is suppressed and flagged instead (`:220,228-229`). Its files under `eval/` produce research results and are not edited or moved. See `docs/ANALYZER_STATUS.md`.

## Pilot design note (draft, appended 2026-10-22)

A pilot cannot read the cost weights off one comparison: the cost of delayed help (w_D, immediate versus delayed hints) and the cost of interruption (w_I, hint on error versus no hint among students who would have succeeded alone) need separate comparisons, and recovery latency can inform but cannot derive the weights, which stay a stated value judgment. Everything is draft and needs an ethics review, a school sponsor, parent/guardian consent and an adult responsible for the data; see `docs/aria_pace/PILOT_PROTOCOL.md`, section 10.
