# Limits of ARIA-Pace (read this before quoting a number)

Each item says what the evidence does and does not support. Sources are file:line on branch `aria-pace` unless a branch is named.

## a. Interruption cost is an assumed weight, not a measured harm

The cost J = w_I x (unnecessary interventions) + w_D x (delayed help) is an offline policy simulation (`aria_pace/SPEC.md:135`; weights at `:129-133`). "Unnecessary interruption" is a proxy defined from logs (`SPEC.md:218`), not an observed harm. Whether the Quiet weighting reduces frustration or preserves working memory in students has not been measured; that needs a live human comparison, such as the pilot in `PILOT_PROTOCOL.md` (branch `learning-graph`), which needs approval first. The three profiles are cost weights, not learner types, and the A6 sweep varies the assumed costs (`docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md:5,76`).

## b. The 52% depends on the mastery definition

"3 correct in a row" is the ASSISTments skill-builder's own completion rule for the logged problem sets (`docs/aria_pace/DATA.md:15`), and the headline 52% is under that definition. Under the four other definitions tested (4 in a row; 5 in a row with T=15 and T=20; an EMA rule), the Quiet saving was 8.6%, 32.2%, 45.4% and 9.0% (`docs/aria_pace/MASTERY_SENSITIVITY.md:20,23,26,29`), against 49.8% for the 3-in-a-row control on the same validation students (`:17`). These checks are validation only, exploratory and BKT only, so in our setup the headline saving is fragile to the definition; nothing is claimed about other knowledge-tracing benchmarks, which were not tested.

## c. The risk model is an offline baseline calibration

The parameters were fitted on ASSISTments skill-builder math logs (grade level not stated: `aria_pace/SPEC.md:94`, `docs/aria_pace/RESULTS.md:205`, `docs/aria_pace/DATA.md:140`), not on calculus. The live demo applies them unchanged to algebra, trigonometry and calculus problems (branch `summit-pace-mac`, `agent/demo_pace.py:91-107`, which loads the frozen parameters). Re-calibration on the target domain is required before any deployment.

## d. Status of the language analyzer

The analyzer was evaluated only on synthetic messages, with no independent human ground truth, and accuracy dropped by about 9 points when a different model generated the messages (`docs/AUDIT_20260826.md:163`). The ARIA-Pace pipeline does not depend on its accuracy: in the demo it only adds optional affect flags, which need an explicit feeling word in the student's own words and never override correctness (branch `summit-pace-mac`, `agent/demo_tutor.py:96-105,137-138`; `tests/test_demo_flags.py`). Its files under `eval/` produce research results and are not edited or moved. See `docs/ANALYZER_STATUS.md`.
