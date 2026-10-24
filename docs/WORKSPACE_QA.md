# Workspace verification, October 23, 2026

## Automated checks

The focused suite passed 287 tests. It includes the existing mathematical checker, bank verification, base tutor, frozen Pace parity, concept graph, robustness, flags, harness and teacher-summary tests, plus new workspace and readiness tests.

Readiness coverage includes:

- 81 common-mistake/profile cases across the eight featured problems, each followed by a correct-answer recovery. These include distribution, sign, incorrect roots, unit-circle values, product/chain-rule errors and integration coefficients.
- 90 authored-wrong-answer/profile cases across all 30 bank problems, each followed by a correct-answer recovery. Some inputs overlap the common-mistake matrix; these are not 171 distinct mathematical misconceptions.
- 3,066 policy snapshots over all binary right/wrong histories of lengths 1 through 9 and all three profiles. These check implementation of the preserved gate, not calibration or learning outcomes.
- Nine threshold-boundary checks: immediately below, exactly at and immediately above each frozen threshold.
- Wrong attempts after the decision window, explicit requests below the threshold, no observation for help requests, symbolic answer-leak/false-claim screening, deliberate repeat answers versus stale duplicate callbacks, revisions, failure rollback, completion, retries, session clear and isolation.

For the scripted two-valid-steps / four-wrong-steps example, the first automatic mathematical hint occurs at attempt 3 for Supportive, 4 for Balanced and 6 for Quiet. All settings retain the early/late attempt rule. These positions are properties of this particular history, not a guarantee that each profile always waits that number of attempts.

Run from the project root after installing `requirements-demo-test.txt`:

```bash
python -m pytest tests/test_math_check.py tests/test_demo_workspace.py tests/test_workspace_readiness.py tests/test_demo_bank.py tests/test_demo_tutor.py tests/test_demo_summary.py tests/test_demo_pace.py tests/test_demo_graph.py tests/test_demo_robustness.py tests/test_demo_flags.py tests/test_demo_harness.py -q -p no:cacheprovider
```

Two old expectations were intentionally updated: distribution is a valid step rather than a restatement, and the expanded authored hint pathways appear in summaries. The old single-dropdown UI test now identifies the problem picker by label rather than counting every dropdown.

## Live browser checks

The actual Gradio app was launched with the optional LLM and file logging disabled, not inspected solely through historical screenshots.

Verified a wrong distribution, correcting an earlier step, valid intermediate work without watching messages, a Quiet decision below threshold, an explicit requested hint with the observation count unchanged, switching profiles for subsequent wrong answers, completion after revision, optional reflection, next problem, archived work, teacher navigation and guarded session clear. The teacher report updated without refresh. Later work disappeared from the active branch after revision and remained retained as an earlier branch.

Rendered screenshots were inspected at 320, 375, 414, 768 and 1440 pixels. Page scroll width equalled viewport width at each size. The initial Check step button was within the 850px-high mobile viewport. Equation fonts were 22px on small mobile and 26px at larger sizes. Gradio dark-theme inheritance and stylesheet rewriting issues found during QA were corrected.

Screenshots are in `/Users/karthikmalireddy/Documents/aria/output/tutor-redesign/`. They show the running app rather than a mockup.

## Readiness boundary

This is a locally usable, tested conference demonstration of a supported problem bank. It is not an arbitrary-question tutor, a classroom deployment, or evidence that waiting improves learning. Alternate strategies can exceed authored hint coverage. The 22 non-featured problems have more limited support than the expanded featured set. Optional Ollama behavior, a real network-disconnected run, assistive-technology testing with a screen reader and multi-user load testing remain unverified in this pass.

No research figures, frozen evaluation artifacts, public website, production deployment or GitHub branch were updated. New hint content and interaction behavior must not be represented as having the historical poster's measured performance.
