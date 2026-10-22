# Language analyzer: status and role

- **Evidence:** evaluated only on synthetic messages, with no independent human ground truth; a different generator lowered accuracy by about 9 points (`docs/AUDIT_20260826.md:163`). The site reports this as "same-generator synthetic accuracy", not as accuracy on students.
- **ARIA-Pace:** the gate and its published results do not depend on the analyzer (they use the SymPy verdict and the BKT risk).
- **Role in the demo** (branch `summit-pace-mac`): the analyzer's marker scores produce flags. A frustrated or confused flag needs an explicit feeling word in the student's own words (`agent/demo_tutor.py:103-104`). A flag sets the state to FRUSTRATED or CONFUSED (`:137-140`), and the rule-based policy can then reply with `encouragement` or `break_suggestion` after repeated frustration (`eval/metacognition/jitai.py:189-197`) or a hint or task chunk after repeated confusion (`:206-213`). A words-only message with no math can also be classed as help-seeking or a question and get a reply (`agent/demo_tutor.py:109,123-129`). Flags never override correctness (`:136`), and a verified wrong step below the risk threshold is flagged, not spoken (`:220,228-229`).
- **Not touched:** its files under `eval/` produce research results and are neither edited nor moved.
- **See also:** `docs/aria_pace/LIMITS.md` (section d).
