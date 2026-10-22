# Language analyzer: status and role

- **Evidence:** evaluated only on synthetic messages, with no independent human ground truth; a different generator lowered accuracy by about 9 points (`docs/AUDIT_20260826.md:163`). The site reports this as "same-generator synthetic accuracy", not as accuracy on students.
- **Role in the demo:** the analyzer's marker scores can only add optional frustrated / confused / help-seeking flags, and those need an explicit feeling word in the student's own words. They never decide correctness, which comes from the SymPy check, and they do not gate when ARIA speaks (branch `summit-pace-mac`: `agent/demo_tutor.py:96-105`, `:137-138`; `tests/test_demo_flags.py`).
- **Not touched:** its files under `eval/` produce research results and are neither edited nor moved.
- **See also:** `docs/aria_pace/LIMITS.md` (section d).
