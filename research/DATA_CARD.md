# ARIA research data card

## Dataset classes

| Class | Examples | Role | Evidence status |
|---|---|---|---|
| Synthetic think-alouds | `eval/data/synthetic_thinkaloud/` | Development and stress tests | Not human evidence |
| Real behavioral logs | ASSISTments/EduAgent adapters | Proxy validation | Behavior proxy only |
| Real tutoring dialogue | Eedi dialogue raw files | Intent development and private annotation | Real text; weak intent labels |
| Developer challenge | `student_language_challenge.json` | Adversarial development | Seen during development |
| Product task bank | generated 100 tasks | Practice and offline episodes | Internally authored; educator review pending |
| Product session logs | local ignored files | Debugging and future approved analysis | Not automatically research-consented |
| Prospective participant data | none collected | Future validation and outcomes | Institutional approval pending |

## Splitting and leakage

- Synthetic sets are split by generation process where possible.
- Real dialogue must be split by student or complete session/intervention.
- Numeric variants and paraphrases of the same task template remain in one
  split.
- Development challenges cannot become final confirmatory tests.
- Outcome problems are independent of tutoring tasks.

## Licensing

Every external source retains its original terms. Eedi dialogue used for intent
development is noncommercial CC BY-NC-SA 4.0 and raw files remain ignored.
Other Eedi resources in the repository may use different terms; they must not
be conflated. Access-controlled or nonredistributable corpora are never copied
into public artifacts.

## Privacy

Synthetic and public/licensed research data do not authorize publication of
future student conversations. Product logs are not research data merely because
they exist. Prospective minor-participant data require the approved protocol,
permission/assent, deidentification, access controls, and retention schedule in
`DATA_MANAGEMENT_PLAN.md`.

## Label quality

- Synthetic state labels are constructed and partly circular.
- Behavioral states are proxies.
- Intent labels are weak supervision.
- Observable reasoning-move labels are pending independent annotation.
- LLM ratings are not educator judgments.

These distinctions must appear in every result table and claim.

