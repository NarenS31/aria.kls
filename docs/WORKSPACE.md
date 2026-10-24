# ARIA math workspace

## Entry points

The default `./launch_aria.sh` opens `ui/demo_app.py` -> `ui/workspace_app.py` on localhost:7861. It disables file logging. Use Python 3.11 or 3.12 and `requirements-demo.txt`; no Ollama, model weights, training, vector database or desktop extensions are needed for the authored workflow.

`ARIA_PYTHON=/absolute/path/to/python ./launch_aria.sh --port 7861` selects an existing environment. The optional module flag `--llm-path` retains the existing guarded Ollama path; it is off by default. The `--all-problems` flag is retained for compatibility; the new topic picker already exposes all 30 problems, initially showing the eight featured problems.

`./launch_aria.sh --legacy` opens the older `main.py` / `ui/app.py` learning-and-research interface. Its broader requirements and storage behavior are unchanged. It is not reachable from the student workspace. Separate launchers are not an authentication boundary: do not expose either server publicly without a separate deployment/security review.

## What changed

- The initial view is a real problem, chronological math steps, a composer and contextual help. There are no waiting-message bubbles or student-facing inferred cognitive-state badges.
- Equations render through Gradio's bundled KaTeX assets. Plain-text input remains the authoritative checker input. Symbol controls insert supported notation at the cursor. No voice, photo or drawing feature is simulated.
- Requested hints use the real screened tutor path, bypass automatic waiting and do not add a mathematical observation. Metadata distinguishes automatic and requested support.
- Hint selection follows verified work rather than hint count. Equation progress compares the sides of authored work states, not just their solution sets, so equivalent equations do not all count as completed operations.
- All eight featured problems received additional authored support. Algebra distribution, collection and isolation are separate operations. The original featured derivative now includes the remaining linear/constant terms; product/chain/integral problems have follow-through prompts. These are authored alternatives, not a general generative tutor.
- The equation checker now distinguishes a written distributed equation from a literal restatement of the original. Previously SymPy simplification could make both look identical and omit a valid observation. This changes product-level observation sequences; it does not change the frozen policy itself.
- The analyzer's standalone-import fallback now appends its path instead of prepending it, avoiding an `eval/ui` collision with the product UI.

## History and revisions

Each browser session gets a separate Workspace. It retains step and hint events, the profile that produced each event, prior attempts and replaced branches in memory. Revising step N preserves the replaced branch, truncates the active branch from N onward, then replays the retained prefix before checking the revision. Replays do not duplicate file-log records. A failed check restores a checkpoint rather than replaying through a failing checker.

Duplicate event callbacks with an outdated revision are rejected. A deliberate identical answer at a newer revision remains a new attempt. The UI serializes workspace mutations; completion prevents repeated final-answer submissions and hint requests.

Retry starts a fresh per-problem timing tracker while retaining the exploratory session graph and archiving the previous attempt. Next problem behaves similarly. Clear session requires confirmation and removes the session work, archives, branches and graph. These estimates are not durable student mastery records.

The teacher report is derived automatically after mutations. Its follow-up language describes observable incorrect steps and subsequent verified steps, not diagnoses or causal effects of hints. Previous work is available in the student sidebar; detailed policy and concept estimates remain optional disclosures.

## Research semantics

No frozen configuration, calibration, threshold, cost weight, policy window, evaluation dataset or historical reference screenshot was edited. Quiet/Balanced/Supportive retain their frozen thresholds (57%, 19%, 9%) on verified observations 3–9. Wrong observations before or after that window receive support. The demo uses pooled BKT parameters, not the evaluated per-skill model.

Changed hint content, revised branches, the distribution/restatement fix and explicit help events are product changes. Existing poster statistics are not evaluations of this revised interaction. The tests check implementation behavior, not student learning, calibration on new questions or classroom efficacy.

## Data and limits

The default launcher stores no transcript files. Work text, hints, revisions and reflections are held in server-session memory. Gradio State expires four hours after its last update; restarting the server or starting a new browser session does not restore work. The direct module launcher logs derived numeric/identifier metadata by default; use `--no-log` to disable it. No cloud database, analytics or external model is enabled by the default workspace.

The full bank still contains authored checks and a limited number of authored hints. The non-featured problems were not given the same expanded hint pathways; they may exhaust useful help early. Even featured problems do not cover every valid strategy. Do not call this support for arbitrary questions. Symbolic equivalence does not establish that the student's reasoning is sound, and partial-answer recognition is not proof of mastery.

Network-disconnected operation must be tested separately before claiming offline readiness. A localhost URL alone is not that verification. The optional model path was not enabled during product QA.
