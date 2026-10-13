# ARIA Summit-Demo Audit — 2026-10-13

**Scope.** Phase 1, read-only. No product code was changed. This file is the only
addition to the repo. Every codebase claim cites `path:line`; anything I could not
find says **not found**. Where I ran code, the command or probe is described so it
can be reproduced.

**Audited tree:** `NarenS31/aria.kls`, branch `summit-demo` cut from `main` at
`ef5b7d6` (2026-08-26, "Expose per-state confidence and split errors from near-ties").

---

## 0. Read this first — four caveats that change how to read everything below

1. **This clone may be behind your working copy.** `main`'s newest commit is
   2026-08-26; your brief refers to an audit dated 2026-09-01, and the repo's own
   audit file is `docs/AUDIT_20260826.md:1`. Anything you built after Aug 26 and
   did not push is invisible to me. Full history was available (23 commits,
   `git rev-parse --is-shallow-repository` → `false`).
2. **`docs/aria_tutor_demo.html` — not found.** It has never existed on any ref
   (`git log --all -- docs/aria_tutor_demo.html` is empty; `docs/` contains only
   `AUDIT_20260826.md`). The only other branch with "summit demo" in its name,
   `origin/codex/aria-summit-demo`, is a 2026-07-23 ancestor of `main` and also has
   no such file. Item 1 therefore covers `ui/app.py` only.
3. **`data/question_bank.json` — not found in the repo.** `.gitignore:17`
   (`/data/*`) excludes it and no allow-list line re-includes it, although
   `scripts/build_question_bank.py:3` and `agent/reasoning.py:820` both call it
   "committed". I rebuilt it deterministically with `scripts/build_question_bank.py`
   into a scratch directory and audited that copy (identical to what your Mac
   should hold). With a rebuilt copy placed at its expected path, the
   product-reliability tests go from 6 failing to 22/22 passing (§8).
4. **No Ollama in this sandbox.** I could not run `llama3.2:3b` or `llama3.1:8b`,
   so **nothing here measures LLM latency or LLM output quality.** Every number
   below is from deterministic code. LLM latency must be measured on your Mac.

---

## 1. Demo entry point

**`docs/aria_tutor_demo.html`: not found** (see §0.2). I cannot say whether it is
static or calls Ollama.

**`ui/app.py` + `main.py`** (the only runnable UI in the repo):

- `main.py:332` calls `check_ollama()`, then builds the full stack —
  `VectorStore` (chromadb), `LearningGraph`, `ARIAAgent` (`main.py:372`),
  `NudgeScheduler` (`main.py:380`) — and serves Gradio on `127.0.0.1:7860`
  (`main.py:387-393`). First run triggers an interactive intake
  (`main.py:77`, called when no profile exists).
- `ui/app.py:2122` `build_ui()` makes four tabs: Learn (`:2147`), My Progress
  (`:2213`), Research (`:2250`), Settings (`:2298`).
- The think-aloud flow in Learn is `new_think_problem` (`ui/app.py:335-360`) →
  `submit_think_aloud` (`:433-489`) → `ARIAAgent.think_aloud_turn`
  (`agent/reasoning.py:2113`). Wired at `ui/app.py:2350` and `:2355`.
- A free-chat mode also exists: `respond_stream` (`ui/app.py:90`) →
  `ARIAAgent.chat_stream` (`agent/reasoning.py:2746`), which **does** call Ollama.
- **The default think-aloud path makes zero Ollama calls.** It returns template
  strings (§2). The Ollama-backed pieces it could call are all opt-in and
  default off: `dynamic_problem_coaching` (`reasoning.py:506`, `:1841`),
  `dynamic_problem_generation` (`reasoning.py:1947`),
  `deep_language_understanding` (`reasoning.py:2069-2073`).

**Hazards for a stage demo in `ui/app.py`:** the Settings tab exposes
`reset_everything` (`ui/app.py:1503-1523`), which deletes the learning graph, SRS
state, session logs, chroma, checkpoints and emotion logs; the Research tab
launches experiment subprocesses (`ui/app.py:1154-1172`); startup needs the full
memory stack plus a `rumps`/`pyobjc` macOS dependency set
(`requirements.txt:1-4`).

**Which is the better base?** With the HTML demo missing, `ui/app.py` is the only
base, but I would not demo from the full app. Recommendation in §10: a new slim
demo page that reuses the state-panel and banner patterns
(`ui/app.py:369-430`, `:456-482`) and none of the rest.

---

## 2. The timing policy

**Confirmed, exactly as your brief states.**

- `agent/reasoning.py:2124` gates the legacy path behind
  `profile.get("full_research_metacognition", False)`. A repo-wide grep finds the
  read site only; nothing sets it. (`docs/AUDIT_20260826.md:218-226` says the same.)
- The product branch is `reasoning.py:2124-2224`. It calls `_quick_thinking_state`
  (`:2126`; defined `:2061`), builds a `SituationModel` (`:98`, `:2127`), observes
  the previous intervention's outcome via `ClosedLoopInterventionPipeline`
  (`:2139-2151`), and chooses the action with `_situation_policy_response`
  (`:2152`; defined `:1305-1398`).
- The legacy branch (analyzer + `InterventionTimer` + intervention bank) starts at
  `:2226` (comment: "Historical experiment compatibility only") and runs to `:2388`.
- `eval/metacognition/jitai.py` is **not referenced by any code outside `eval/`**
  (grep over `agent/`, `ui/`, `main.py` → no matches). The legacy branch does not
  use it either; it uses `MetacognitiveInterventionGenerator` and `InterventionTimer`
  (`reasoning.py:2276`, `:2326-2348`).

### Does the deployed model already have "don't intervene" behavior?

**Yes, but only one trigger, and it is not the one the demo claims.**

- The sole abstention is `situation.aria_confidence < 0.5` → action `"observe"`,
  `intervened=False`, text "Keep going. I do not have enough task-grounded evidence
  to interrupt yet." (`reasoning.py:1317-1322`).
- Confidence falls below 0.5 only when the utterance carries no recognized
  reasoning move: no moves / off-task / task-meta → 0.28; moves present but no named
  step, no uncertainty, no help-seeking → 0.38 (`reasoning.py:1243-1259`).
- **Every other branch intervenes** (`intervened=True`): `ask_for_plan` (`:1323`),
  `localize_error` (`:1332`), `advance` (`:1351-1372`), `ask_for_justification`
  (`:1373`), `ask_for_verification` (`:1381`), `orient` (`:1390`).
- **Consequence: the deployed policy intervenes on a correct step** (`advance` fires
  when `proposed_step_is_correct is True`, `:1351-1353`) **and stays silent on
  "I don't know" / "just tell me the answer"** (probe results, §7). That is the
  inverse of the demo's claim (quiet in Flow/Planning, active when Stuck/Frustrated).
- The UI already renders a "watching" state when `intervened` is false: banner
  "OBSERVING THIS TURN" (`ui/app.py:456-457`) and the note "No intervention. ARIA
  will use the next turn as additional evidence." (`ui/app.py:478-482`).

### The research policy that does match the demo thesis

`eval/metacognition/jitai.py` (`RuleBasedJITAIPolicy.decide`, `:145-293`):

| State / condition | Decision | Cite |
|---|---|---|
| FLOW (`POSITIVE_STATES = {FLOW, INSIGHT}`, `:25`) | **quiet**, `jitai.no_interrupt_positive` | `:162-173` |
| INSIGHT | intervene (reflection) | `:149-160` |
| PLANNING | **quiet by fall-through**, `jitai.no_action` — not an explicit rule | `:286-293` |
| Negative state after an ignored prompt | break suggestion | `:175-187` |
| FRUSTRATED, persistent or negative tone | encouragement / break | `:189-204` |
| CONFUSED/STUCK ≥ 2 turns | hint or task chunking | `:206-226` |
| First negative state | low-intensity hint/encouragement | `:267-284` |

The policy returns `should_intervene`, `rule_id`, `rationale` (`:107-116`), and
`build_jitai_intervention` (`:477-513`) produces template text with **no LLM**.

### What running it in the live path would take

Input contract: a `JITAIContext` (`jitai.py:77-93`), built with
`context_from_signals` (`:516-545`). The state comes from
`CognitiveStateAnalyzer.analyze` (`analyzer.py:423`), which is **heuristic-only by
default** (`use_llm=False`, `analyzer.py:414`; regex marker banks, `_heuristic` at
`:598`). **The analyzer classifies the student's wording, not mathematical
correctness.** A correct but terse step will not read as FLOW unless the wording
carries marker phrases, and a wrong step stated confidently will not read as
STUCK. That matters because your harness criterion is "stay quiet on correct
intermediate steps, intervene on wrong ones" (Phase 2B).

Files I would touch (no edits under `eval/`; `reasoning.py` already adds `eval/` to
`sys.path` via `_load_metacognition`, `reasoning.py:50-70`, and `_init_metacognition`
loads the analyzer, `:555-574`):

| File | Change |
|---|---|
| `agent/reasoning.py` | New product branch at `:2124-2224` calling analyzer → `JITAIContext` → `decide()`; feed it consecutive-state and ignored-prompt counters (new state; today `consecutive` is only computed in the legacy branch, `reasoning.py:2291`) |
| `agent/` new adapter | Maps analyzer output + counters → `JITAIContext`; maps decision → UI payload incl. `rule_id`, `rationale` |
| `ui/app.py` (or a new demo UI) | Show the policy's rule/rationale; "watching" already exists (`:456-457`) |
| `requirements.txt` | pin deps (§6, §8) |
| tests | new tests; existing `tests/test_situation_model.py` stays valid if the situation model is kept |

---

## 3. Math correctness

**There is no symbolic checking and the LLM does not judge. Correctness is decided
by string/regex/token-overlap heuristics.** `grep -ri sympy` over the repo
(`*.py`, `*.txt`, `*.md`, excluding the old audit) returns nothing, and
`requirements.txt` has no `sympy`.

Where "correct" is decided (all in `agent/reasoning.py`):

- Final answer: whitespace-stripped, lowercased `answer` is a **substring** of the
  whitespace-stripped input → correct (`:1228-1232`; same test at `:2134-2137`).
- Intermediate step: regex `to get <result>` on each keyed step, then substring match
  (`_matched_solution_progress`, `:1123-1143`).
- Operation named: ≥ 2 shared content tokens with a keyed step → correct
  (`:1212-1224`).
- Wrong: only when the student names the exact inverse operation of the keyed first
  step and the same term (`_expected_step_conflict`, `:1075-1121`), or a named step
  token-matches a listed misconception (`:1225-1226`, `:1041-1073`). Otherwise
  "unknown", deliberately never "wrong" (comment at `:1210-1211`).

**Probe results** (real `think_aloud_turn`, problem `math-algebra-01`
`3(x - 4) = 2x + 5`, key `"x = 17"`; `intervened`/`action` as returned):

| Student says | `proposed_step_is_correct` | Tutor action |
|---|---|---|
| `x = 17` | True | advance |
| `x=17` | True | advance |
| `17` | None | observe (silent) |
| `so x equals 17` | None | observe (silent) |
| `x = 34/2` | None | ask_for_justification |
| **`x = 170`** (wrong) | **True** | **advance** ("matches the task model") |
| **`x = 17.5`** (wrong) | **True** | **advance** |
| `x = -17` (wrong) | None | ask_for_justification |

The wrong answers `170` and `17.5` are accepted because `"x=17"` is a substring of
`"x=170"`. On `math-advanced-07` (`f'(x) = 6x - 5`): exact → correct but **silent**
(confidence stays 0.28, `:1243-1244`); `-5 + 6x` → unknown; `6x - 5 + 0` → correct;
**wrong `6x + 5` → silent, not flagged.**

**Where the answer lives:** `ctx["answer"]`, a free-text **string**, set in
`_normalise_problem_ctx` (`reasoning.py:867-886`, `:880`). Sources: the bank loaded
by `_load_question_bank` (`:818-837`), the static `_FALLBACK_PROBLEMS`
(`:609`, `:810-815`), or an LLM-generated problem when
`dynamic_problem_generation` is on (`:1947`, generation call `:1966-1976`). The
generated path asks the model to author its own answer key — no verification.

---

## 4. Answer leakage

**The check** (`agent/intervention_pipeline.py:431-435`):

```python
candidate_answer = " ".join(_tokens(text))
if answer_norm and answer_norm in candidate_answer and answer_norm not in student_norm:
    valid = False; reasons.append("reveals_answer"); score -= 100
```

`_tokens` (`:44-48`) keeps `[a-z0-9.]+` runs of length ≥ 2 that are not stopwords.
So the "answer" becomes a space-joined string of those tokens, tested as a
contiguous substring of the candidate's token string.

**Probe** (exact replica of the condition, answer `12x^3 - 10x + 7`; its token
form is `'12x 10x'` because the single-character `3` and `7` are discarded):

| Candidate text | Caught? |
|---|---|
| `12x^3 - 10x + 7` (identical) | yes |
| **`7 - 10x + 12x³`** (your example) | **NO** — tokens `10x 12x`, order flipped, `³` is dropped |
| `12*x^3 - 10*x + 7` | **NO** — tokens `12 10 7.` |
| `12 x^3 - 10 x + 7` | **NO** — tokens `12 10` |
| `12x^3 + 7 - 10x` | yes, by coincidence (the `7` is discarded so the order survives) |
| `-2x^2 sin(2x) + 2x cos(2x)` vs key `2x cos(2x) - 2x^2 sin(2x)` | **NO** |

**Short answers disable the check entirely.** If the normalized answer is empty the
`answer_norm and ...` guard short-circuits. `x = 5` → `''`; `7` → `''`.
In the 50-problem math bank, **25 answers (50%) normalize to empty** (31 of all 100),
e.g. `math-algebra-02` `x = 6`, `math-functions-02` `2`. For those problems the
filter never fires.

**Additional finding — the filter is not on the product path.** It lives in
`score_candidates` (`intervention_pipeline.py:369-486`), reached only through
`pipeline.select` (`:488`, which calls it at `:501`), whose only caller is `reasoning.py:1868` inside
`_problem_aware_coaching_response`. That is called at `reasoning.py:2354`, in the
**legacy** branch (after the product `return` at `:2224`). The product path never
calls it. It stays leak-free today only because its replies are fixed templates;
though note `orient` prints the first keyed step verbatim (`reasoning.py:1390-1397`),
and keyed steps can contain intermediate results (e.g. "Distribute 3 to get
3x - 12 = 2x + 5.").

---

## 5. Problem bank

- **Not committed** (§0.3). Built from `scripts/build_question_bank.py`
  (`OUTPUT` at `:15`, `build()` at `:473`, asserts exactly 100 at `:486`, writes at
  `:491`); loaded by `reasoning.py:818-837`.
- **100 tasks**: Math 50 / English 30 / Science and Coding 20. Math topics, 10 each:
  Algebra, Functions and Systems, Geometry, Statistics and Probability, Advanced Math.
- **Format**: JSON list of objects: `id, subject, topic, difficulty, task_type,
  problem, answer (string), solution_steps[], key_ideas[], common_misconceptions[]`
  (`build_question_bank.py:18-45`).
- **Levels**: every task has `difficulty: "medium"` (100/100). There is no
  algebra/trig/calculus tagging. Keyword scan of the 50 math tasks: **trig 3**
  (`math-geometry-03`, `-09`, `math-advanced-05`), **calculus 1**
  (`math-advanced-07`: derivative of `3x^2 - 5x + 2`), quadratic-style 3.
- **Your two demo problems are not present** (`3x^4 - 5x^2 + 7x - 9`,
  `x^2 cos(2x)`; regex search → no hits). The bank cannot supply a 30-problem
  algebra/trig/calculus harness set; most of that would be new authoring.
- Steps are prose (`"Distribute 3 to get 3x - 12 = 2x + 5."`), several are generic
  ("Use inverse operations while keeping both sides balanced."), none is
  machine-checkable.

---

## 6. Models and latency

| Use | Tag | Cite | Options / timeout |
|---|---|---|---|
| Agent default (chat, problem-aware coaching, problem generation) | `llama3.2:3b` | `reasoning.py:508` | chat `num_predict=256`, temp 0.7 (`:2629-2634`, `:2796-2802`); coaching 240 (`:1843-1850`); generation 260 (`:1967-1976`); **no timeout on any of these** |
| Deep intent understanding (opt-in) | `llama3.2:3b` | `student_understanding.py:313` | `ollama.Client(timeout=2.5)`, `num_predict=150` (`:289-297`) |
| Nudges | `llama3.2:3b` | `nudge.py:31` | `num_predict` 80 / 250 (`:77`, `:301`) |
| Analyzer LLM fallback (off by default) | `llama3.2:3b`, env `ARIA_META_MODEL` | `analyzer.py:79`, `:414` | n/a |
| Insight narration | `llama3.1:8b`, falling back to `llama3.2:3b` | `ui/app.py:844-853` | `num_predict=200`, no timeout |

- **Loading:** per request, not at startup. Nothing preloads an Ollama model
  (grep for `warm`/`keep_alive=0`/`ollama.generate` finds only
  `warm_student_intent_model`, which loads a local sklearn pickle, not an LLM:
  `reasoning.py:507`, `student_intent.py:104`). Every call passes `keep_alive=3600`,
  so the model stays resident for an hour *after the first call*; the first call
  pays the cold load.
- **Timeouts:** the only Ollama timeout in the code is the 2.5 s on the opt-in deep
  path. The `timeout=3` at `nudge.py:491` is an `osascript` call, not Ollama.
- **Authors' own latency note:** the code comment at `reasoning.py:504-505` says
  local-model candidate generation stays opt-in "because a 5–12 second wait is
  unacceptable". That is their statement, not something I measured.
- **Product think-aloud path latency (measured here, no LLM):** 3–23 ms per turn
  after warm-up; the very first call took 1,869 ms (not isolated; consistent with
  the lazy sklearn load at `student_intent.py:91-101`).
- **Dependency gap:** `student_intent.py:98-100` loads `models/student_intent.joblib`
  with `joblib` and silently sets the model to `None` on any exception.
  `requirements.txt` lists neither `joblib` nor `scikit-learn`, so a clean install
  degrades the intent classifier without any error. On my box the pickle (built
  with scikit-learn 1.9.0) loaded under 1.9.1 with `InconsistentVersionWarning`.
  An offline demo machine needs pinned versions.

---

## 7. Input handling

Traced through `submit_think_aloud` (`ui/app.py:433-489`) and the real
`think_aloud_turn`. Probe: problem `math-algebra-01`, fresh agent per input.

| Input | What happens | Cite |
|---|---|---|
| **Empty / whitespace** | The UI returns "Type (or record) your reasoning first" and never reaches the agent (`ui/app.py:438-439`). Called directly, the agent does not crash and returns `observe` / "Keep going…", confidence 0.28 | `reasoning.py:1317-1322` |
| **2,000-char paste** | No length cap anywhere (no `max_len`/truncation in `student_understanding.py`, `student_intent.py`, `reasoning_moves.py`). No crash, 23 ms. A repeated sentence containing "distribute the 3 first" is scored **correct, confidence 0.83 → `advance`**. The quote anchor truncates to 8 words (`reasoning.py:922`) | `reasoning.py:1212-1224` |
| **LaTeX** (`\frac{d}{dx}(…)=12x^{3}-10x … 3(x-4)=3x-12`) | Not understood: `observe`, confidence 0.28, even though it contains a correct step | probe |
| **Off-topic** ("what's the weather…") | `observe` / "Keep going…" (silent) | probe |
| **Gibberish** | `observe` (silent) | probe |
| **"I don't know"** | `observe` (silent), confidence 0.28 | probe |
| **"just tell me the answer"** | `observe` (silent), confidence 0.28 | probe |
| **Emoji/unicode minus** (`3x − 12`, ✅, `x²`) | No crash; scored correct → `advance` | probe |
| **Exception inside the agent** | Caught; UI shows "could not analyze that turn… try again" and preserves the text | `ui/app.py:441-448` |

Two of these matter directly for the demo: **"I don't know" and "just tell me the
answer" — the two clearest Stuck/Frustrated utterances — get silence today.**
The chat path (`chat_stream`) falls back to `offline_coaching_response` if Ollama
errors (`reasoning.py:2806-2808`), but it has no timeout, so a *hung* model blocks
rather than errors.

---

## 8. Existing tests

Ran on Python 3.11, scratch venv, `PYTHONDONTWRITEBYTECODE=1`, repo left clean
(`git status --short` empty after every run).

| File | Result (each file in its own process) |
|---|---|
| `tests/test_situation_model.py` | 15 passed |
| `tests/test_intervention_pipeline.py` | 8 passed |
| `tests/test_interaction_grain.py` | 22 passed |
| `tests/test_learner_model.py` | 3 passed |
| `tests/test_reasoning_moves.py` | 6 passed |
| `tests/test_student_intent.py` | 7 passed |
| `tests/test_product_reliability.py` | **22 passed with the rebuilt bank in place; 6 failed without it** |
| `tests/test_educator_ratings.py`, `test_power_analysis.py`, `test_research_stats.py`, `test_rq2_analysis.py` | 3 / 3 / 3 / 11 passed |
| `tests/test_calibration_scoring.py` | **7 failed, 7 passed** — needs `eval/data/annotation/annotation_private_key.json`, deliberately gitignored (`.gitignore:56`); expected on any clean clone |
| `eval/tests/test_full_flow.py`, `test_jitai_logging.py` | collection error as-is (`import metacognition` needs `eval/` on the path); **4 and 8 pass with `PYTHONPATH=eval`** |

**The six `test_product_reliability` failures without the bank** are
`test_committed_question_bank_has_100_reviewable_tasks` (`:96`),
`test_generated_problem_has_answer_key_context` (`:79`),
`test_try_another_problem_does_not_repeat_immediately` (`:106`),
`test_english_bank_task_uses_its_specific_coaching_move` (`:156`),
`test_inverse_operation_coaching_does_not_state_the_operation` (`:333`),
`test_completed_intermediate_step_advances_the_coaching` (`:351`). All six pass
once the bank exists.

**Running `pytest tests` in one process is not reliable** (11 failed, 71 passed,
5 errors): `tests/test_calibration_scoring.py:18` does
`sys.path.insert(0, ".../eval")`, which makes `eval` resolve to the nested
`eval/eval/` package, so later modules fail with
`No module named 'eval.analyze_educator_ratings'`. Each file passes alone. Phase 2
should run tests per file.

There is **no test that touches the live Ollama path**; UI tests use a `FakeAgent`
(`tests/test_product_reliability.py:10-40`).

---

## 9. Where the repo differs from your Phase 2 assumptions

1. **"The LLM only phrases hints"** assumes an LLM in the loop. The product path has
   none; replies are templates. Adding LLM phrasing is new surface area (and the
   authors measured it as slow, §6).
2. **Phase 2B's timing check cannot pass on today's policy** (§2, §7): it intervenes
   on correct steps and stays silent on wrong or stuck turns.
3. **The harness's "math-error rate" and "answer-leak rate" are only non-trivial if
   an LLM writes the replies.** With templates they are zero by construction.
4. **Problem data is not versioned** (§0.3, §5) and has no levels.
5. **No pinned/complete requirements** (§6): `sympy`, `scikit-learn`, `joblib` are
   absent.

---

## 10. Proposed Phase 2 plan (pending your approval — nothing below has been started)

See the chat message that accompanies this file for the plan with estimates and the
decisions I need from you.

---

## Reproduction

```bash
git clone https://github.com/NarenS31/aria.kls && cd aria.kls && git checkout ef5b7d6
python3.11 -m venv v && v/bin/pip install ollama langgraph langchain-core chromadb networkx \
    apscheduler reportlab pytest sympy joblib scikit-learn numpy matplotlib pyyaml gradio
python3.11 scripts/build_question_bank.py          # writes data/question_bank.json (gitignored)
for f in tests/test_*.py; do v/bin/python -m pytest "$f" -q; done
PYTHONPATH=eval v/bin/python -m pytest eval/tests -q
```
