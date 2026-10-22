# Scaffolding levels: a three-level gate (design note, future work)

**Status: design note only.** Nothing here is implemented, evaluated or claimed. No file under `agent/` or `ui/` was changed. It describes how a gate with three levels (L0 silence, L1 metacognitive prompt, L2 execution hint) could be built on what the demo already has, what data would be needed to set its two thresholds, and why that is future work.

## 1. What exists today (cited)

- **A single speak / stay-quiet gate.** The frozen ARIA-Pace rule decides silence versus speaking only: a verified wrong step gets a hint when the calibrated BKT risk reaches the profile's `tau` (attempts 3 to 9), otherwise a short flag (`agent/demo_pace.py` and `agent/demo_tutor.py` on branch `summit-pace-mac`; `aria_pace/SPEC.md` Amendments 1 and 2). It does not choose *what kind* of help.
- **The verdicts.** SymPy returns one of `correct | partial | step | restate | wrong | none` (`agent/math_check.py:9`, `:434` priority list, `:516-542` `check_answer`). `derive_state` maps wrong to STUCK, correct to INSIGHT, partial and step to FLOW, restate to PLANNING, and no-math text to STUCK / CONFUSED / UNKNOWN from help-seeking and question flags (`agent/demo_tutor.py:69-90`).
- **The `hint_level` counter.** It starts at 0 per attempt (`agent/demo_tutor.py:107`), is incremented by every reply that is not a reflection (`:179`), and selects the first not-yet-achieved authored move at or after that index (`:182-190`, `moves` from `data/demo_problems.json`, e.g. `calc-01`: "Apply the power rule to 3x^4" then "Apply the power rule to 5x^2"). It is a pointer into authored steps, not a level of help.
- **What a "hint" reply is today.** `PHRASINGS["hint"]` rotates over four templates (`agent/demo_templates.py:28-33`): three are execution-style ("Try this: {do}. What do you get?" and variants that name the authored move or its result) and one is the exact JITAI bank sentence, a metacognitive question such as "What is the first thing the problem is asking you to find?" (`eval/metacognition/jitai.py:419-425`). A seeded round-robin (`Rotor`) picks which one, so **metacognitive prompt versus execution hint is decided by rotation, not by any level rule.**
- **Other intervention types** come from the paper's rule policy: `task_chunking`, `encouragement`, `break_suggestion`, `reflection` (`eval/metacognition/jitai.py:149-260`, bank at `:419-450`); `break_suggestion` is the existing escalation, triggered only after an ignored intervention (`:175-187`).
- **Not found:** any code that escalates through prompt, hint and worked method on a parallel problem; the strings "escalate" and "parallel problem" do not occur in `agent/demo_tutor.py`, `agent/demo_templates.py` or `agent/demo_bank.py`. (The website's Method page describes a three-level escalation for the live tutor; see `RESULTS_v3.md`.)

## 2. A possible mapping

| SymPy verdict (what the student wrote) | Gate input | Level |
|---|---|---|
| `step`, `partial` (verified right) | none | **L0** silence ("watching"), as today |
| `restate` | none | **L0** |
| `wrong` (verified error), risk below the profile bar | risk < `tau_1` | **L0 + flag** (today's flagged notice; never silent, `demo_tutor.py` flagged path) |
| `wrong`, risk at or above `tau_1`, first time on this step | `hint_level == 0` | **L1**: a prompt with no mathematical content (the JITAI bank "hint" sentences) |
| `wrong` again after an L1 prompt, or risk at or above `tau_2` | `hint_level >= 1` or risk >= `tau_2` | **L2**: the authored execution hint (`moves[hint_level]`, "Try this: {do}") |
| `none` with an explicit help request | help flag | **L1**, escalating to **L2** on repetition |
| `none` with "just tell me the answer" | `ask_answer` flag | **L1/L2 as above; never the answer** (the leak check runs on every reply, `demo_tutor.py:_speak`) |
| `correct` | none | reflection (unchanged) |

Implementation sketch (not done): split the single counter into `prompts_given` and `hints_given`; make `PHRASINGS["hint"]` choose by level instead of by rotation; keep `_next_move` pointing at the authored moves for L2 only. `tau_1` would be the existing frozen threshold of the profile; `tau_2` and the repetition rule are new.

## 3. What data would set the two thresholds

**None exists offline today.**
- The 2009 and 2012 ASSISTments logs record student-requested hints (`hint_count`, `first_action`; `docs/aria_pace/DATA.md:32`, `:53`) but no tutor-initiated metacognitive prompts and no outcome of one; the meaning of `first_action` codes is itself only inferred (`aria_pace/SPEC.md:16`, A1.3). They cannot distinguish a prompt from a hint.
- The demo's turn log has verdicts, states and intervention types without student text (`agent/demo_log.py`), and the teacher-moments file holds 8 chosen moments with ARIA's decision, not outcomes; neither contains outcomes of different levels for comparable students.
- `data/real_sessions.jsonl` has 6 chat lines from one person; it is not usable for estimation and is not described further here.
- A replay cannot supply the missing quantity: whether a prompt or a hint helps a student is a causal effect, and the SPEC states that replay cannot say what would have happened had the student been helped (`aria_pace/SPEC.md` section 7, "Limitation, part of the definition").

What would set `tau_1` and `tau_2`: costs for three actions (an unnecessary prompt, an unnecessary hint, a delayed hint) and the effect of each level on the next step, both from a reviewed study with students plus teacher judgments of when each level is appropriate (the open question already on the site's Research page). Until then any value would be a guess, and the A6 sensitivity analysis shows how strongly the answer depends on exactly this kind of assumption (`docs/aria_pace/CLOSED_LOOP_SENSITIVITY.md`).

## 4. Why this is future work

1. The two thresholds cannot be estimated from anything we have (section 3); choosing them now would be unfalsifiable tuning.
2. The existing frozen policy was chosen and tested for one decision (silence versus speaking) on locked splits that are spent (`TEST_LOCK`, `TEST_LOCK_2012`); a new gate needs a new, pre-declared holdout.
3. Each added level multiplies the surface that must pass the leak and math checks and the Safari / rehearsal gates before a demo.
4. The ethical path (interviews, consent, a reviewed pilot) comes first (`docs/DEMO_RUNBOOK.md`, site Roadmap).

## 5. What a first test could look like (not scheduled)

A small within-student crossover in a reviewed pilot where L1 and L2 are assigned at random at comparable verified errors, with pre-registered outcomes (next-step correctness, recovery within three attempts, later help requests) and the same pre-declared margin-plus-interval rule used throughout this project. Teacher ratings of the eight activity moments could seed the choice of `tau_1`.
