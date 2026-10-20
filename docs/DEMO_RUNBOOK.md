# ARIA summit demo: runbook

Build: branch `summit-pace-mac`, tag `summit-2026-10-24-pace` (Khan Lab School summit, 2026-10-24; this is `summit-2026-10-24` plus ARIA-Pace, see section 10). The original tag `summit-2026-10-24`
(branch `summit-demo`) is the build without ARIA-Pace and was not moved. After the pace tag the only allowed code change is a fix for a crash found in rehearsal; each one is listed in the change log at the bottom.

## 1. Launch (exact)

```bash
cd ~/aria
GRADIO_ANALYTICS_ENABLED=False /usr/bin/python3 -m ui.demo_app
```

Then open **http://127.0.0.1:7861** in Chrome. The terminal shows `Running on local URL: http://127.0.0.1:7861`; first paint takes about 5 s from a cold start
(2026-10-20, 5 launches, process start to first HTTP 200: min 4.70 s, median 4.83 s, max 5.16 s; 3.8 s was measured on 2026-10-15 before ARIA-Pace, and the offline-sandbox rehearsal in section 7 took 6.1 s). Port 7861 is the default (`ui/demo_app.py:308`), the server binds to 127.0.0.1 only (`ui/demo_app.py:318`).

Why exactly this command:

- **`/usr/bin/python3`, not `python3`.** `/usr/bin/python3` (3.9.6) is the interpreter that has sympy 1.14.0, gradio 4.44.1 and ollama installed. Plain `python3`
  here is Homebrew 3.14.7 with **neither gradio nor sympy**, and Homebrew `python3.11` has gradio 6.19.0 but **no sympy**, so the demo does not start with either (`ModuleNotFoundError`). `launch_aria.sh` uses
  `python3.11` for the main ARIA app; it is *not* the demo launcher.
- **Working directory must be the repo root, and it must be `-m ui.demo_app`.** From any other directory: `No module named 'ui'`. `python ui/demo_app.py`
  fails with `No module named 'agent'`. (Both checked.)
- **The `ui` trap.** Importing anything from `eval.metacognition` runs `eval/metacognition/analyzer.py:51-52`, which puts the `eval/` directory first on `sys.path`;
  after that `import ui` resolves to `eval/ui/__init__.py`, not `ui/`. `python -m ui.demo_app` is immune because Python resolves `ui.demo_app` *before* anything imports
  `eval`. So: never start the demo any other way, and any helper script must load `ui/demo_app.py` by file path (as `scripts/demo_screens.py`, `scripts/demo_rehearsal.py`
  and the tests do), never `import ui` after importing `agent.demo_tutor`.
- `GRADIO_ANALYTICS_ENABLED=False` stops Gradio's usage telemetry when online. Startup works offline either way (3.3 s with it on, 4.1 s with it off, both measured offline).

The default launch shows **only the 8 featured problems**, has the **LLM path OFF** (section 5) and starts on the **Balanced** pace profile (`--pace-profile quiet|balanced|supportive` changes only the starting choice; the page's selector changes it live).

## 2. Presenter checklist

1. Wi-Fi off (section 3). 2. Launch (section 1). 3. Open the page, confirm the badge says **READY**. 4. Pick the problem, run the scripted flow (section 6).
5. Open the **Teacher summary** tab to show the timeline. 6. **Reset** before the next visitor (section 4).

The page has two tabs: **Student** (badge, chat, text box, Reset) and **Teacher summary** (state timeline, hints given, where the student got stuck; no chat transcript).
Badge meanings: READY (nothing yet), WATCHING (ARIA read the work and chose to say nothing), STEPPING IN (ARIA is speaking; the line under it says which policy rule decided). The page also has a **profile selector** (Quiet / Balanced / Supportive, default Balanced) and a **live line** under the badge: risk of getting stuck, the risk the profile needs to step in, and why (section 6).

## 3. Wi-Fi off

Nothing in the demo needs the internet: no model download, no remote API, no CDN script. With the LLM path off it does not even need Ollama.

What I tested, and what I did not:

- **Tested:** a cold start of the UI under a macOS sandbox that denies every non-loopback outbound connection (`https://example.com` fails, `1.1.1.1:443` is denied, localhost
  works), with a real Chrome whose DNS is blocked for everything except localhost (a fetch to `fonts.googleapis.com` fails instantly). All 8 featured problems pass in
  that mode (section 7). Command: `/usr/bin/python3 scripts/demo_rehearsal.py --offline`.
- **Not tested by me:** physically switching the Wi-Fi off. Doing it would also have cut the connection this build session runs on. **Do that final check yourself** (step below).
- **What you will see offline:** the browser console logs two errors, `Unable to preload CSS ... fonts.googleapis.com` (Source Sans Pro, IBM Plex Mono). They are cosmetic: the page
  falls back to system fonts. Gradio also prints harmless client warnings ("Too many arguments provided for the endpoint", "Attempted to select a non-interactive or hidden tab").
  Nothing else fails.

Do this at the venue, in this order: switch Wi-Fi off (menu bar) → `curl -m 3 -I https://example.com` must **fail** → launch (section 1) → run problem 1's flow → Teacher summary → Reset.
(If you start the server first and switch Wi-Fi off afterwards it also works; nothing is fetched after page load except the fonts above.)

## 4. Reset between visitors

- Click **Reset (new attempt)** on the Student tab. The badge returns to READY, the chat empties, the Teacher summary shows "No attempt to summarise yet", and ARIA forgets
  the attempt (hints, steps achieved, state). The problem stays selected. Takes under a second.
- To switch problem, use the dropdown; that also starts a fresh session.
- If the page looks wrong or frozen: reload the browser tab (F5; new session). Last resort: stop the server (Ctrl-C in its terminal, or `kill $(lsof -ti tcp:7861)`) and relaunch (about 5 s).

## 5. The LLM flag, and what happens when a model hangs

**Decision: leave `--llm-path` OFF for the summit** (it is off by default: `ui/demo_app.py:306`). Do not enable it unless the check below passes on the demo laptop *and*
you decide you want free-form questions answered in the model's words.

Rule applied: 50+ live calls on the demo laptop with p95 < 5 s and 0 leaks. Numbers below were measured on the 2026-10-15 build (before ARIA-Pace) and **have not been re-run** with it; the LLM path only answers free-form questions, which the pace gate does not touch, but re-run the check before enabling it. Numbers measured on the build machine (Apple M4, macOS 26.4.1, `llama3.2:3b` through the local Ollama), 3 full
harness runs, 90 free-form questions, 1,155 turns in total:

| Measure | Result |
|---|---|
| LLM calls | 90 (88 used; 2 rejected by the symbolic leak/false-claim check and replaced by a template, as designed) |
| LLM call latency | min 1.07 s, median 2.00 s, **p95 2.62 s**, max 2.98 s |
| Answer leaks / math errors / crashes / timeouts | **0 / 0 / 0 / 0** over all 1,155 turns |

The criteria are met on that machine, **but I cannot tell whether that machine is the demo laptop**, so the decision stays OFF. To re-check on the demo laptop (about 3 minutes):
`cd ~/aria && /usr/bin/python3 scripts/demo_harness.py --backend demo-llm --strict --log /tmp/llm.jsonl` three times; each prints leaks, math errors, crashes and latency, and the
LLM outcomes (`ok` / `rejected`). To turn it on you must **restart**: `... -m ui.demo_app --llm-path`. It cannot be flipped in the page, and it must not be flipped live.

If the model hangs or Ollama is not running (only matters with `--llm-path`):

- **Warm-up fails at launch:** the terminal prints `LLM path: Ollama not reachable, using templates only` (`ui/demo_app.py:317`) and the demo carries on with templates.
  The presenter sees nothing different. (Warm-up waits up to 60 s for a cold model load: `agent/demo_tutor.py:47`.)
- **A call runs past 10 s** (`agent/demo_tutor.py:44`): that visitor waits up to 10 s once, then gets the normal template reply; the LLM is then skipped for 20 s (`:48`) so the next
  visitors never wait. Replies are capped at 80 tokens (`:45`) and cut at the last full sentence.
- **Any LLM answer that states the answer or a false identity** is dropped and replaced by a template (the 2 rejections above).
- **Symbolic checker takes too long** (3 s budget, `agent/demo_tutor.py:49`): the presenter sees the quiet line *"Watching. I couldn't read that one. Try writing your next step as plain math."*
  (`agent/demo_templates.py:70`) and no verdict is guessed.

## 6. The 8 featured problems and the "same student, three wrong answers" flow

Type these exactly (the answers were re-derived by SymPy at load time; a typo in the bank stops the server with a clear error). **Type, never paste.**

**The point of the demo now:** the same wrong answers, typed three times, get a different reaction from each pace profile. Profiles are cost weights (how much an unnecessary
interruption costs against a late hint), not learner types. ARIA-Pace keeps a running estimate of the risk that the student gets stuck on this problem (a BKT estimate from the right/wrong
verdicts only) and a wrong step gets a hint only when that risk reaches the profile's bar: **Quiet 57%, Balanced 19%, Supportive 9%** (loaded from `aria_pace/results/frozen_val.json`).
The bar applies at attempts 3 to 9 (an attempt = a message SymPy marked right or wrong; "I don't know" and restating the problem do not count). **Attempts 1 and 2 keep the old rule: a wrong step always gets a hint.**
A wrong step below the bar is never ignored: ARIA shows a short "Not quite yet, holding the hint" note (badge WATCHING, no hint, no answer).

**Flow, per problem** (the profile selector sits under the problem list; **Balanced is the default**):

1. Type the **correct step** (message A below).
2. Type the **same correct step again**. Both are WATCHING in every profile.
3. Type the **wrong answer** (message B) **three times**. These are attempts 3, 4 and 5; the risk reads 15%, 25%, 46%.
4. Repeat 1 to 3 after **Reset** with the next profile (Reset keeps the profile selection; changing the selector mid-attempt also works and applies from the next message).

| Wrong answer # (attempt) | Risk of getting stuck | Quiet (needs 57%) | Balanced (needs 19%) | Supportive (needs 9%) |
|---|---|---|---|---|
| 1st (attempt 3) | 15% | WATCHING | WATCHING | **STEPPING IN** |
| 2nd (attempt 4) | 25% | WATCHING | **STEPPING IN** | STEPPING IN |
| 3rd (attempt 5) | 46% | WATCHING | STEPPING IN | STEPPING IN |

Quiet never steps in during these three; a 4th wrong answer takes the risk to 70%, which is over its 57% bar, so it steps in then. Every row shows the live line under the badge, e.g.
`risk of getting stuck 25% | Balanced needs 19% to step in | STEPPING IN - wrong step and the risk is high enough: hint given`. The three numbers for the table come from the frozen files, and
`scripts/demo_rehearsal.py` stops with an error if they ever stop giving these badges.

Other messages are unchanged: `I don't know` (ARIA speaks, any profile, nothing counted), and the correct answer (ARIA asks the student to reflect). The correct answer is **message C**; type it only if a visitor has
reached it, or at the end for the Teacher summary, and never read it out first.

| Id | Level | Problem | A: correct step (type twice) | B: wrong answer (type 3 times) | C: correct answer |
|---|---|---|---|---|---|
| `alg-01` | intro | Solve for x: 3(x - 4) = 2x + 5 | `x - 12 = 5` | `x = 7` | `x = 17` |
| `alg-05` | core | Solve for x: x^2 - 5x + 6 = 0 | `(x - 2)(x - 3) = 0` | `x = 1` | `x = 2 or x = 3` |
| `trig-01` | intro | Evaluate sin(pi/6) + cos(pi/3) | `sin(pi/6) = 1/2` | `My answer is 3/2` | `My answer is 1` |
| `trig-04` | core | Simplify sin(x)cos(x)/(1 - sin(x)^2) | `1 - sin(x)^2 = cos(x)^2` | `My answer is sin(x)` | `My answer is tan(x)` |
| `calc-01` | core | Differentiate f(x) = 3x^4 - 5x^2 + 7x - 9 | `12x^3 - 10x` | `My answer is 12x^3 - 10x - 9` | `My answer is 12x^3 - 10x + 7` |
| `calc-02` | advanced | Differentiate f(x) = x^2 cos(2x) | `2x cos(2x)` | `My answer is 2x cos(2x) + 2x^2 sin(2x)` | `My answer is -2x^2 sin(2x) + 2x cos(2x)` |
| `calc-06` | advanced | Differentiate f(x) = (x^2 + 1)^3 | `the outer derivative is 3(x^2+1)^2` | `My answer is 6x(x^2+1)^3` | `My answer is 6x (x^2 + 1)^2` |
| `calc-07` | core | Find the integral of 6x^2 - 4x + 1 dx | `2x^3 - 2x^2` | `My answer is 6x^3 - 4x^2 + x` | `My answer is 2x^3 - 2x^2 + x` |

Teacher summary after a run of the flow: "Not solved yet", the problem, the stuck stretch, and no student text. Balanced and Supportive differ **only** on the first wrong answer; if a visitor asks
why, that is the 15% reading falling between their 9% and 19% bars.

What this does **not** show: it measures nothing about learning. The bars were chosen offline on logged ASSISTments data (a replay), the risk is a population-level BKT estimate with pooled parameters
(this demo has no skill labels), and the three profiles are cost settings. Do not describe them as types of student.

## 7. Rehearsal (cold start, network blocked): 2026-10-20

`/usr/bin/python3 scripts/demo_rehearsal.py --offline --theme light` → raw results in `docs/rehearsal_results.json`, Teacher-summary screenshots in `docs/demo_screens/rehearsal/`.
(The 2026-10-15 rehearsal covered the old flow. `docs/demo_screens/light/` and `dark/` were regenerated on 2026-10-20 with `scripts/demo_screens.py` from this build: they show the profile selector and the live line, and follow the same-student flow on `alg-01` under Balanced.)

- Cold start to first answer under the offline sandbox: **6.1 s** (5.9 s in the first run of 2026-10-20; 3.8 s on 2026-10-15). Without the sandbox, 5 launches gave median 4.83 s (section 1). Loading the ARIA-Pace files adds under 0.1 s (measured); the rest of the difference to 2026-10-15 was not investigated.
- **8/8 problems passed** (120 turns: 8 problems x 3 profiles x 5 messages). Browser round trip per message: median 226 ms, p95 406 ms.
- Per problem and per profile, all of these had to hold: the badge sequence Quiet W W W W W, Balanced W W W S S, Supportive W W S S S; the live line shows the right risk % and "needs N% to step in"; a flagged wrong answer shows its
  visible note; Teacher summary says "Not solved yet", shows the problem, has the four sections, contains none of the student's messages; Reset returns to READY with an empty chat; the summary is cleared by Reset.
- Console: the same cosmetic messages as before (two font preloads, "Too many arguments provided for the endpoint", "Attempted to select a non-interactive or hidden tab").

**Browsers.** Chrome: verified (driven through the real UI; both themes). **Safari: NOT verified.** Driving Safari needs *Safari → Settings → Developer → Allow remote automation*, a setting only you can switch
on, so I could not script it. Do this once by hand (2 minutes): open http://127.0.0.1:7861 in Safari, then check (a) the badge text is readable in READY / WATCHING / STEPPING IN, (b) the chat keeps the newest
message in view after the 5-message flow of section 6, (c) the Teacher summary tab is readable, (d) Reset empties the chat, (e) the profile selector (3 radio buttons) and the live line under the badge are readable and update when you click a profile, in light and dark mode. Run one problem per profile: correct step twice, the same wrong answer three times, Reset (section 6). Not verified on 2026-10-20 either: Safari's "Allow JavaScript from Apple Events" and screen capture are off on the build machine. If anything is off, use Chrome. The page uses nothing newer than flex layout with `gap` (Safari 14.1+).

## 8. The turn log

- File: `data/demo_logs/turns.jsonl` (`agent/demo_log.py:16`), created on first turn, rotated to `turns.jsonl.1` at 5 MB (`:17`). It is git-ignored (`.gitignore:17`).
- One line per tutor turn: time, random session id, attempt number, problem id, turn number, latency, correctness verdict, derived state, rule id, intervention type, template id,
  LLM outcome, input flags, character counts, and the ARIA-Pace fields (profile, bar as a fraction, risk, number of right/wrong verdicts so far, whether this turn updated it, reason code). **No student text and no tutor text**; only short identifiers are accepted (`agent/demo_log.py:19-42`), and a test checks this with distinctive strings.
- **Wipe it:** `cd ~/aria && rm -f data/demo_logs/turns.jsonl data/demo_logs/turns.jsonl.1` (safe while the demo runs). Or start with `--no-log` and no file is written.

## 9. Do not

- **Do not use `--all-problems`.** It lists all 30 bank problems; only the 8 above are rehearsed.
- **Do not flip `--llm-path` live** (and do not run it at all unless you re-ran the check on the demo laptop). Changing it needs a restart anyway.
- **Do not paste the problem's answer into the chat on stage** (message C in section 6, or the answer from this document), and do not read it out for the audience to type: ARIA would accept it and ask for reflection, which is correct, but it makes the demo look scripted and hides the point (ARIA decides *whether* to speak).
- Do not start the demo with `python3`, from another folder, or with `python ui/demo_app.py` (section 1).
- Do not edit `data/demo_problems.json` or any code at the venue. If a problem fails to load the server stops at launch with a message naming it.
- Do not run `git pull`/`git checkout` at the venue, and do not leave the `aria-pace` branch checked out; the build for the summit is the branch you decide on (`summit-demo` at the tag has no ARIA-Pace; `summit-pace-mac` has it).
- Do not rely on the screenshots as a live backup for Safari; they are Chrome.

## 10. Change log after the tag

Not a rehearsal crash fix: ARIA-Pace was added on branches `summit-pace` / `summit-pace-mac` (from the tag, which was not moved). `summit-demo` at the tag is still the build without it.

- 2026-10-20, feature: per-session BKT risk tracker, per-profile bar for hints on wrong steps at attempts 3-9, profile selector and live line, pace fields in the turn log; section 6 and `scripts/demo_rehearsal.py` rewritten to the "same student, three wrong answers" flow. Commit: see `git log summit-pace-mac`.
- 2026-10-20, fix: the profile explanation under the selector was unreadable in dark mode (dark grey text, no card background); it is now a card like the others. Teacher-summary wording "threshold" became "bar". Screens in `docs/demo_screens/light|dark` regenerated. Final build = tag `summit-2026-10-24-pace` (`git show summit-2026-10-24-pace`).
- Consequence to know about: at Quiet and Balanced a wrong answer after a correct step is often flagged without a hint (the old flow, wrong answer on message 3, no longer gets a hint there).
