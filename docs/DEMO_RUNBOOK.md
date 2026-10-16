# ARIA summit demo: runbook

Build: branch `summit-demo`, tag `summit-2026-10-24` (Khan Lab School summit, 2026-10-24). After the tag the only allowed code change is a fix for a
crash found in rehearsal; each one is listed in the change log at the bottom.

## 1. Launch (exact)

```bash
cd ~/aria
GRADIO_ANALYTICS_ENABLED=False /usr/bin/python3 -m ui.demo_app
```

Then open **http://127.0.0.1:7861** in Chrome. The terminal shows `Running on local URL: http://127.0.0.1:7861`; first paint takes about 4 s from a cold start
(measured 3.8 s offline, see section 7). Port 7861 is the default (`ui/demo_app.py:210`), the server binds to 127.0.0.1 only (`ui/demo_app.py:219`).

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

The default launch shows **only the 8 featured problems** and has the **LLM path OFF** (section 5).

## 2. Presenter checklist

1. Wi-Fi off (section 3). 2. Launch (section 1). 3. Open the page, confirm the badge says **READY**. 4. Pick the problem, run the scripted flow (section 6).
5. Open the **Teacher summary** tab to show the timeline. 6. **Reset** before the next visitor (section 4).

The page has two tabs: **Student** (badge, chat, text box, Reset) and **Teacher summary** (state timeline, hints given, where the student got stuck; no chat transcript).
Badge meanings: READY (nothing yet), WATCHING (ARIA read the work and chose to say nothing), STEPPING IN (ARIA is speaking; the line under it says which policy rule decided).

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

**Decision: leave `--llm-path` OFF for the summit** (it is off by default: `ui/demo_app.py:208`). Do not enable it unless the check below passes on the demo laptop *and*
you decide you want free-form questions answered in the model's words.

Rule applied: 50+ live calls on the demo laptop with p95 < 5 s and 0 leaks. Numbers measured on the build machine (Apple M4, macOS 26.4.1, `llama3.2:3b` through the local Ollama), 3 full
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

- **Warm-up fails at launch:** the terminal prints `LLM path: Ollama not reachable, using templates only` (`ui/demo_app.py:218`) and the demo carries on with templates.
  The presenter sees nothing different. (Warm-up waits up to 60 s for a cold model load: `agent/demo_tutor.py:42`.)
- **A call runs past 10 s** (`agent/demo_tutor.py:39`): that visitor waits up to 10 s once, then gets the normal template reply; the LLM is then skipped for 20 s (`:43`) so the next
  visitors never wait. Replies are capped at 80 tokens (`:40`) and cut at the last full sentence.
- **Any LLM answer that states the answer or a false identity** is dropped and replaced by a template (the 2 rejections above).
- **Symbolic checker takes too long** (3 s budget, `agent/demo_tutor.py:44`): the presenter sees the quiet line *"Watching. I couldn't read that one. Try writing your next step as plain math."*
  (`agent/demo_templates.py:70`) and no verdict is guessed.

## 6. The 8 featured problems and their scripted flows

Type these exactly (the answers were re-derived by SymPy at load time; a typo in the bank stops the server with a clear error). "ARIA speaks" = badge STEPPING IN with a reply; "Watching" =
badge WATCHING, no reply.

**On stage, use messages 1 to 3** (stuck, a correct step, a wrong answer): that is the timing contrast the demo is about (ARIA speaks, stays quiet, speaks). **Message 4 is the correct
answer.** It is what the rehearsal types so the Teacher summary reads "Solved", and it makes ARIA ask the student to reflect. Type it only if a visitor has reached it themselves, or at the end
for the Teacher summary; **type it, never paste it, and never read it out before the visitor has it.**

| Id | Level | Problem | Flow | Rehearsal |
|---|---|---|---|---|
| `alg-01` | intro | Solve for x: 3(x - 4) = 2x + 5 | `I don't know` (ARIA speaks) → `x - 12 = 5` (Watching) → `x = 7` (ARIA speaks) → `x = 17` (ARIA speaks) | PASS |
| `alg-05` | core | Solve for x: x^2 - 5x + 6 = 0 | `I don't know` (ARIA speaks) → `(x - 2)(x - 3) = 0` (Watching) → `x = 1` (ARIA speaks) → `x = 2 or x = 3` (ARIA speaks) | PASS |
| `trig-01` | intro | Evaluate sin(pi/6) + cos(pi/3) | `I don't know` (ARIA speaks) → `sin(pi/6) = 1/2` (Watching) → `My answer is 3/2` (ARIA speaks) → `My answer is 1` (ARIA speaks) | PASS |
| `trig-04` | core | Simplify sin(x)cos(x)/(1 - sin(x)^2) | `I don't know` (ARIA speaks) → `1 - sin(x)^2 = cos(x)^2` (Watching) → `My answer is sin(x)` (ARIA speaks) → `My answer is tan(x)` (ARIA speaks) | PASS |
| `calc-01` | core | Differentiate f(x) = 3x^4 - 5x^2 + 7x - 9 | `I don't know` (ARIA speaks) → `12x^3 - 10x` (Watching) → `My answer is 12x^3 - 10x - 9` (ARIA speaks) → `My answer is 12x^3 - 10x + 7` (ARIA speaks) | PASS |
| `calc-02` | advanced | Differentiate f(x) = x^2 cos(2x) | `I don't know` (ARIA speaks) → `2x cos(2x)` (Watching) → `My answer is 2x cos(2x) + 2x^2 sin(2x)` (ARIA speaks) → `My answer is -2x^2 sin(2x) + 2x cos(2x)` (ARIA speaks) | PASS |
| `calc-06` | advanced | Differentiate f(x) = (x^2 + 1)^3 | `I don't know` (ARIA speaks) → `the outer derivative is 3(x^2+1)^2` (Watching) → `My answer is 6x(x^2+1)^3` (ARIA speaks) → `My answer is 6x (x^2 + 1)^2` (ARIA speaks) | PASS |
| `calc-07` | core | Find the integral of 6x^2 - 4x + 1 dx | `I don't know` (ARIA speaks) → `2x^3 - 2x^2` (Watching) → `My answer is 6x^3 - 4x^2 + x` (ARIA speaks) → `My answer is 2x^3 - 2x^2 + x` (ARIA speaks) | PASS |

Teacher summary after each flow: "Solved at turn N" (N = number of messages), the problem, the stuck stretch, and no student text.

## 7. Rehearsal (cold start, network blocked): 2026-10-15

`/usr/bin/python3 scripts/demo_rehearsal.py --offline --theme light` → raw results in `docs/rehearsal_results.json`, Teacher-summary screenshots in `docs/demo_screens/rehearsal/`.

- Cold start to first answer: **3.8 s**; **8/8 problems passed** (32 turns).
- Browser round trip per message (real Chrome): median 211 ms, p95 221 ms.
- Per problem, all of these had to hold: READY on selection; badge sequence as in the flow; Teacher summary says "Solved at turn N", shows the problem, has the four sections, contains none of the
  student's messages; Reset returns to READY with an empty chat; Teacher summary is cleared by Reset.
- Screenshots of every state (READY, STEPPING IN, WATCHING, 6-turn chat, Teacher summary, after Reset), light and dark theme: `docs/demo_screens/light/`, `docs/demo_screens/dark/`.

**Browsers.** Chrome: verified (driven through the real UI; both themes). **Safari: NOT verified.** Driving Safari needs *Safari → Settings → Developer → Allow remote automation*, a setting only you can switch
on, so I could not script it. Do this once by hand (2 minutes): open http://127.0.0.1:7861 in Safari, then check (a) the badge text is readable in READY / WATCHING / STEPPING IN, (b) the chat keeps the newest
message in view after 6 turns, (c) the Teacher summary tab is readable, (d) Reset empties the chat. If anything is off, use Chrome. The page uses nothing newer than flex layout with `gap` (Safari 14.1+).

## 8. The turn log

- File: `data/demo_logs/turns.jsonl` (`agent/demo_log.py:16`), created on first turn, rotated to `turns.jsonl.1` at 5 MB (`:17`). It is git-ignored (`.gitignore:17`).
- One line per tutor turn: time, random session id, attempt number, problem id, turn number, latency, correctness verdict, derived state, rule id, intervention type, template id,
  LLM outcome, input flags, character counts. **No student text and no tutor text**; only short identifiers are accepted (`agent/demo_log.py:20-34`), and a test checks this with distinctive strings.
- **Wipe it:** `cd ~/aria && rm -f data/demo_logs/turns.jsonl data/demo_logs/turns.jsonl.1` (safe while the demo runs). Or start with `--no-log` and no file is written.

## 9. Do not

- **Do not use `--all-problems`.** It lists all 30 bank problems; only the 8 above are rehearsed.
- **Do not flip `--llm-path` live** (and do not run it at all unless you re-ran the check on the demo laptop). Changing it needs a restart anyway.
- **Do not paste the problem's answer into the chat on stage** (message 4 in section 6, or the answer from this document), and do not read it out for the audience to type: ARIA would accept it and ask for reflection, which is correct, but it makes the demo look scripted and hides the point (ARIA decides *whether* to speak).
- Do not start the demo with `python3`, from another folder, or with `python ui/demo_app.py` (section 1).
- Do not edit `data/demo_problems.json` or any code at the venue. If a problem fails to load the server stops at launch with a message naming it.
- Do not run `git pull`/`git checkout` at the venue, and do not leave the `aria-pace` branch checked out; `summit-demo` at the tag is the build.
- Do not rely on the screenshots as a live backup for Safari; they are Chrome.

## 10. Change log after the tag

(none yet; each entry: date, symptom, fix, commit)
