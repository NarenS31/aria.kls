# Think-aloud coding rubric

**150 items · one category per item · confidence 1-5**

You will read the problem a student was given and their think-aloud — what they said before, during, and after attempting it — and assign the ONE category that best describes their cognitive state. Then rate how confident you are, from 1 (guessing) to 5 (certain).

Work down the sheet in order. Do not go back and revise earlier items after you have developed a feel for the set: the point of two independent raters is that neither has been tuned by the other or by hindsight.

## Order of work

There are two rubric files and two sheets, used in this order:

1. Read `rubric_calibration.md` — the definitions, without examples.
2. Code the **calibration sheet** (14 items) — `raterN_calibration.csv`.
3. Then read `rubric.md` — the same document plus a worked example or two per category.
4. Code the **main sheet** (150 items) — `raterN_blind.csv`.

The calibration items are the same passages that appear as worked examples in the full rubric, shown complete and in a different order. You code them before you have seen those examples, so that round measures the definitions rather than your memory of a quoted passage. It is a check on the rubric, not on you: if you cannot place one, write that down — it means the definition or the original label is unclear. It is scored separately and never counted in the main analysis.

## Before you start: three rules

1. **Code what is on the page, not the student.** You are describing the reasoning in this text, not diagnosing a person.
2. **The literal word is not the category.** A student who writes "I don't wanna get stuck" is not thereby STUCK, and one who says "this is confusing" is not thereby CONFUSED — those words show up inside several different categories. Judge the reasoning; the vocabulary is a coincidence of how teenagers talk.
3. **Every item gets exactly one category.** If two fit, pick the one that governs the passage as a whole, and drop your confidence to 2 or 1 to record that it was close.

## The seven categories

### PLANNING

Deliberate setup before starting — laying out a plan, listing what they need, or choosing an approach — rather than smooth continuous progress.

*Typical markers:* 'first I need to', 'let me think', deliberate step-by-step setup

*Most often confused with:* FLOW. Both move forward confidently. PLANNING is arranging the approach before working; FLOW is already working.

### FLOW

Smooth continuous progress. The student is engaged and moving through the problem without stalling, second-guessing, or stopping to reorganise.

*Typical markers:* smooth, step by step, confident transitions, momentum

*Most often confused with:* PLANNING and INSIGHT. The corpus deliberately contains think-alouds written to LOOK like FLOW that are really one of those two, so an engaged surface is not sufficient.

### CONFUSED

Actively does not understand something and says so — circling, second-guessing, contradicting themselves. Still engaged with the content, but not tracking it.

*Typical markers:* wait, huh, I don't get it, circular reasoning, second-guessing

*Most often confused with:* STUCK. CONFUSED is muddled but still generating ideas; STUCK has run out of them.

### RUSHING

Moving too fast for the work — skipping steps, guessing past uncertainty, settling for an answer without checking it.

*Typical markers:* fast, skips steps, 'probably', 'whatever', careless jumps

*Most often confused with:* FLOW. Both are fast. FLOW's steps are there and correct; RUSHING's are skipped or waved through.

### FRUSTRATED

Emotional reaction to the task itself — irritation, resentment, wanting to abandon it. The feeling is the dominant content, not the reasoning.

*Typical markers:* ugh, 'this makes no sense', 'forget it', irritation

*Most often confused with:* STUCK. STUCK is an absence of next steps; FRUSTRATED is a reaction. A student can be either without the other.

### STUCK

Has no next step and knows it. Progress has stopped, and the student is not producing new candidate moves.

*Typical markers:* 'I don't know' repeated, [pause], mental breakdown of progress

*Most often confused with:* CONFUSED and FRUSTRATED. Ask what the student would do next: CONFUSED has an idea but it is muddled, FRUSTRATED may have one but resents it, STUCK has none.

### INSIGHT

A sudden realization after difficulty — a click moment where it finally makes sense — rather than steady progress that was never blocked.

*Typical markers:* 'OH', 'wait', 'ohhhh', sudden clarity, a click moment

*Most often confused with:* FLOW. INSIGHT requires the before-and-after: something was not understood, and then it was. Smooth confidence throughout is FLOW.

## When two categories both fit

Most disagreement in a set like this comes from four pairs. Use these tie-breakers rather than inventing your own, so both raters break ties the same way.

| If you are torn between | Ask |
|---|---|
| STUCK vs CONFUSED | Do they still have a candidate next step? An idea, however muddled, is CONFUSED. No idea at all is STUCK. |
| STUCK vs FRUSTRATED | Is the dominant content an absence of steps, or a reaction to the task? Absence is STUCK, reaction is FRUSTRATED. |
| FLOW vs PLANNING | Are they working, or arranging how to work? Arranging is PLANNING even when it sounds confident. |
| FLOW vs INSIGHT | Was anything unclear earlier in the passage? INSIGHT needs a before-and-after. Uninterrupted confidence is FLOW. |
| FLOW vs RUSHING | Are the steps present and sound, or skipped? Fast and correct is FLOW; fast and waved through is RUSHING. |

## Filling in the sheet

`problem_text` — the problem the student was working on, shown so you have the context they had. It is background, not evidence: the same problem appears across many different categories in this set, so what the problem was tells you nothing about the state the student was in.

`think_aloud` — what the student said, in three labelled parts.

`predicted_category` — one of: `PLANNING`, `FLOW`, `CONFUSED`, `RUSHING`, `FRUSTRATED`, `STUCK`, `INSIGHT`.

`confidence` — 1 guessing · 2 leaning · 3 reasonably sure · 4 confident · 5 certain. Use the low end freely; a well-placed 1 is more useful to the analysis than a polite 3.

Leave nothing blank. If an item seems uncodable, choose the closest category, mark confidence 1, and carry on.

---

*Category definitions and markers are reproduced from the generation prompts in `eval/metacognition/generate.py` that produced these items.*
