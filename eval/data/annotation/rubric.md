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

> First I need to figure out what the discount was, which is just... 20% off, right? So if $40 is what you get after that, then... let me think about how much it decreased from. It decreased by... um, x percent... (pauses) Wait, no. That's not how you do it.…

> First I need to make sure I get the range right... let me think... we don't wanna include 6 in this one... hmm... `for i in range(6)`? No, that's not it... wait, I can just do `range(1, 6)`... yeah, that's what I want... but now how do I print these numbers?…

### FLOW

Smooth continuous progress. The student is engaged and moving through the problem without stalling, second-guessing, or stopping to reorganise.

*Typical markers:* smooth, step by step, confident transitions, momentum

*Most often confused with:* PLANNING and INSIGHT. The corpus deliberately contains think-alouds written to LOOK like FLOW that are really one of those two, so an engaged surface is not sufficient.

> Alright, f(x) = 2x + 1, right? So if I'm finding f(3), that means I need to plug 3 in for x. Just multiply it by 2, that's easy... 6. And then add the 1... so, um, f(3) is just 7! Yeah, got it! Yeah, no problem at all. That one felt like a breeze.

> Okay, so it's f(x) = 2x + 1... If x is 3, that means I replace x with 3... So, it becomes 2*3 + 1... And then I just do the math... *taps pen on paper* That's right, 6 + 1 is... 7! Easy, so that's the answer: f(3) is 7. Just gotta double-check now... Yeah, that seems right. Makes total sense.…

### CONFUSED

Actively does not understand something and says so — circling, second-guessing, contradicting themselves. Still engaged with the content, but not tracking it.

*Typical markers:* wait, huh, I don't get it, circular reasoning, second-guessing

*Most often confused with:* STUCK. CONFUSED is muddled but still generating ideas; STUCK has run out of them.

> …then I need to divide by... uh, what's the percentage...? Yeah, I think it's 0.8... or is it 0.2? No, no, that can't be right... Oh man, I'm totally lost... This problem is just too weird, I don't get why it doesn't make sense to me. Ugh, ACT Math is so messed up.

> …I'll just use the formula for discounts, which is... um... what was it again? Oh yeah, (x - y) = z. But where do I put the 20% off? That's like, a variable or something... I don't get it, huh. Ugh, I feel so stupid now. Maybe I should just look up the answer and forget about this problem.

### RUSHING

Moving too fast for the work — skipping steps, guessing past uncertainty, settling for an answer without checking it.

*Typical markers:* fast, skips steps, 'probably', 'whatever', careless jumps

*Most often confused with:* FLOW. Both are fast. FLOW's steps are there and correct; RUSHING's are skipped or waved through.

> Alright, so it's like, (3² · 3⁴) / 3³... I'm pretty sure that's right... multiply the exponents first... (3² * 3⁴)... probably gets me 3 to the power of 6? whatever, doesn't matter for now... just get the numerator and denominator done separately... so it's like 3 to the 12th over 3 to the third...…

> Hmm... 'wept' probably means like crying or something... So if the sky is crying, that means it's raining right? Whatever, doesn't matter how hard it rained just that it was raining, so it was a rain day. That makes sense, I guess. Uh, yeah, I'm pretty sure that's right... Hopefully!

### FRUSTRATED

Emotional reaction to the task itself — irritation, resentment, wanting to abandon it. The feeling is the dominant content, not the reasoning.

*Typical markers:* ugh, 'this makes no sense', 'forget it', irritation

*Most often confused with:* STUCK. STUCK is an absence of next steps; FRUSTRATED is a reaction. A student can be either without the other.

> …Forget it, this makes no sense. Hmm, let me try again... print('even' if n % 2 == 0 else 'odd'). Ugh, do I need to use the ternary operator for this? This is so annoying. I guess that's right... sort of. Yeah, I hope I got it right...

> Okay, so it's gotta be a for loop, right? Or maybe while? Ugh, I always get these two mixed up. Wait, no, it's for. So, I'll start with 'for i in range(1,6)':... no, that can't be right, because 5 is not included in the range... umm... 'for i in range(6)'? Forget it, this makes no sense.…

### STUCK

Has no next step and knows it. Progress has stopped, and the student is not producing new candidate moves.

*Typical markers:* 'I don't know' repeated, [pause], mental breakdown of progress

*Most often confused with:* CONFUSED and FRUSTRATED. Ask what the student would do next: CONFUSED has an idea but it is muddled, FRUSTRATED may have one but resents it, STUCK has none.

> …Ugh, I always get these mixed up... Okay, let me try again... We could use a comma in between the two independent clauses, but... I don't know... what's the best way to do it? [pause] Oh man, this is killing me! Why did they even ask this?! Ugh, I have no idea how to fix it.…

> …Oh yeah, 12! And then there are two oxygens, so that would be... Um, what's the molar mass of an oxygen atom? *stares blankly ahead* I don't know, I don't know... [pauses] This is going nowhere... Okay, let me start over.…

### INSIGHT

A sudden realization after difficulty — a click moment where it finally makes sense — rather than steady progress that was never blocked.

*Typical markers:* 'OH', 'wait', 'ohhhh', sudden clarity, a click moment

*Most often confused with:* FLOW. INSIGHT requires the before-and-after: something was not understood, and then it was. Smooth confidence throughout is FLOW.

> Uh, if it's doubling every week and it's 2 cm in week one... wait, that means it's 4 cm in week two... ohhhh, yeah! It doubles every time... so I just keep multiplying by 2... Week three is 8 cm, right? And then for week four... it'll be 16 cm! OH, I get it now!…

> Wait, is this like a slope problem? Yeah, we need to find the slope. Alright, so I gotta get the difference in x-coordinates... (pauses) OH! We can use the formula y2 - y1 over x2 - x1. That's it! (laughs slightly) No, no, no... that's not it... OHHHHH... okay, so we need to subtract 11 from 2...…

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

*Category definitions and markers are reproduced from the generation prompts in `eval/metacognition/generate.py` that produced these items. Every example above is drawn from items that are NOT in your 150-item sheet.*
