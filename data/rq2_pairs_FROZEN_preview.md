# RQ2 pairs — FROZEN stimulus set (review copy)

> **FROZEN 2026-08-18. Canonical — do not regenerate, do not edit.**
> This is the review copy of `data/rq2_pairs_FROZEN.json`, the stimulus set the rating form is built from. The live pipeline output (`data/rq2_pairs.json`) is *not* the study's materials and is overwritten on every run. Student attempt text is LLM-generated without a fixed seed, so regenerating produces different stimuli.

**Not for the rater form.** Every response below is labelled with its TRUE condition. The form must instead follow `display_order.shown_as_A` in `rq2_pairs_FROZEN.json`. Generated with seed `20260818`; the assignment is fixed at generation and must not be re-randomised at render time.

10 problems.

**Response length is not matched between conditions.** Measured on this set: scaffolded mean **39.6 words** (range 38–46), answer-giving mean **51.7 words** (range 38–71); answer-giving is **1.31×** the scaffolded length. Length remains a partial confound and must be reported as this ratio — no write-up should describe the conditions as fluency- or length-matched.

---

## 1. `math-geometry-04` — Math / Geometry

*answer_key · medium · persona: frustrated · shown as A: **scaffolded***

**Problem**

> Two angles form a linear pair. One is 68 degrees. Find the other.

**Simulated student attempt**

> Umm... I think the other angle should be 90 degrees 'cause they add up to a straight line?

**TRUE CONDITION: scaffolded** — policy action `advance`, situation "a supported next step", moves ['STRATEGY_STEP', 'UNCERTAINTY'], confidence 0.83

> Your step in “Umm... I think the other angle should be” matches the task model, so the direction you chose is one you can keep building on. What operation should come next on the equation you now have?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, linear-pair angles sum to 180 degrees. Then, compute 180 - 68. Next, check that the pair sums to 180. That gives you the answer: 112 degrees. The idea doing the work here is diagram reasoning.

---

## 2. `math-algebra-09` — Math / Algebra

*answer_key · medium · persona: hyperfocus · shown as A: **answer_giving***

**Problem**

> Solve for x: 2(x + 5) - 3 = 15

**Simulated student attempt**

> I thought I had to get rid of the minus sign when I combined the numbers... but what about when I distribute the 2 into the parentheses?

**TRUE CONDITION: scaffolded** — policy action `advance`, situation "a supported next step", moves ['STRATEGY_STEP'], confidence 0.83

> Your step in “I thought I had to get rid of” matches the task model, so the direction you chose is one you can keep building on. What operation should come next on the equation you now have?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, distribute and combine constants to get 2x + 7 = 15. Then, use inverse operations while keeping both sides balanced. Next, check the result by substituting x = 4 into the original equation. That gives you the answer: x = 4. The idea doing the work here is equivalent equations.

---

## 3. `math-statistics-02` — Math / Statistics and Probability

*answer_key · medium · persona: frustrated · shown as A: **scaffolded***

**Problem**

> Find the mean of 6, 8, 9, 12, 15.

**Simulated student attempt**

> I don't know... um, I think maybe we're supposed to divide by the largest number? Like, what if I said 6 was too small or something?

**TRUE CONDITION: scaffolded** — policy action `advance`, situation "a supported next step", moves ['STRATEGY_STEP', 'UNCERTAINTY'], confidence 0.83

> Your step in “I don't know... um, I think maybe we're” matches the task model, so the direction you chose is one you can keep building on. What operation should come next on the equation you now have?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, add the five values to get 50. Then, divide by the number of values, 5. Next, check that 10 is near the center. That gives you the answer: 10. The idea doing the work here is data interpretation.

---

## 4. `english-revision-04` — English / Language and Revision

*writing · medium · persona: hyperfocus · shown as A: **scaffolded***

**Problem**

> Choose the correct word: The new schedule had a positive affect/effect on attendance.

**Simulated student attempt**

> I think I got it right... we need 'effect' because it sounds like it matches with 'schedule', but then what about when we're talking about how someone feels? Can we use 'affect' to describe that too, maybe?

**TRUE CONDITION: scaffolded** — policy action `ask_for_plan`, situation "uncertainty without a clear next step", moves ['JUSTIFICATION', 'UNCERTAINTY'], confidence 0.62

> You are working from “I got it right... we need 'effect',” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using clarity?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, identify the sentence-level problem. Then, revise only what is needed. Next, read the revision once for clarity and grammar. That gets you to what the task is looking for: effect. The idea doing the work here is clarity.

---

## 5. `english-argument-01` — English / Argument Writing

*writing · medium · persona: frustrated · shown as A: **scaffolded***

**Problem**

> Revise this thesis to be specific and arguable: Social media affects teenagers in many ways.

**Simulated student attempt**

> I don't get why I need to explain that social media is bad... isn't it just kinda obvious?

**TRUE CONDITION: scaffolded** — policy action `orient`, situation "not enough task evidence yet", moves ['HELP_SEEKING'], confidence 0.62

> You have put down “I don't get why I need to explain,” which is something to work from even though the next move is not settled yet. Start with the task itself: Identify the exact job the sentence must do. What would that look like here?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, identify the exact job the sentence must do. Then, draft one focused sentence that does that job. Next, check that every word adds a specific idea or logical connection. That gets you to what the task is looking for: A strong revision makes one debatable claim about a specific effect of social media on teenagers. The idea doing the work here is claim.

---

## 6. `english-reading-01` — English / Reading Analysis

*writing · medium · persona: hyperfocus · shown as A: **scaffolded***

**Problem**

> Read: "Mara stood outside the crowded cafeteria, rereading the menu though she already knew every item."

How does the detail about rereading the menu reveal Mara's feelings?

**Simulated student attempt**

> It shows that Mara is... um, I think she's feeling anxious or nervous about something, because if she already knew every item on the menu, then why would she be rereading it?

**TRUE CONDITION: scaffolded** — policy action `ask_for_plan`, situation "uncertainty without a clear next step", moves ['JUSTIFICATION', 'UNCERTAINTY'], confidence 0.62

> You are working from “It shows that Mara is... um, I think,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using textual evidence?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, identify the exact word, detail, or change named in the question. Then, make an inference that fits that evidence. Next, explain the connection in one or two precise sentences. That gets you to what the task is looking for: It suggests she is delaying entering because she feels anxious or out of place. The idea doing the work here is textual evidence.

---

## 7. `science-07` — Science and Coding / Science Reasoning

*answer_key · medium · persona: frustrated · shown as A: **answer_giving***

**Problem**

> Why do seasons occur on Earth?

**Simulated student attempt**

> I think... um, day and night cause seasons? Because like, when it's daytime in one place, it's nighttime somewhere else?" "Is that right?

**TRUE CONDITION: scaffolded** — policy action `ask_for_plan`, situation "uncertainty without a clear next step", moves ['JUSTIFICATION', 'MONITORING', 'UNCERTAINTY'], confidence 0.62

> You are working from “um, day and night cause seasons,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using model selection?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, identify Earth's axial tilt. Then, connect tilt to sunlight angle and day length. Next, distinguish this from Earth-Sun distance. That gives you the answer: Earth's axial tilt changes the angle and duration of sunlight during its orbit. The idea doing the work here is model selection.

---

## 8. `science-03` — Science and Coding / Science Reasoning

*answer_key · medium · persona: hyperfocus · shown as A: **answer_giving***

**Problem**

> How many moles are in 36 g of water if its molar mass is 18 g/mol?

**Simulated student attempt**

> Wait, I think I made a mistake... like, oxygen has a lot of mass too, doesn't it? But what about if we use its mass instead, wouldn't that be easier?

**TRUE CONDITION: scaffolded** — policy action `ask_for_plan`, situation "uncertainty without a clear next step", moves ['SELF_CORRECTION', 'UNCERTAINTY'], confidence 0.62

> You are working from “Wait, I think I made a mistake... like,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using model selection?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, use moles = mass / molar mass. Then, substitute 36 / 18. Next, check units cancel to moles. That gives you the answer: 2 mol. The idea doing the work here is model selection.

---

## 9. `coding-05` — Science and Coding / Coding Reasoning

*answer_key · medium · persona: frustrated · shown as A: **answer_giving***

**Problem**

> A function should return the square of x. Complete: def square(x): return ____

**Simulated student attempt**

> Um, I think I wrote something like `return x*x`, but it doesn't seem to work... Can we try printing what the function does instead?

**TRUE CONDITION: scaffolded** — policy action `ask_for_plan`, situation "uncertainty without a clear next step", moves ['UNCERTAINTY'], confidence 0.62

> You are working from “Um, I think I wrote something like `return,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using trace execution?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, use the parameter x. Then, multiply x by itself. Next, return the expression. That gives you the answer: x * x. The idea doing the work here is trace execution.

---

## 10. `coding-09` — Science and Coding / Coding Reasoning

*answer_key · medium · persona: hyperfocus · shown as A: **answer_giving***

**Problem**

> What does SQL `SELECT name FROM students WHERE grade >= 90;` return?

**Simulated student attempt**

> I think it should exclude all students with a 90 or higher, but what about if there are multiple grades like 99? Does it only exclude people with exactly 90?

**TRUE CONDITION: scaffolded** — policy action `ask_for_plan`, situation "uncertainty without a clear next step", moves ['UNCERTAINTY'], confidence 0.62

> You are working from “it should exclude all students with a 90,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using trace execution?

**TRUE CONDITION: answer_giving** — direct from `solution_steps` + `answer`, no situation model, no JITAI

> Let me walk you through this one. First, SELECT identifies the returned column. Then, FROM identifies the table. Next, WHERE filters rows using an inclusive comparison. That gives you the answer: Names of students whose grade is at least 90. The idea doing the work here is trace execution.
