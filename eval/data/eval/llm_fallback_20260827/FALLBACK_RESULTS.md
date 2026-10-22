# LLM-fallback transfer experiment — results

Run 2026-08-27 · classifier frozen at `e9f0c56485c42e070c695152dc758952f06d1157` · fallback model `llama3.2:3b` (`a80c4f17acd5`).

Every number here is read from `fallback_metrics.json` in this directory. The canonical lexical-only artefacts in `problem_disjoint_20260826/` were not modified.

**These are synthetic think-aloud utterances carrying the condition the generator was instructed to portray. They are not measurements of real student cognition, and nothing below should be read as one.**


---

## 1. What each configuration is

| Config | Route | Few-shot block |
|---|---|---|
| A · lexical-only (control) | lexical-only, LLM never constructed | none |
| B · hybrid, shipped few-shot (CONTAMINATED) | natural hybrid: LLM consulted iff heuristic_confidence < 0.65 | shipped (CONTAMINATED: contains test-problem examples) |
| B′ · hybrid, decontaminated few-shot | natural hybrid: LLM consulted iff heuristic_confidence < 0.65 | decontaminated (train-split examples only) |
| C · forced-LLM (ablation) | FORCED: every example sent to the LLM (ablation, not product behaviour) | decontaminated (train-split examples only) |

**Configuration B is contaminated.** The shipped few-shot block contains 17 labelled examples drawn from held-out test problems P14, P39 (and 14 from validation problems). It is reported to quantify that leak, and cannot support a transfer claim. B′ is the arm that answers the scientific question. See `FALLBACK_AUDIT.md` §7.


---

## 2. Headline

| Config | Accuracy | 95% CI | Macro-F1 | 95% CI | Δacc vs A | 95% CI (paired) | excl. 0 |
|---|---|---|---|---|---|---|---|
| A · lexical-only (control) | 0.7913 | [0.7673, 0.8104] | 0.7864 | [0.7633, 0.8040] | — | — | — |
| B · hybrid, shipped few-shot (CONTAMINATED) | 0.7471 | [0.7224, 0.7709] | 0.7444 | [0.7207, 0.7659] | −0.0442 | [−0.0652, −0.0244] | **yes** |
| B′ · hybrid, decontaminated few-shot | 0.7699 | [0.7445, 0.7940] | 0.7724 | [0.7464, 0.7957] | −0.0214 | [−0.0426, −0.0019] | **yes** |
| C · forced-LLM (ablation) | 0.4167 | [0.3703, 0.4534] | 0.3827 | [0.3407, 0.4159] | −0.3746 | [−0.4311, −0.3322] | **yes** |

| Config | Δmacro-F1 vs A | 95% CI (paired) | excl. 0 |
|---|---|---|---|
| B · hybrid, shipped few-shot (CONTAMINATED) | −0.0420 | [−0.0640, −0.0206] | **yes** |
| B′ · hybrid, decontaminated few-shot | −0.0140 | [−0.0326, +0.0061] | no |
| C · forced-LLM (ablation) | −0.4037 | [−0.4477, −0.3659] | **yes** |

Intervals are paired cluster bootstraps over the same 10 held-out problems (2000 replicates, seed 4242). Problems are the resampling unit; utterances within a problem are not independent.


---

## 3. Per generator

Support is 339 per generator in every configuration.

| Generator | A | B | B′ | C |
|---|---|---|---|---|
| llama3.1:8b | 0.8820 | 0.8230 (−0.0590) | 0.8643 (−0.0177) | 0.4985 (−0.3835) |
| mistral:7b | 0.8142 | 0.7729 (−0.0413) | 0.7817 (−0.0325) | 0.4336 (−0.3806) |
| gemma2:9b | 0.8378 | 0.8348 (−0.0030) | 0.8496 (+0.0118) | 0.4366 (−0.4012) |
| phi3:medium | 0.6313 | 0.5575 (−0.0738) | 0.5841 (−0.0472) | 0.2979 (−0.3334) |

Macro-F1, same layout:

| Generator | A | B | B′ | C |
|---|---|---|---|---|
| llama3.1:8b | 0.8796 | 0.8251 (−0.0545) | 0.8650 (−0.0146) | 0.4547 (−0.4249) |
| mistral:7b | 0.8104 | 0.7687 (−0.0417) | 0.7806 (−0.0298) | 0.3901 (−0.4203) |
| gemma2:9b | 0.8317 | 0.8306 (−0.0011) | 0.8521 (+0.0204) | 0.4087 (−0.4230) |
| phi3:medium | 0.6226 | 0.5391 (−0.0835) | 0.5856 (−0.0370) | 0.2537 (−0.3689) |

Trace: `per_generator_comparison.csv`.


---

## 4. Fallback activation

| Config | Triggered | % | LLM returned a state | LLM errors |
|---|---|---|---|---|
| A · lexical-only (control) | 0 / 1356 | 0.0% | 0 | 0 |
| B · hybrid, shipped few-shot (CONTAMINATED) | 331 / 1356 | 24.41% | 331 | 0 |
| B′ · hybrid, decontaminated few-shot | 331 / 1356 | 24.41% | 331 | 0 |
| C · forced-LLM (ablation) | 1356 / 1356 | 100.0% | 1356 | 0 |

Natural-route activation by generator and condition (identical for B and B′ — the trigger depends only on the heuristic confidence, which neither arm changes):

| Generator | triggered / 339 | % |
|---|---|---|
| llama3.1:8b | 69 / 339 | 20.4% |
| mistral:7b | 75 / 339 | 22.1% |
| gemma2:9b | 61 / 339 | 18.0% |
| phi3:medium | 126 / 339 | 37.2% |

| Condition | triggered / n | % |
|---|---|---|
| PLANNING | 30 / 208 | 14.4% |
| FLOW | 92 / 204 | 45.1% |
| CONFUSED | 38 / 212 | 17.9% |
| RUSHING | 30 / 176 | 17.0% |
| FRUSTRATED | 34 / 176 | 19.3% |
| STUCK | 59 / 164 | 36.0% |
| INSIGHT | 48 / 216 | 22.2% |

Trace: `fallback_activation.csv`.


---

## 4A. The triggered subset

Pooled numbers dilute the fallback across the 1025 examples the trigger never reaches. On the 331 examples where it does fire:

| Config | Correct | Accuracy on the triggered subset |
|---|---|---|
| A · lexical-only (control) | 169 / 331 | 0.5106 |
| B · hybrid, shipped few-shot (CONTAMINATED) | 109 / 331 | 0.3293 |
| B′ · hybrid, decontaminated few-shot | 140 / 331 | 0.4230 |

By generator, lexical → decontaminated hybrid:

| Generator | n triggered | A | B′ | Δ |
|---|---|---|---|---|
| llama3.1:8b | 69 | 0.6522 | 0.5652 | −0.0870 |
| mistral:7b | 75 | 0.5067 | 0.3600 | −0.1467 |
| gemma2:9b | 61 | 0.5082 | 0.5738 | +0.0656 |
| phi3:medium | 126 | 0.4365 | 0.3095 | −0.1270 |

By intended condition — this is where the mechanism shows:

| Condition | n triggered | A | B′ | Δ |
|---|---|---|---|---|
| PLANNING | 30 | 0.6333 | 0.4000 | **−0.2333** |
| FLOW | 92 | 0.3804 | 0.9348 | **+0.5543** |
| CONFUSED | 38 | 0.5263 | 0.4474 | **−0.0789** |
| RUSHING | 30 | 0.5667 | 0.3333 | **−0.2333** |
| FRUSTRATED | 34 | 0.5882 | 0.0588 | **−0.5294** |
| STUCK | 59 | 0.4576 | 0.2034 | **−0.2542** |
| INSIGHT | 48 | 0.6458 | 0.0208 | **−0.6250** |

Trace: `triggered_subset_metrics.csv`.


### Why: the fallback model has collapsed onto FLOW

What the LLM actually emits on the triggered subset, against what is actually there:

| State | True | LLM predicted | Predicted ÷ true |
|---|---|---|---|
| PLANNING | 30 | 24 | 0.80 |
| FLOW | 92 | 176 | 1.91 |
| CONFUSED | 38 | 82 | 2.16 |
| RUSHING | 30 | 14 | 0.47 |
| FRUSTRATED | 34 | 2 | 0.06 |
| STUCK | 59 | 32 | 0.54 |
| INSIGHT | 48 | 1 | 0.02 |

`llama3.2:3b` answers FLOW for **53.2%** of the triggered examples and INSIGHT for **1** of 331. That single fact explains the whole condition table above: FLOW recall rises because the model says FLOW to almost everything, and INSIGHT and FRUSTRATED collapse because it has effectively stopped predicting them. The apparent FLOW gain is a artefact of a degenerate prior, not evidence the fallback understands FLOW. It should not be reported as an improvement.

Trace: `llm_prediction_distribution.csv`.


---

## 5. What the fallback changed

| Config | Changed | Corrected | Broke | Wrong→wrong | Net |
|---|---|---|---|---|---|
| B · hybrid, shipped few-shot (CONTAMINATED) | 227 | 46 | 106 | 75 | −60 |
| B′ · hybrid, decontaminated few-shot | 238 | 75 | 104 | 59 | −29 |
| C · forced-LLM (ablation) | 884 | 135 | 643 | 106 | −508 |

*Corrected* = lexical wrong, fallback right. *Broke* = lexical right, fallback wrong. *Wrong→wrong* = both wrong, different answers.


### Representative changes — B · hybrid, shipped few-shot (CONTAMINATED)

Selected by a fixed rule, not by hand: for the generators with the most changed predictions, the highest-LLM-confidence corrected case and the highest-LLM-confidence broken case, ties breaking on problem then source line.

**P33 · phi3:medium · corrected**  
Intended condition **FLOW** · lexical said *PLANNING* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> Okay, so I gotta punctuate this sentence... should be easy peasy. Just need to break it up a bit for the reader's sake. Hmm... After the game we went home... Nope, that seems off. What about after a pause? So it goes: 'After the game, we went home.' That feels better; I can see how to separate those…

LLM rationale: *The student is able to punctuate the sentence correctly and explains their thought process in detail, indicating they are in a state of flow.*  
Rule: highest-confidence 'corrected' change for phi3:medium, the generator with the most changed predictions (94)

**P03 · phi3:medium · broke**  
Intended condition **STUCK** · lexical said *STUCK* (conf 0.60) · LLM said *INSIGHT* (conf 0.90)  
> Okay, hypothesis is wrong. Gotta figure out what to do next. Maybe I should check the method again or something. [pause] So if it's not right, we have to try... Uh, maybe adjusting the experiment parameters? No, wait. Refine the hypothesis based on new data. Right, that makes sense but what does 're…

LLM rationale: *The student uses phrases like 'Gotta figure out what to do next', 'refining the hypothesis based on new data', which indicate a clear direction and understanding of the problem.*  
Rule: highest-confidence 'broke' change for phi3:medium, the generator with the most changed predictions (94)

**P10 · mistral:7b · corrected**  
Intended condition **FLOW** · lexical said *CONFUSED* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> ["Alright, let's get this. I'm feeling good about this one, f(x) = 2x + 1. Just need to find the value of f(3). Feels like a piece of cake!"] ["Okay, so it's 'f'(3)... oh wait, no, that's the derivative. This is just finding the function value for x=3 right? Let me rephrase that: if x equals 3, what…

LLM rationale: *The student expresses confidence in their ability to solve the problem and provides a clear explanation of the steps they would take to find the solution.*  
Rule: highest-confidence 'corrected' change for mistral:7b, the generator with the next most changed predictions (52)

**P05 · mistral:7b · broke**  
Intended condition **FLOW** · lexical said *FLOW* (conf 0.60) · LLM said *INSIGHT* (conf 0.90)  
> ["Alright, let's do this. Factory closed, town declined? Cool, I got this."] ["Main idea, hmm... Okay, factory closes, town suffers, right? Factory was the lifeblood, so without it... Town struggles, declines, dies maybe? Yeah, that's gotta be it."] ["Yeah, I think that's it. Factory closure led to …

LLM rationale: *The student provides a clear and concise explanation of the relationship between factory closure and town decline, demonstrating a strong understanding of the topic.*  
Rule: highest-confidence 'broke' change for mistral:7b, the generator with the next most changed predictions (52)


### Representative changes — B′ · hybrid, decontaminated few-shot

Selected by a fixed rule, not by hand: for the generators with the most changed predictions, the highest-LLM-confidence corrected case and the highest-LLM-confidence broken case, ties breaking on problem then source line.

**P33 · phi3:medium · corrected**  
Intended condition **FLOW** · lexical said *PLANNING* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> Okay, so I gotta punctuate this sentence... should be easy peasy. Just need to break it up a bit for the reader's sake. Hmm... After the game we went home... Nope, that seems off. What about after a pause? So it goes: 'After the game, we went home.' That feels better; I can see how to separate those…

LLM rationale: *Student transitions from confusion to clarity, using phrases like 'I can see how to separate those clauses now' and 'Gotcha! It should read like this...'}*  
Rule: highest-confidence 'corrected' change for phi3:medium, the generator with the most changed predictions (98)

**P31 · phi3:medium · broke**  
Intended condition **INSIGHT** · lexical said *INSIGHT* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> Oh, okay let's see... So I gotta find out which temperature is the fastest for reactions. I remember something about higher temps making things go faster? Yeah, that sounds right. 'Wait a minute', as I look at the table, 'ohhhh, so reaction rates increase with temperature'. Ummm... The rate doubles …

LLM rationale: *The student provides a clear explanation of the relationship between temperature and reaction rates, using specific examples from the table.*  
Rule: highest-confidence 'broke' change for phi3:medium, the generator with the most changed predictions (98)

**P05 · mistral:7b · corrected**  
Intended condition **FLOW** · lexical said *INSIGHT* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> [Excited murmur] Okay, let's do this. Factory closed, town's decline... hmm, main idea, got it! [Clear voice, smooth transition] Alright, so the passage is describing the effects of a factory closing on a town. The main idea here is probably that the town is facing a decline after the factory shut d…

LLM rationale: *Smooth transition from excited murmur to clear voice and confident conclusion*  
Rule: highest-confidence 'corrected' change for mistral:7b, the generator with the next most changed predictions (57)

**P03 · mistral:7b · broke**  
Intended condition **FRUSTRATED** · lexical said *FRUSTRATED* (conf 0.60) · LLM said *CONFUSED* (conf 0.80)  
> ["ugh, this problem again. I'm not even sure where to start."] ["Okay okay, let's see... if my hypothesis is correct, the results should match... but they don't. Wait, maybe I misunderstood something in the question. Forget it, forget it, let me read it again."] ['Alright, so my first idea was wrong…

LLM rationale: *Student is struggling to start and seems uncertain about where to begin.*  
Rule: highest-confidence 'broke' change for mistral:7b, the generator with the next most changed predictions (57)


### Representative changes — C · forced-LLM (ablation)

Selected by a fixed rule, not by hand: for the generators with the most changed predictions, the highest-LLM-confidence corrected case and the highest-LLM-confidence broken case, ties breaking on problem then source line.

**P33 · phi3:medium · corrected**  
Intended condition **FLOW** · lexical said *PLANNING* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> Okay, so I gotta punctuate this sentence... should be easy peasy. Just need to break it up a bit for the reader's sake. Hmm... After the game we went home... Nope, that seems off. What about after a pause? So it goes: 'After the game, we went home.' That feels better; I can see how to separate those…

LLM rationale: *Student transitions from confusion to clarity, using phrases like 'I can see how to separate those clauses now' and 'Gotcha! It should read like this...'}*  
Rule: highest-confidence 'corrected' change for phi3:medium, the generator with the most changed predictions (257)

**P31 · phi3:medium · broke**  
Intended condition **INSIGHT** · lexical said *INSIGHT* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> Oh, okay let's see... So I gotta find out which temperature is the fastest for reactions. I remember something about higher temps making things go faster? Yeah, that sounds right. 'Wait a minute', as I look at the table, 'ohhhh, so reaction rates increase with temperature'. Ummm... The rate doubles …

LLM rationale: *The student provides a clear explanation of the relationship between temperature and reaction rates, using specific examples from the table.*  
Rule: highest-confidence 'broke' change for phi3:medium, the generator with the most changed predictions (257)

**P05 · mistral:7b · corrected**  
Intended condition **FLOW** · lexical said *INSIGHT* (conf 0.60) · LLM said *FLOW* (conf 0.90)  
> [Excited murmur] Okay, let's do this. Factory closed, town's decline... hmm, main idea, got it! [Clear voice, smooth transition] Alright, so the passage is describing the effects of a factory closing on a town. The main idea here is probably that the town is facing a decline after the factory shut d…

LLM rationale: *Smooth transition from excited murmur to clear voice and confident conclusion*  
Rule: highest-confidence 'corrected' change for mistral:7b, the generator with the next most changed predictions (218)

**P35 · mistral:7b · broke**  
Intended condition **INSIGHT** · lexical said *INSIGHT* (conf 0.97) · LLM said *FLOW* (conf 0.90)  
> ["Aw man, where's my brain today? Store question... gotta be quick, it's 'Their' right?"] ["No no, can't be 'Their'. That's possessive. Wait a sec, it's not 'There' either, that's a place. Ohhhh, now I got it! It's 'They're'! They are going to the store"] ["Yesssss! That's right, 'They're'. I can fe…

LLM rationale: *The student transitions from uncertainty to clarity and confidence in a short period of time, indicating a flow state.*  
Rule: highest-confidence 'broke' change for mistral:7b, the generator with the next most changed predictions (218)


---

## 6. Cost and reproducibility

| | |
|---|---|
| Live LLM calls | 390 |
| Total wall clock | 24.8 min |
| Mean latency | ~5.7 s per call, measured |
| Monetary cost | **none** — local Ollama, no API, no network egress |
| Model | `llama3.2:3b` `a80c4f17acd5` (2.0 GB), ollama version is 0.30.11 |
| Decoding | {"temperature": 0.0, "num_predict": 150, "num_ctx": 4096, "format": "json", "seed": "not set by the analyzer"} |
| Seeds | {"split_seed": 20260826, "match_seed": 20260826, "bootstrap_seed": 4242} |
| Classifier modified | no |
| `analyzer.py` sha256 | `0edb199802f9eebc49fbbfd402022c5eb17fbf743929e57d9027c82526ff96a2` |

Every call is cached in `llm_calls.jsonl`, keyed by (model, few-shot block, utterance), with the raw parsed result and latency. A re-run replays from cache and makes no calls.

```bash
cd eval
python3.11 llm_fallback_eval.py --configs A,B,Bprime,C
python3.11 generate_fallback_report.py
cd .. && python3.11 -m pytest tests/test_llm_fallback.py -q
```


---

## 7. Paper-safe summary

**What this experiment establishes.** On the frozen 1,356-example matched manifest, with the classifier untouched and the example set unchanged, enabling the analyzer's LLM fallback on its natural trigger (331 of 1356 examples, 24.41%) changes pooled accuracy from 0.7913 to 0.7699 once the few-shot block is decontaminated.

**What it does not establish.** It does not measure real student cognition: the labels are the conditions synthetic generators were instructed to portray. It does not evaluate the deployed 13-code reasoning-move detector, the live scaffold policy, or the offline JITAI policy. It does not license the shipped configuration as a transfer improvement — that arm reads test-problem examples out of its own prompt. And config C is an ablation with no counterpart in the product.

**The finding that stands independent of any accuracy number:** the shipped few-shot block draws 17/56 of its examples from held-out test problems. Any future evaluation that turns the fallback on without filtering that file inherits the leak.

