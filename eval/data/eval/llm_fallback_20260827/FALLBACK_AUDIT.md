# LLM-fallback audit — offline seven-state `CognitiveStateAnalyzer`

Read-only audit performed 2026-08-27 against commit
`e9f0c56485c42e070c695152dc758952f06d1157`. No production file was modified.
All paths are relative to the repository root.

**Headline: the fallback is reachable, local, free and deterministic — but the
few-shot prompt it uses contains 17 labelled think-alouds drawn from held-out
TEST problems.** Turning the fallback on in its shipped configuration injects
test-set data into every classification. That does not block the experiment, but
it decides how the experiment must be structured and what it can claim.

---

## 1. Where the fallback lives, and what triggers it

| | |
|---|---|
| Implementation | `eval/metacognition/analyzer.py` → `CognitiveStateAnalyzer._llm_classify` |
| Routing decision | `CognitiveStateAnalyzer.analyze`, the branch on `h_conf >= self.HEURISTIC_TRUST or not self.use_llm` |
| Threshold | `HEURISTIC_TRUST = 0.65` |
| Client | `_llm_client()` — a module-level `ollama.Client` with `ARIA_META_TIMEOUT` (default 120 s) |

There is exactly one trigger path. The LLM runs if and only if **both** hold:

```
self.use_llm is True          AND     heuristic_confidence < 0.65
```

`use_llm` defaults to **`False`**. Nothing else in the class can invoke the LLM:
the audio-nudge path (`_apply_audio`) and the behavioural-fusion path
(`_fuse_behavioral`) both operate on an already-decided state and never call out.

## 2. Model, prompt, decoding, parser, failure behaviour

| Aspect | Value |
|---|---|
| Model tag | `llama3.2:3b` (from `ARIA_META_MODEL`, unset here so the default applies) |
| Local digest | `a80c4f17acd5`, 2.0 GB, served by Ollama 0.30.11 |
| System prompt | *"You classify a student's cognitive state from their think-aloud. Respond with JSON only, no prose."* |
| User prompt | few-shot block (8,142 chars, 56 labelled examples) + a one-paragraph instruction + the JSON schema + `Think-aloud: "<text>"` |
| Context supplied | **The utterance only.** No problem text, no subject, no persona, no prior turn, and — importantly — no ground-truth label. |
| Decoding | `temperature=0.0`, `num_predict=150`, `num_ctx=4096`, `format="json"`. **No seed is set.** |
| Parser | strip ``` fences → `json.loads` → regex `{.*}` fallback → uppercase and validate against the seven states → substring match → else the heuristic's own verdict |
| Confidence | `float(data["confidence"])`, clamped to [0, 1], defaulting to 0.6 when absent or unparseable |
| Failure behaviour | `except Exception: return None`, and `if ollama is None: return None`. The caller then keeps the heuristic verdict. A failure is therefore **silent** — it degrades to lexical-only with no error surfaced. |

**Determinism.** No seed is passed, so determinism rests on greedy decoding at
temperature 0. Verified empirically rather than assumed: 5 repeats of each of 3
utterances returned byte-identical state, confidence *and* free-text evidence
(3/3 texts, 1 distinct output each); a 12-example probe run at 4-way concurrency
matched the sequential result 12/12. Every call in the experiment is additionally
cached to `llm_calls.jsonl` keyed by (model, few-shot block, utterance), so the
scored artefacts are reproducible even if the model host later changes.

## 3. Is the fallback reachable from the offline evaluation route?

**Yes.** `analyze()` is the same entry point the offline evaluation uses; the
fallback is one constructor argument away.

The existing study never reaches it: `problem_disjoint_eval.py` and
`matched_generator_eval.py` both construct `CognitiveStateAnalyzer(use_llm=False)`,
and the canonical artefacts confirm it — `routing` is `{"heuristic": N}` for every
condition, in every published result file. So the canonical numbers are lexical-only
by construction, not by accident.

## 4. What "fallback enabled" actually means

**The LLM runs only when the lexical classifier is uncertain** — the first of the
three options in the brief. Specifically:

* it is **not** consulted for every prediction;
* it **replaces** the heuristic verdict outright when it is consulted and returns
  a parseable state (the heuristic result is discarded, not blended or voted);
* the confidence reported downstream becomes the *LLM's self-reported* confidence,
  which is on a different and equally uncalibrated scale from the heuristic's
  marker-margin score;
* if it fails or returns nothing, the heuristic verdict silently stands.

A forced-LLM route does not exist in the product. Configuration C below constructs
one by calling `_llm_classify` directly for every example; it is an **ablation**,
not a description of any shipping behaviour.

## 5. How often the trigger would fire on the frozen 1,356

Computed from the existing lexical-only predictions on the canonical matched
manifest — no LLM calls needed, since the trigger depends only on
`heuristic_confidence`.

**331 of 1,356 examples (24.4%) satisfy the trigger.**

By generator (n = 339 each):

| Generator | Triggering | % | Lexical accuracy on the triggering subset | Lexical accuracy overall |
|---|---|---|---|---|
| llama3.1:8b | 69 | 20.4% | 0.652 | 0.882 |
| mistral:7b | 75 | 22.1% | 0.507 | 0.814 |
| gemma2:9b | 61 | 18.0% | 0.508 | 0.838 |
| **phi3:medium** | **126** | **37.2%** | **0.437** | 0.631 |

By intended condition:

| Condition | Triggering | of | % | Lexical accuracy on the triggering subset |
|---|---|---|---|---|
| PLANNING | 30 | 208 | 14.4% | 0.633 |
| **FLOW** | **92** | 204 | **45.1%** | 0.380 |
| CONFUSED | 38 | 212 | 17.9% | 0.526 |
| RUSHING | 30 | 176 | 17.0% | 0.567 |
| FRUSTRATED | 34 | 176 | 19.3% | 0.588 |
| **STUCK** | **59** | 164 | **36.0%** | 0.458 |
| INSIGHT | 48 | 216 | 22.2% | 0.646 |

Two observations that make the experiment worth running. The trigger fires most
where the classifier is weakest — phi3:medium (37.2%, nearly double llama's rate)
and the FLOW/STUCK conditions — so the fallback gets its opportunities exactly
where the canonical gaps are. And the lexical route is right on only **169 of 331
(51.1%)** triggering examples, so there is real headroom: a perfect fallback would
lift pooled accuracy by up to **+11.9 points**, and a fallback no better than
chance-on-that-subset would lose ground.

## 6. Can it run cleanly? Cost and dependencies

| Requirement | Status |
|---|---|
| Runs locally | **Yes** — Ollama 0.30.11 on this machine |
| API charges | **None** — `llama3.2:3b` is already pulled (2.0 GB), no network egress |
| Private data | **None** — the corpus is synthetic LLM-generated think-alouds |
| Production changes | **None** — no product file is written; the analyzer source hash is recorded before and after |
| Retraining | **None** — no weights are touched and no few-shot example is added or rewritten |
| Determinism | **Verified** (§2) |
| Cost | ~5.7 s per call, measured. 1,687 live calls across all arms ≈ 2.7 h wall clock, single machine, CPU/GPU only |

## 7. The blocking finding: the few-shot prompt leaks test problems

`data/eval/mixed_fewshot_examples.json` supplies the 56 labelled think-alouds that
are pasted verbatim into **every** LLM prompt. Resolving each one against the
corpus by exact utterance match (56/56 resolved, 0 unresolved):

| Source split of the few-shot example | Count | Share | Problems |
|---|---|---|---|
| train | 25 | 44.6% | P04, P16, P18, P22, P32, P36, P37, P42, P44 |
| validation | 14 | 25.0% | P08, P09, P26, P34 |
| **test** | **17** | **30.4%** | **P14, P39** |

The file's own metadata describes it as *"Highest-confidence examples per (state,
generator)"* selected from the corpus — with no split filter, which is how held-out
problems ended up in it.

**Consequence.** Enabling the fallback as shipped puts 17 labelled examples from
two held-out test problems into the prompt for every one of the 1,356
classifications. The entire point of the canonical study is that the test problems
were never seen. A transfer improvement measured in that configuration would be
partly a measurement of the model reading test answers out of its own prompt.

The analyzer's own comment names the stakes: *"This is the 'retraining' for an
LLM-as-classifier — updating the in-context examples."* By the repository's own
definition the few-shot block is the LLM route's training data, and 30% of it is
test data.

### How the experiment is structured around it

Rather than stop, the experiment measures the contamination instead of inheriting it:

| Config | Route | Few-shot block | Status |
|---|---|---|---|
| **A** `lexical_only` | `use_llm=False` | none | Control. Must reproduce the canonical numbers exactly. |
| **B** `hybrid_shipped` | natural trigger | shipped, 56 examples | **Contaminated.** Reported to quantify the leak; cannot support a transfer claim. |
| **B′** `hybrid_decontaminated` | natural trigger | shipped block filtered to its 25 train-split examples | The scientifically valid arm. |
| **C** `forced_llm` | every example | decontaminated | **Ablation**, not product behaviour. |

B′ only ever **removes** examples from the shipped block — nothing is added,
rewritten or re-selected — so it is decontamination rather than the in-context
retraining the brief prohibits. This is asserted in
`tests/test_llm_fallback.py::test_decontaminated_block_is_a_strict_subset`.

**The confound this introduces, stated plainly:** B′'s block is smaller than B's
(25 vs 56 examples). Any B-minus-B′ difference therefore mixes *contamination*
with *fewer in-context examples*, and cannot be attributed to leakage alone. The
clean comparisons are A vs B′ (does an uncontaminated fallback help?) and A vs C
(what does a pure LLM route do?). B is reported for completeness and as an upper
bound on what the shipped configuration would appear to achieve.

The few-shot block is swapped by rebinding the module attribute that the prompt
builder already reads at call time. No file on disk is edited, and the
re-rendered "shipped" block is asserted byte-identical to the one the analyzer
built at import.

## 8. Audit verdict

| Question | Answer |
|---|---|
| Fallback present? | Yes, `_llm_classify` |
| Reachable from the offline route? | Yes, via `use_llm=True` |
| Missing / unreachable? | No |
| Nondeterministic? | No — greedy decoding, verified byte-identical over repeats |
| Paid or unavailable API? | No — local Ollama, model already present |
| Requires changing the classifier? | No — configuration only; source hashes recorded |
| **Clean to run as shipped?** | **No** — the shipped few-shot block contains 17 test-problem examples |
| Proceeding? | Yes, with the four-arm structure above. The leakage is measured, not inherited. |
