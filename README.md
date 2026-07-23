# ARIA KLS

ARIA KLS is an AI education research prototype focused on student
metacognition. It models what a student is doing while thinking aloud, then
uses that state to choose a short Socratic intervention instead of giving away
the answer.

## What Is In This Repo

- `metacognition/`: core metacognition engine.
  - `analyzer.py` classifies think-aloud text into seven states:
    `PLANNING`, `FLOW`, `CONFUSED`, `RUSHING`, `FRUSTRATED`, `STUCK`, and
    `INSIGHT`.
  - `interventions.py` selects one reflective prompt for the detected state.
  - `interaction_logger.py` stores structured local SQLite logs for
    interactions, interventions, outcomes, and typing pauses.
  - `jitai.py` implements an interpretable rule-based JITAI policy.
  - `tracker.py` stores longitudinal student state/intervention history.
  - `calibration.py`, `timing.py`, and `transfer.py` evaluate learning habits
    like confidence calibration, intervention timing, and self-initiated
    metacognition.
- `metacognition/generate.py`: generates synthetic think-aloud training and
  evaluation data with a local Ollama model.
- `metacognition_eval.py`: evaluates state detection, intervention quality,
  transition effectiveness, transfer detection, calibration validity, and
  timing validity.
- `ui/app.py` and `main.py`: local tutor console with live state evidence,
  JITAI decisions, outcome controls, research links, and session history.
- `data/synthetic_thinkaloud/`: generated dataset splits and stats.
- `data/eval/eval_100.json`: latest 100-sample evaluation results.
- `experiment.py`, `full_experiment.py`, `runner.py`, `report.py`: broader
  ARIA-vs-baseline experiment scaffolding.

## Current Saved Results

The included synthetic think-aloud dataset has 3,507 samples, balanced across
seven cognitive states with 501 samples per state.

The latest local Ollama evaluation (`data/eval/eval_100.json`) reports:

- State detection accuracy: `82.0%` on 100 examples
- Macro-F1: `0.814`
- Intervention appropriateness: `1.286 / 2.0`
- `CONFUSED`: `1.50 / 2.0`
- `STUCK`: `1.30 / 2.0`
- `INSIGHT`: `2.00 / 2.0`
- Negative-state simulated improvement: `50%` across three trials per state
- Transfer-detection F1: `0.886`
- Calibration computation: `PASS`
- Timing-rule validation: `PASS`

The intervention and transition metrics use local models as judges and
simulated students. They are development indicators, not evidence of real
student learning outcomes.

## Setup

This project expects Python 3.11 and a local Ollama installation for most AI
features.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and run Ollama separately, then pull the local models used by the code:

```bash
ollama pull llama3.1:8b
ollama pull llama3.2:3b
```

Optional API-backed evaluation also supports OpenAI and Gemini if the relevant
SDKs and environment variables are installed/configured.

## Useful Commands

From this repo directory:

```bash
python3.11 main.py
python3.11 metacognition_eval.py --no-llm --skip-judge --skip-sim --limit 50 --output /private/tmp/aria_metacognition_smoke.json
python3.11 metacognition_eval.py --limit 50 --sample-n 3 --trials 1
python3.11 metacognition/generate.py --samples 50
python3.11 -m unittest discover -s tests
```

The first command is the cheapest smoke test because it skips LLM judging and
simulation. The second uses Ollama-backed evaluation. The third generates a
small synthetic dataset.

## Local Interaction Logging

ARIA now includes a SQLite logger for research-grade event records:

```python
from metacognition.interaction_logger import InteractionLogger

logger = InteractionLogger()
logger.log_interaction(
    session_id="demo-session",
    profile_id="alex_chen",
    topic="algebra",
    student_message="ugh I don't get it",
    cognitive_state="CONFUSED",
    analyzer_confidence=0.82,
)
```

By default logs are stored locally at:

```text
data/logs/aria_interactions.sqlite3
```

These logs are separate from ChromaDB. ChromaDB remains memory/retrieval
storage; SQLite is used for structured analysis of interactions,
interventions, and outcomes. The tone fields are simple heuristic proxies, not
diagnostic claims.

## Rule-Based JITAI Policy

The JITAI policy is deterministic and interpretable:

```python
from metacognition.jitai import JITAIContext, RuleBasedJITAIPolicy

decision = RuleBasedJITAIPolicy().decide(JITAIContext(
    profile_id="alex_chen",
    topic="algebra",
    cognitive_state="CONFUSED",
    consecutive_state_count=2,
))
```

It returns a `JITAIDecision` with:

- whether to intervene
- intervention type
- trigger condition
- rule id
- rationale
- JITAI component metadata
- citation keys linked to `data/research/citations.json`

Current intervention types are `hint`, `encouragement`, `break_suggestion`,
`task_chunking`, `reflection`, and `none`.

## Research Integrity

`data/research/citations.json` contains verified bibliographic metadata and
source links for the theory behind ARIA's design. These papers provide design
rationale; they do not clinically validate ARIA or prove that its exact rule
thresholds are optimal. The thresholds still need evaluation with real student
interactions and appropriate consent.

## Important Engineering Notes

- Stale `eval.*` imports in the flat zip layout have been updated. Both
  `python3.11 metacognition_eval.py` and
  `python3.11 eval/metacognition_eval.py` are supported.
- The Gradio interface is a local demo, not a production student platform. It
  has no authentication, deployment, or multi-school data controls.
- The core metacognition path is the strongest demo candidate:
  think-aloud input -> cognitive-state detection -> intervention -> tracking.

## Strong Summit Demo Direction

For the KAH AI Education Summit, the clearest story is:

> ARIA listens to how a student thinks, not just whether they got the answer.
> It detects when the student is confused, rushing, stuck, frustrated,
> planning, flowing, or having insight, and adapts the tutor response to build
> metacognitive skill.

The local interactive demo lets someone type a think-aloud message and see:

1. detected cognitive state,
2. confidence/evidence,
3. one Socratic intervention,
4. session-level habit metrics over time.
