# JITAI Logging And Rule Engine

ARIA now has two local-first pieces for the next research milestone:

- `metacognition/interaction_logger.py`
- `metacognition/jitai.py`

## Logging

`InteractionLogger` writes to SQLite at:

```text
data/logs/aria_interactions.sqlite3
```

This is separate from ChromaDB. ChromaDB remains memory/retrieval storage;
SQLite is the structured research log for later AIED/JITAI analysis.

The logger stores:

- interaction events
- intervention events
- intervention outcomes
- optional typing pause events

The tone fields are keyword-based proxy signals only. They are not diagnostic
claims and should not be described that way.

## JITAI Policy

`RuleBasedJITAIPolicy` is deterministic and interpretable. It represents the
six JITAI components directly in each `JITAIDecision`:

1. tailoring variable
2. decision point
3. decision rule
4. intervention option
5. proximal outcome
6. distal outcome

The current intervention types are:

- `hint`
- `encouragement`
- `break_suggestion`
- `task_chunking`
- `reflection`
- `none`

The policy uses:

- latest cognitive state
- consecutive-state count
- response latency
- inactivity or pause summary
- tone proxy
- ignored prior interventions
- topic struggle history from `ProfileLearningGraph`

## Research Integrity

`data/research/citations.json` contains verified bibliographic metadata and
source links. Live JITAI and timing rules carry citation keys. These sources
provide theoretical motivation only; they do not empirically validate ARIA's
exact thresholds or establish clinical effectiveness.

Part 4, the ADHD-intervention knowledge graph, should wait until real source
papers are available. Part 6, fine-tuning, should wait until the rule engine has
been validated and real local logs exist.

## Minimal Use

```python
from metacognition.interaction_logger import InteractionLogger
from metacognition.jitai import JITAIContext, RuleBasedJITAIPolicy

logger = InteractionLogger()
logger.log_interaction(
    session_id="s1",
    profile_id="alex_chen",
    topic="algebra",
    student_message="ugh I don't get it",
    cognitive_state="CONFUSED",
    analyzer_confidence=0.82,
)

decision = RuleBasedJITAIPolicy().decide(JITAIContext(
    profile_id="alex_chen",
    topic="algebra",
    cognitive_state="CONFUSED",
    consecutive_state_count=2,
))
```
