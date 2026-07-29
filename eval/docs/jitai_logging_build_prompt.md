# ARIA JITAI Logging Build Prompt

Use this prompt with Codex or another coding agent when you are ready to
implement the next ARIA milestone.

```text
You are working in the ARIA KLS repo. This is a local-first Python AI education
prototype for metacognitive tutoring. The current strongest path is:

student think-aloud text -> CognitiveStateAnalyzer -> MetacognitiveInterventionGenerator
-> MetacognitionTracker.

Your task is to implement the next milestone: separate interaction logging plus
an interpretable JITAI rule engine. Do not add ML training, cloud APIs, video,
facial analysis, or fake research claims.

Read these files first:
- README.md
- metacognition/analyzer.py
- metacognition/interventions.py
- metacognition/tracker.py
- metacognition/timing.py
- metacognition/transfer.py
- profiles.py, especially ProfileLearningGraph
- metacognition_eval.py

Implement Part 2: Logging infrastructure.

Requirements:
- Add a new module, preferably metacognition/interaction_logger.py.
- Use SQLite from the Python standard library. Do not use ChromaDB for these logs.
- Default DB path: data/logs/aria_interactions.sqlite3.
- Create tables for:
  - interaction_events
  - intervention_events
  - intervention_outcomes
  - optional typing_pause_events if the caller can provide per-keystroke or pause data
- Log enough fields to support later AIED/JITAI analysis:
  - timestamp
  - session_id
  - student_id or profile_id
  - topic/subject
  - student message
  - ARIA response, if available
  - response latency in ms
  - time since previous message in ms
  - pause summary, if available
  - tone/sentiment proxy as a clearly named heuristic field, not a diagnosis
  - cognitive state and analyzer confidence, if available
  - trigger condition for each intervention
  - intervention type delivered
  - intervention text
  - outcome: responded, re_engaged, ignored, ended_session, unknown
- Keep the logger callable from scripts/tests without requiring Ollama.
- Use dataclasses or small typed dictionaries for records so the schema is easy
  to understand.
- Add simple query helpers for later analysis, for example:
  - get_recent_interventions(session_id)
  - get_topic_intervention_stats(profile_id, topic)
  - get_ignored_intervention_count(session_id)
- Include a lightweight tone/sentiment heuristic. It can be basic keyword-based
  scoring, but name it as a proxy only.

Implement Part 3: JITAI-based heuristic decision rules.

Requirements:
- Add a new module, preferably metacognition/jitai.py.
- Implement an interpretable rule engine only. No trained model.
- Represent the six JITAI components explicitly:
  1. tailoring variable
  2. decision point
  3. decision rule
  4. intervention option
  5. proximal outcome
  6. distal outcome / learning habit signal
- Inputs should include:
  - profile_id
  - topic/subject
  - latest cognitive state from CognitiveStateAnalyzer
  - consecutive count from MetacognitionTracker.consecutive_state()
  - recent response latency / inactivity / pause data from InteractionLogger
  - topic struggle history from ProfileLearningGraph.graph node fields:
    confidence, study_count, struggle_count, study_hours, explanation_styles
  - previous intervention outcomes from InteractionLogger
- Output a JITAIDecision object/dict with:
  - should_intervene: bool
  - intervention_type: one of hint, encouragement, break_suggestion, task_chunking, none
  - trigger_condition
  - rule_id
  - rationale
  - citation_keys, initially empty or from a controlled local citation registry
  - extension_metadata for future learned-policy replacement
- Use clear explainable rules:
  - If the student is STUCK or CONFUSED for >= 2 consecutive turns, choose
    task_chunking or hint.
  - If FRUSTRATED persists or sentiment proxy is strongly negative, choose a
    break_suggestion or encouragement.
  - If RUSHING and latency is very short, choose task_chunking or a slow-down
    planning prompt.
  - If there are ignored interventions in the current session, escalate once
    from hint/task_chunking to break_suggestion.
  - If the learning graph shows high struggle history on this topic, prefer
    task_chunking over generic encouragement.
  - If the student is FLOW or INSIGHT, usually do not interrupt unless the
    caller explicitly requests a reflection prompt.
- Route the chosen type into the existing intervention system where practical.
  If the current intervention bank cannot express the type, add a small mapping
  layer rather than rewriting all existing interventions.
- Leave a clean extension point:
  - a PolicyProvider protocol/interface
  - RuleBasedJITAIPolicy as the default
  - comments showing where a future calibrated/learned model could be plugged
    in after real logged data exists

Research integrity constraints:
- Do not invent citations.
- If comments mention published findings, citations must come from a small local
  citation registry file, for example data/research/citations.json.
- If real papers or citation metadata are not already present, create the
  registry structure with TODO entries only, and write code so citation_keys can
  be empty.
- Do not implement Part 4's ADHD knowledge graph from model memory. Only scaffold
  it if needed; actual graph nodes/edges require real source text and manual
  review.

Testing:
- Add tests or smoke scripts that run without Ollama.
- Verify:
  - SQLite DB/tables are created.
  - One interaction can be logged and queried.
  - One intervention can be logged.
  - An outcome can be updated/logged.
  - JITAI rule engine returns task_chunking or hint for repeated CONFUSED/STUCK.
  - JITAI rule engine returns break_suggestion or encouragement for repeated
    FRUSTRATED/negative tone.
  - Learning graph high struggle history changes the decision rationale/type.
  - FLOW/INSIGHT usually returns should_intervene=false.
- Run Python compile checks.

Documentation:
- Update README.md with:
  - how to initialize/use InteractionLogger
  - how to call the JITAI policy
  - what is and is not research-backed yet
  - how logged data stays local
- Add a short docs/jitai_design.md explaining the rule engine, schema, and
  why Part 4/Part 6 are intentionally deferred.

Important sequencing:
- Finish and verify Parts 2 and 3 first.
- Do not proceed to Part 4 until logging and JITAI rules work end-to-end with
  real test interactions.
- Do not do fine-tuning. Part 6 is explicitly later, only after rules are
  validated and real logs exist.
```

## How To Use This

For the current repo, the right first build is Parts 2 and 3 only:

1. Create a SQLite interaction log.
2. Create a deterministic JITAI decision engine.
3. Wire decisions into the existing metacognition analyzer/intervention path.
4. Add smoke tests that do not require Ollama.
5. Only then collect real cited papers for the research knowledge graph.

Part 4 is not just a coding task. It requires actual source PDFs or citation
metadata, because the project should not claim research backing from generated
model memory.
