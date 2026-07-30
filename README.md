# ARIA

ARIA is an on-device metacognitive learning assistant. It uses observable
evidence in a student's reasoning, a keyed problem model, learner context, and
recent history to ask one useful question without revealing the answer.
Friendly “thinking pattern” labels in the interface are uncertain hypotheses,
not research ground truth or diagnoses.

The repository contains both the student product and the evaluation system.

## Repository structure

- `ui/`: the Gradio student product and research interface
- `agent/`: reasoning, emotion, and intervention behavior
- `memory/`: learning graph, checkpoints, review scheduling, and local memory
- `knowledge/`: document ingestion and knowledge organization
- `shared/`: shared profile and reporting utilities
- `lora/`: optional local model adaptation tools
- `papers/` and `reports/`: research and educator reporting code
- `eval/`: the ARIA evaluation system and its existing Git history
- `research/`: evidence map, annotation taxonomy, preregistration drafts, task
  validation, ethics, privacy, and study materials

## Run ARIA

ARIA requires Python 3.11 and Ollama.

```bash
python3.11 -m pip install -r requirements.txt
ollama pull llama3.2:3b
./launch_aria.sh
```

The product opens locally at `http://127.0.0.1:7860`.

## Privacy

Student profiles, session logs, local vector stores, learning graphs,
checkpoints, exports, and emotion logs stay in the local `data/` directory and
are excluded from Git.

## Evaluation

The evaluation system lives in `eval/`.

```bash
python3.11 -m pip install -r eval/requirements.txt
python3.11 eval/metacognition_eval.py
```

See `eval/README.md` and `eval/EVIDENCE.md` for evaluation details and current
limitations. See `research/README.md` for the prospective evidence program.
No educator validation, IRB approval, or student efficacy result is implied by
the presence of those protocols.
