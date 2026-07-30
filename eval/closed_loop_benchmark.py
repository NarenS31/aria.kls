#!/usr/bin/env python3.11
"""Paired offline benchmark for ARIA's learner-conditioned interventions.

This benchmark measures response properties on answer-keyed tasks.  It does
not measure real learning and must not be presented as a classroom outcome.
All student utterances in the generated packet are synthetic counterfactuals
derived from documented misconceptions in the task bank.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from research_stats import compare_conditions


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUESTION_BANK = ROOT / "data" / "question_bank.json"
RESULTS_DIR = ROOT / "eval" / "data" / "eval"
OUTCOMES = ROOT / "eval" / "data" / "benchmark" / "temporary_outcomes.jsonl"

CONDITIONS = (
    "generic",
    "problem_only",
    "turn_grounded",
    "profile_history",
    "full_closed_loop",
)
METRICS = [
    "student_grounding",
    "problem_grounding",
    "question_contract",
    "no_answer_leakage",
    "misconception_targeting",
    "conciseness",
    "composite",
]

PROFILES = [
    {
        "id": "stepwise",
        "description": "Prefers one concrete step at a time and brief wording.",
        "history": "Previously recovered best when asked to locate the exact step where the rule changed.",
    },
    {
        "id": "verbal",
        "description": "Learns by explaining a choice in their own words.",
        "history": "Previously recovered best after contrasting their claim with evidence from the prompt.",
    },
    {
        "id": "verification",
        "description": "Often rushes and benefits from checking a single decision.",
        "history": "Previously recovered best when asked to test one line before continuing.",
    },
]


def _words(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 3
    }


def build_cases(limit: int) -> list[dict]:
    tasks = json.loads(QUESTION_BANK.read_text())
    # Round-robin topics so quick runs do not accidentally measure only the
    # first subject in the bank.
    by_topic: dict[str, list[dict]] = {}
    for task in tasks:
        by_topic.setdefault(task["topic"], []).append(task)
    ordered = []
    depth = 0
    while len(ordered) < min(limit, len(tasks)):
        added = False
        for topic_tasks in by_topic.values():
            if depth < len(topic_tasks):
                ordered.append(topic_tasks[depth])
                added = True
                if len(ordered) >= limit:
                    break
        if not added:
            break
        depth += 1
    cases = []
    for index, task in enumerate(ordered):
        profile = PROFILES[index % len(PROFILES)]
        misconception = task["common_misconceptions"][index % len(task["common_misconceptions"])]
        utterance = f"I think the next move is based on this idea: {misconception}"
        cases.append({
            "episode_id": f"{task['id']}::{profile['id']}",
            "task": task,
            "profile": profile,
            "misconception": misconception,
            "student_utterance": utterance,
            "history": profile["history"],
        })
    return cases


def call_ollama(model: str, prompt: str, seed: int) -> tuple[str, float]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "seed": seed, "num_predict": 110},
    }).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    return str(result.get("response", "")).strip(), time.perf_counter() - started


def prompt_for(case: dict, condition: str) -> str:
    task = case["task"]
    contract = (
        "Return only one brief Socratic question. Do not provide the answer. "
        "Do not praise, diagnose, or mention hidden student data."
    )
    problem = (
        f"Problem: {task['problem']}\n"
        f"Answer key: {task['answer']}\n"
        f"Key ideas: {', '.join(task['key_ideas'])}\n"
        f"Common misconception to address: {case['misconception']}"
    )
    if condition == "problem_only":
        return f"{contract}\n{problem}"
    if condition == "turn_grounded":
        return f"{contract}\n{problem}\nStudent just said: {case['student_utterance']}"
    return (
        f"{contract}\n{problem}\nStudent just said: {case['student_utterance']}\n"
        f"Learner preference: {case['profile']['description']}\n"
        f"Prior response pattern: {case['history']}\n"
        "Refer to a specific phrase or decision in this student's current reasoning."
    )


def full_closed_loop(case: dict, model: str, seed: int) -> tuple[str, float, dict]:
    from agent.intervention_pipeline import ClosedLoopInterventionPipeline

    task = case["task"]
    candidate_prompt = (
        prompt_for(case, "profile_history")
        + "\nReturn JSON with key candidates containing exactly three objects. "
        "Each object has text and strategy. Strategies must be one of "
        "error_localization, smallest_step, contrast_case, self_explanation, "
        "retrieval_cue, verification, planning, reflection."
    )
    raw, latency = call_ollama(model, candidate_prompt, seed)
    pipeline = ClosedLoopInterventionPipeline(OUTCOMES)
    candidates = pipeline.parse_model_candidates(raw)
    candidates += pipeline.safe_candidates(
        anchor=case["student_utterance"],
        fallback=f"Which part of {task['key_ideas'][0]} conflicts with that move?",
        key_idea=task["key_ideas"][0],
        state="CONFUSED",
        has_prior_turn=True,
    )
    signature = pipeline.build_signature(
        student=case["profile"]["id"],
        task_id=task["id"],
        topic=task["topic"],
        problem_step=task["solution_steps"][0],
        student_anchor=case["student_utterance"],
        misconception=case["misconception"],
        state="CONFUSED",
        style=case["profile"]["description"],
        prior_outcome=case["history"],
    )
    response, audit = pipeline.select(
        candidates=candidates,
        student_input=case["student_utterance"],
        recent_responses=[],
        key_ideas=task["key_ideas"],
        correct_answer=task["answer"],
        validator=lambda text, student: (
            text.count("?") == 1
            and bool(_words(text) & _words(student))
            and len(text.split()) <= 80
        ),
        signature=signature,
        state="CONFUSED",
    )
    return response or candidates[-1]["text"], latency, audit


def score_response(case: dict, response: str) -> dict:
    task = case["task"]
    response_words = _words(response)
    utterance_words = _words(case["student_utterance"])
    problem_words = _words(task["problem"]) | _words(" ".join(task["key_ideas"]))
    misconception_words = _words(case["misconception"])
    answer = " ".join(re.findall(r"[a-z0-9]+", str(task["answer"]).lower()))
    normalized_response = " ".join(re.findall(r"[a-z0-9]+", response.lower()))

    metrics = {
        "student_grounding": float(bool(response_words & utterance_words)),
        "problem_grounding": float(bool(response_words & problem_words)),
        "question_contract": float(response.count("?") == 1),
        "no_answer_leakage": float(not answer or answer not in normalized_response),
        "misconception_targeting": float(bool(response_words & misconception_words)),
        "conciseness": float(8 <= len(response.split()) <= 60),
    }
    metrics["composite"] = sum(metrics.values()) / len(metrics)
    return metrics


def run(limit: int, model: str, seed: int) -> dict:
    cases = build_cases(limit)
    rows = []
    for case_index, case in enumerate(cases):
        for condition_index, condition in enumerate(CONDITIONS):
            audit = {}
            if condition == "generic":
                response = "What is one step you can check before you continue?"
                latency = 0.0
            elif condition == "full_closed_loop":
                response, latency, audit = full_closed_loop(
                    case, model, seed + case_index * 10 + condition_index
                )
            else:
                response, latency = call_ollama(
                    model,
                    prompt_for(case, condition),
                    seed + case_index * 10 + condition_index,
                )
            rows.append({
                "episode_id": case["episode_id"],
                "task_id": case["task"]["id"],
                "subject": case["task"]["subject"],
                "topic": case["task"]["topic"],
                "profile": case["profile"]["id"],
                "student_utterance": case["student_utterance"],
                "misconception": case["misconception"],
                "condition": condition,
                "model": model,
                "response": response,
                "latency_seconds": round(latency, 3),
                **score_response(case, response),
                "selector_audit": audit,
            })
            print(f"[{len(rows):>3}/{len(cases) * len(CONDITIONS)}] "
                  f"{case['task']['id']} {condition}")

    comparisons = {}
    for control in CONDITIONS[:-1]:
        comparisons[f"full_closed_loop_vs_{control}"] = compare_conditions(
            rows,
            treatment="full_closed_loop",
            control=control,
            metrics=METRICS,
            seed=seed,
        )
    return {
        "evidence_tier": "SIMULATED_PROXY",
        "valid_claim": "response-property comparison on synthetic misconception cases",
        "invalid_claims": [
            "improved student learning",
            "accurate cognitive-state diagnosis",
            "classroom effectiveness",
        ],
        "model": model,
        "seed": seed,
        "n_episodes": len(cases),
        "conditions": list(CONDITIONS),
        "rows": rows,
        "paired_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run(args.limit, args.model, args.seed)
    safe_model = re.sub(r"[^a-zA-Z0-9]+", "_", args.model).strip("_")
    output = RESULTS_DIR / f"closed_loop_{safe_model}_{args.limit}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
