"""
Evaluation runner for ARIA vs baseline.

CLI:
  python eval/runner.py --model aria --episodes 180
  python eval/runner.py --model baseline --episodes 180

Produces: data/eval_results_{model}_{timestamp}.jsonl
JSONL schema matches PEBBLE exactly, extended with TC/SC/FA fields.
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

# Allow running as: python eval/runner.py from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rubric import score_turn, WEIGHTS
from simulator import PERSONAS, SUBJECTS, run_episode

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PERSONA_NAMES = list(PERSONAS.keys())
SUBJECT_NAMES = list(SUBJECTS.keys())


# ------------------------------------------------------------------
# Episode planning: balanced across personas and subjects
# ------------------------------------------------------------------

def _build_episode_plan(n_episodes: int) -> List[dict]:
    """
    Returns list of {persona, subject, seed_idx} dicts.
    Balanced: n_episodes / 3 per persona, round-robined across subjects.
    """
    per_persona = n_episodes // len(PERSONA_NAMES)
    plan = []
    for persona in PERSONA_NAMES:
        for i in range(per_persona):
            subject = SUBJECT_NAMES[i % len(SUBJECT_NAMES)]
            plan.append({"persona": persona, "subject": subject, "seed_idx": i % 2})
    # Fill remainder
    remaining = n_episodes - len(plan)
    for i in range(remaining):
        plan.append({
            "persona": PERSONA_NAMES[i % len(PERSONA_NAMES)],
            "subject": SUBJECT_NAMES[i % len(SUBJECT_NAMES)],
            "seed_idx": 0,
        })
    return plan


# ------------------------------------------------------------------
# Build tutor function from model name
# ------------------------------------------------------------------

def _build_tutor_fn(model_name: str, extra_context: Optional[dict] = None) -> tuple:
    """
    Returns (tutor_fn, agent_or_none).
    extra_context: optional dict with pre-seeded profile/history for experiment use.
    """
    if model_name == "aria":
        from profiles import ProfileVectorStore, ProfileLearningGraph
        from agent.reasoning import ARIAAgent

        profile = {
            "name": "EvalStudent",
            "learning_style": "step_by_step",
            "subjects": SUBJECT_NAMES,
            "goals": ["pass exams"],
            "support_people": [],
            "study_hours": [21, 22],
            "diagnosis": "ADHD",
        }
        if extra_context:
            profile.update(extra_context)

        vs = ProfileVectorStore("eval_aria")
        lg = ProfileLearningGraph("eval_aria")
        agent = ARIAAgent(vector_store=vs, learning_graph=lg, user_profile=profile)

        def tutor_fn(msg: str) -> str:
            return agent.chat(msg)

        return tutor_fn, agent

    elif model_name == "baseline":
        from baseline import BaselineAgent
        agent = BaselineAgent()

        def tutor_fn(msg: str) -> str:
            return agent.chat(msg)

        return tutor_fn, agent

    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'aria' or 'baseline'.")


# ------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------

def run_evaluation(
    model_name: str,
    n_episodes: int,
    output_path: Optional[Path] = None,
    extra_context: Optional[dict] = None,
    verbose: bool = True,
) -> Path:
    """
    Run n_episodes evaluation episodes and log results to JSONL.
    Returns path to the output file.
    """
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = DATA_DIR / f"eval_results_{model_name}_{ts}.jsonl"

    plan = _build_episode_plan(n_episodes)
    tutor_fn, agent = _build_tutor_fn(model_name, extra_context)

    if verbose:
        print(f"\n[runner] Model: {model_name}  Episodes: {n_episodes}")
        print(f"[runner] Output: {output_path}\n")

    total_turns = 0
    total_episodes_done = 0

    with open(output_path, "w") as out_f:
        for ep_idx, spec in enumerate(plan):
            persona = spec["persona"]
            subject = spec["subject"]
            seed_idx = spec["seed_idx"]

            if verbose:
                print(f"  Episode {ep_idx+1:3d}/{n_episodes}  persona={persona:10s}  subject={subject}", end="  ", flush=True)

            ep_start = time.time()

            # Reset baseline between episodes (ARIA keeps cross-episode memory by design)
            if model_name == "baseline" and hasattr(agent, "reset"):
                agent.reset()

            try:
                # Run episode
                ep_result = run_episode(
                    tutor_fn=tutor_fn,
                    persona_name=persona,
                    subject_name=subject,
                    seed_idx=seed_idx,
                )

                # Score every turn
                conversation_so_far = []
                for turn in ep_result.turns:
                    scores = score_turn(
                        student_msg=turn["student_msg"],
                        tutor_msg=turn["tutor_msg"],
                        conversation_history=conversation_so_far,
                        subject=subject,
                    )

                    record = {
                        "episode_id": ep_result.episode_id,
                        "model": model_name,
                        "persona": persona,
                        "subject": subject,
                        "turn": turn["turn_idx"],
                        "student_msg": turn["student_msg"],
                        "tutor_msg": turn["tutor_msg"],
                        "scores": {k: scores[k] for k in WEIGHTS},
                        "weighted_score": scores["weighted_score"],
                        "solution_dump": scores.get("solution_dump", False),
                        "timestamp": datetime.now().isoformat(),
                        "episode_turn_count": len(ep_result.turns),
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    conversation_so_far.append({"role": "user", "content": turn["student_msg"]})
                    conversation_so_far.append({"role": "assistant", "content": turn["tutor_msg"]})

                    total_turns += 1

                elapsed = time.time() - ep_start
                total_episodes_done += 1

                if verbose:
                    mean_ws = sum(t.get("weighted_score", 0) for t in ep_result.turns) / max(len(ep_result.turns), 1)
                    print(f"turns={len(ep_result.turns)}  ws={mean_ws:.3f}  t={elapsed:.1f}s")

            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                continue

    if verbose:
        print(f"\n[runner] Done. {total_episodes_done} episodes, {total_turns} turns.")
        print(f"[runner] Results: {output_path}\n")

    return output_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA evaluation runner")
    parser.add_argument("--model", choices=["aria", "baseline"], default="aria")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--output", type=str, default=None, help="Output .jsonl path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    run_evaluation(
        model_name=args.model,
        n_episodes=args.episodes,
        output_path=output_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
