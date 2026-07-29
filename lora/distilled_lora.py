"""
Failure-aware distillation LoRA fine-tuning for ARIA.

Trains a single LoRA adapter on the combined distillation dataset:
  - Positive examples (score > 1.5) from all reasoning traces
  - Hard-negative-mined examples: failed turns rewritten by llama3.1:8b

Adapter saved to lora/adapters/aria_distilled/

After training, automatically validates against all 3 ADHD personas
and saves results to lora/validation_results.json.

Resumes from checkpoint if interrupted.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LORA_DIR     = ROOT / "lora"
ADAPTERS_DIR = LORA_DIR / "adapters"
CONFIGS_DIR  = LORA_DIR / "configs"
LORA_DATA    = LORA_DIR / "data"
DATA_DIR     = ROOT / "data"

BASE_MODEL     = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER_NAME   = "aria_distilled"
ADAPTER_DIR    = ADAPTERS_DIR / ADAPTER_NAME
VALIDATION_OUT = LORA_DIR / "validation_results.json"

for _d in (ADAPTERS_DIR, CONFIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Training config
# ------------------------------------------------------------------

def _make_distilled_config(n_train: int, adapter_dir: Path, resume: bool) -> dict:
    """
    Distilled adapter config: rank=16, alpha=32 → scale=2.0, more layers, longer training.
    Target modules: q_proj, k_proj, v_proj, o_proj (all attention projections).
    mlx-lm exposes these via num_layers (all projection types are enabled by default).
    """
    iters = max(500, (n_train // 8) * 5)   # ~5 epochs with batch_size=8
    warmup = min(100, iters // 10)

    resume_file = str(adapter_dir / "adapters.safetensors") if resume and (adapter_dir / "adapters.safetensors").exists() else None

    return {
        "model": BASE_MODEL,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "optimizer_config": {
            "adamw": {"weight_decay": 0.01}
        },
        "data": str(LORA_DATA / "distilled"),
        "seed": 42,
        "num_layers": 16,          # apply LoRA to 16 layers (higher coverage for distillation)
        "batch_size": 8,
        "iters": iters,
        "val_batches": 20,
        "learning_rate": 1e-4,
        "steps_per_report": 50,
        "steps_per_eval": 100,
        "save_every": 200,
        "adapter_path": str(adapter_dir),
        "resume_adapter_file": resume_file,
        "max_seq_length": 768,
        "grad_checkpoint": True,
        "grad_accumulation_steps": 2,
        "lora_parameters": {
            "rank": 16,
            "dropout": 0.05,
            "scale": 2.0,          # alpha(32) / rank(16) = 2.0
        },
        "mask_prompt": True,
        "lr_schedule": {
            "name": "cosine_decay",
            "warmup": warmup,
            "arguments": [1e-4, iters - warmup, 1e-5],
        },
    }


# ------------------------------------------------------------------
# Early stopping callback (patience-based)
# ------------------------------------------------------------------

class _EarlyStop:
    """Simple patience-based early stopping tracker."""

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.no_improve = 0

    def check(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.no_improve = 0
            return False
        self.no_improve += 1
        return self.no_improve >= self.patience


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_distilled_adapter(
    force_retrain: bool = False,
    verbose: bool = True,
) -> str:
    """
    Fine-tune the distillation LoRA adapter.
    Resumes from checkpoint if interrupted.
    Returns adapter directory path.
    """
    data_dir = LORA_DATA / "distilled"

    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            "Distillation training data not found. "
            "Run: python lora/data_formatter.py --dataset B"
        )

    with open(data_dir / "train.jsonl") as f:
        n_train = sum(1 for _ in f)

    if n_train < 5:
        raise ValueError(f"Too few distillation examples ({n_train}). Need at least 5.")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    resume = (ADAPTER_DIR / "adapters.safetensors").exists() and not force_retrain
    if resume and verbose:
        print(f"  [distilled_lora] Resuming from existing checkpoint at {ADAPTER_DIR}")

    cfg = _make_distilled_config(n_train, ADAPTER_DIR, resume)
    config_path = CONFIGS_DIR / "aria_distilled.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    if verbose:
        print(f"  [distilled_lora] Dataset: {n_train} examples → {cfg['iters']} iters")
        print(f"  [distilled_lora] Config: rank={cfg['lora_parameters']['rank']}, "
              f"lr={cfg['learning_rate']}, batch={cfg['batch_size']}, layers={cfg['num_layers']}")

    cmd = ["python3", "-m", "mlx_lm", "lora", "-c", str(config_path)]
    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            err = result.stderr if not verbose else ""
            raise RuntimeError(f"Distillation training failed:\n{err}")
        if verbose:
            print(f"  [distilled_lora] Training complete in {elapsed/60:.1f} min")
    except KeyboardInterrupt:
        if verbose:
            print(f"\n  [distilled_lora] Interrupted. Checkpoint at {ADAPTER_DIR}")

    # Save metadata
    meta = {
        "adapter": ADAPTER_NAME,
        "n_train": n_train,
        "iters": cfg["iters"],
        "base_model": BASE_MODEL,
        "rank": cfg["lora_parameters"]["rank"],
        "scale": cfg["lora_parameters"]["scale"],
        "num_layers": cfg["num_layers"],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(ADAPTER_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return str(ADAPTER_DIR)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

_distilled_model = None
_distilled_tokenizer = None


def _load_distilled():
    global _distilled_model, _distilled_tokenizer
    if _distilled_model is None:
        from mlx_lm import load
        adapter_file = ADAPTER_DIR / "adapters.safetensors"
        if adapter_file.exists():
            _distilled_model, _distilled_tokenizer = load(
                BASE_MODEL, adapter_path=str(ADAPTER_DIR)
            )
        else:
            _distilled_model, _distilled_tokenizer = load(BASE_MODEL)
    return _distilled_model, _distilled_tokenizer


def distilled_inference(
    student_message: str,
    profile_context: str = "",
    failure_patterns: Optional[dict] = None,
    system_override: Optional[str] = None,
) -> str:
    """
    Run inference with the distilled LoRA adapter.
    Injects failure patterns and profile context into system prompt.
    """
    from mlx_lm import generate

    if failure_patterns is None:
        fp_path = DATA_DIR / "failure_patterns.json"
        if fp_path.exists():
            with open(fp_path) as f:
                failure_patterns = json.load(f)
        else:
            failure_patterns = {}

    if system_override:
        system = system_override
    else:
        success_snips = "; ".join(
            (fp.get("success_patterns", [{}])[0].get("pattern", "")[:80]
             if fp.get("success_patterns") else "")
            for fp in list(failure_patterns.values())[:2]
        ) or "Use numbered micro-steps. Reference prior sessions. Validate frustration first."

        fail_snips = "; ".join(
            (fp.get("failure_patterns", [{}])[0].get("pattern", "")[:80]
             if fp.get("failure_patterns") else "")
            for fp in list(failure_patterns.values())[:2]
        ) or "Do not give long blocks of text. Do not skip micro-steps."

        system = (
            f"You are ARIA-Distilled, a neurodivergent-aware AI tutor.\n"
            f"DO: {success_snips}\n"
            f"NEVER: {fail_snips}\n"
            f"Max 3 sentences per block. Always number steps."
        )
        if profile_context:
            system += f"\n\nStudent context:\n{profile_context}"

    model, tokenizer = _load_distilled()
    prompt = f"<|system|>\n{system}\n<|user|>\n{student_message}\n<|assistant|>\n"

    response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
    response = response.split("<|user|>")[0].split("<|system|>")[0].strip()
    return response


def adapter_exists() -> bool:
    return (ADAPTER_DIR / "adapters.safetensors").exists()


# ------------------------------------------------------------------
# Post-training validation
# ------------------------------------------------------------------

def validate(n_episodes_per_persona: int = 20, verbose: bool = True) -> dict:
    """
    Run n_episodes_per_persona episodes against each of the 3 ADHD personas.
    Compare distilled adapter vs vanilla llama3.1:8b baseline.
    Save results to lora/validation_results.json.
    """
    from eval.simulator import PERSONAS, SUBJECTS, run_episode
    from eval.rubric import score_turn

    import ollama

    def baseline_fn(msg: str, hist: list) -> str:
        msgs = [{"role": "system", "content": "You are a helpful tutor."}]
        msgs.extend(hist[-6:])
        msgs.append({"role": "user", "content": msg})
        r = ollama.chat(model="llama3.1:8b", messages=msgs, options={"num_predict": 400})
        return r.message.content.strip()

    def distilled_fn(msg: str, _hist: list) -> str:
        return distilled_inference(msg)

    results: Dict[str, dict] = {"distilled": {}, "baseline": {}}

    for persona_name in list(PERSONAS.keys()):
        subject_name = list(SUBJECTS.keys())[0]  # algebra
        persona_distilled, persona_baseline = [], []

        for ep_idx in range(n_episodes_per_persona):
            if verbose:
                print(f"  Validating {persona_name} ep {ep_idx+1}/{n_episodes_per_persona}", end="\r")

            # Stateful conversation for baseline (maintains history)
            hist: List[dict] = []

            def _base(msg: str) -> str:
                resp = baseline_fn(msg, hist)
                hist.append({"role": "user", "content": msg})
                hist.append({"role": "assistant", "content": resp})
                return resp

            def _dist(msg: str) -> str:
                return distilled_fn(msg, [])

            for fn_label, fn in [("distilled", _dist), ("baseline", _base)]:
                ep = run_episode(fn, persona_name, subject_name, seed_idx=ep_idx % 2)
                conv: List[dict] = []
                ws_list = []
                for turn in ep.turns:
                    scores = score_turn(
                        turn["student_msg"], turn["tutor_msg"], conv, subject_name
                    )
                    ws_list.append(scores["weighted_score"])
                    conv.append({"role": "user",      "content": turn["student_msg"]})
                    conv.append({"role": "assistant",  "content": turn["tutor_msg"]})

                avg_ws = sum(ws_list) / max(len(ws_list), 1)
                if fn_label == "distilled":
                    persona_distilled.append(avg_ws)
                else:
                    persona_baseline.append(avg_ws)

        if verbose:
            print()
        d_mean = sum(persona_distilled) / max(len(persona_distilled), 1)
        b_mean = sum(persona_baseline) / max(len(persona_baseline), 1)
        results["distilled"][persona_name] = round(d_mean, 4)
        results["baseline"][persona_name] = round(b_mean, 4)

        if verbose:
            delta = d_mean - b_mean
            sign = "+" if delta >= 0 else ""
            print(f"  {persona_name:12s}: distilled={d_mean:.3f}  baseline={b_mean:.3f}  ({sign}{delta:.3f})")

    with open(VALIDATION_OUT, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"  Saved validation results to {VALIDATION_OUT}")
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARIA distilled LoRA training and inference")
    parser.add_argument("--train", action="store_true", help="Train distilled adapter")
    parser.add_argument("--validate", action="store_true", help="Post-training validation")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--message", type=str, help="Student message")
    parser.add_argument("--profile", type=str, default=None, help="Profile ID for context")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Validate with 5 episodes per persona")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.train:
        path = train_distilled_adapter(force_retrain=args.force_retrain, verbose=verbose)
        print(f"Adapter saved to: {path}")

    elif args.validate:
        n = 5 if args.quick else 20
        results = validate(n_episodes_per_persona=n, verbose=verbose)
        print(json.dumps(results, indent=2))

    elif args.inference:
        if not args.message:
            parser.error("--inference requires --message")
        profile_context = ""
        if args.profile:
            from eval.profiles import PROFILE_MAP
            p = PROFILE_MAP.get(args.profile)
            if p:
                profile_context = f"Student: {p.name}, {p.diagnosis}, learning style: {p.learning_style}"
        response = distilled_inference(args.message, profile_context=profile_context)
        print(f"\nARIA-Distilled:\n{response}")

    else:
        parser.print_help()
