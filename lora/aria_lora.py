"""
Profile-specific LoRA fine-tuning for ARIA.

Trains a separate LoRA adapter for each of the 5 student profiles using mlx-lm.
Adapters are saved to lora/adapters/{profile_id}/ and loaded at inference time.

Training: mlx-lm LoRA via subprocess (YAML config per profile).
Inference: mlx_lm.load() + mlx_lm.generate() with adapter_path.

Resumes from last checkpoint automatically if lora/adapters/{profile}/adapters.safetensors exists.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LORA_DIR      = ROOT / "lora"
ADAPTERS_DIR  = LORA_DIR / "adapters"
CONFIGS_DIR   = LORA_DIR / "configs"
LORA_DATA     = LORA_DIR / "data"

for _d in (ADAPTERS_DIR, CONFIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
PROFILE_IDS = ["alex_chen", "jordan_rivera", "sam_okonkwo", "maya_patel", "eli_washington"]


# ------------------------------------------------------------------
# YAML config generator
# ------------------------------------------------------------------

def _make_config(
    profile_id: str,
    data_dir: Path,
    adapter_dir: Path,
    resume: bool = False,
) -> dict:
    """
    Generate mlx-lm YAML training config for a profile adapter.
    LoRA rank=8, alpha=16 → scale = alpha/rank = 2.0 (mlx-lm uses scale not alpha).
    """
    resume_file = str(adapter_dir / "adapters.safetensors") if resume and (adapter_dir / "adapters.safetensors").exists() else None

    return {
        "model": BASE_MODEL,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "data": str(data_dir),
        "seed": 42,
        "num_layers": 8,          # 8 transformer layers get LoRA applied (q_proj, v_proj)
        "batch_size": 4,
        "iters": 600,              # ~3 epochs for 100 sessions × ~2 turns each
        "val_batches": 10,
        "learning_rate": 2e-4,
        "steps_per_report": 25,
        "steps_per_eval": 50,
        "save_every": 100,
        "adapter_path": str(adapter_dir),
        "resume_adapter_file": resume_file,
        "max_seq_length": 512,
        "grad_checkpoint": True,   # M4 Pro has plenty of unified memory but checkpoint saves peak usage
        "grad_accumulation_steps": 2,
        "lora_parameters": {
            "rank": 8,
            "dropout": 0.05,
            "scale": 2.0,          # alpha/rank = 16/8 = 2.0
        },
        "mask_prompt": True,       # only compute loss on assistant turns
    }


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_profile_adapter(
    profile_id: str,
    force_retrain: bool = False,
    verbose: bool = True,
) -> str:
    """
    Fine-tune a LoRA adapter for one student profile.
    Returns the adapter directory path.
    Resumes from checkpoint if adapter already exists and force_retrain=False.
    """
    data_dir    = LORA_DATA / profile_id
    adapter_dir = ADAPTERS_DIR / profile_id
    config_path = CONFIGS_DIR / f"{profile_id}.yaml"

    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            f"Training data not found at {data_dir}/train.jsonl. "
            f"Run: python lora/data_formatter.py --dataset A"
        )

    # Count training examples
    with open(data_dir / "train.jsonl") as f:
        n_train = sum(1 for _ in f)
    if n_train < 5:
        raise ValueError(f"Too few training examples ({n_train}) for {profile_id}. Need at least 5.")

    # Determine resume
    resume = (adapter_dir / "adapters.safetensors").exists() and not force_retrain
    if resume and verbose:
        print(f"  [aria_lora] Resuming {profile_id} from existing checkpoint")

    cfg = _make_config(profile_id, data_dir, adapter_dir, resume=resume)
    # Adjust iters based on dataset size: ~3 passes through data
    cfg["iters"] = max(200, (n_train // cfg["batch_size"]) * 3)
    if verbose:
        print(f"  [aria_lora] {profile_id}: {n_train} examples → {cfg['iters']} iters")

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = ["python3", "-m", "mlx_lm", "lora", "-c", str(config_path)]

    if verbose:
        print(f"  [aria_lora] Training {profile_id} adapter...")
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
            raise RuntimeError(f"Training failed for {profile_id}:\n{err}")
        if verbose:
            print(f"  [aria_lora] {profile_id} trained in {elapsed/60:.1f} min")
    except KeyboardInterrupt:
        if verbose:
            print(f"\n  [aria_lora] Training interrupted. Checkpoint saved to {adapter_dir}")

    # Save training metadata
    meta = {
        "profile_id": profile_id,
        "n_train": n_train,
        "iters": cfg["iters"],
        "base_model": BASE_MODEL,
        "rank": cfg["lora_parameters"]["rank"],
        "scale": cfg["lora_parameters"]["scale"],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(adapter_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return str(adapter_dir)


def train_all_adapters(force_retrain: bool = False, verbose: bool = True) -> dict:
    """Train LoRA adapters for all 5 student profiles."""
    results = {}
    for pid in PROFILE_IDS:
        try:
            adapter_path = train_profile_adapter(pid, force_retrain=force_retrain, verbose=verbose)
            results[pid] = {"status": "ok", "adapter_path": adapter_path}
        except Exception as e:
            results[pid] = {"status": "error", "error": str(e)}
            if verbose:
                print(f"  [aria_lora] ERROR training {pid}: {e}")
    return results


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

_loaded_models: dict = {}  # cache: profile_id -> (model, tokenizer)


def load_profile_adapter(profile_id: str):
    """Load model + profile adapter. Cached per process."""
    if profile_id in _loaded_models:
        return _loaded_models[profile_id]

    adapter_dir = ADAPTERS_DIR / profile_id
    adapter_file = adapter_dir / "adapters.safetensors"

    from mlx_lm import load
    if adapter_file.exists():
        model, tokenizer = load(BASE_MODEL, adapter_path=str(adapter_dir))
    else:
        # No adapter trained yet: load base model
        model, tokenizer = load(BASE_MODEL)

    _loaded_models[profile_id] = (model, tokenizer)
    return model, tokenizer


def aria_lora_inference(
    student_message: str,
    profile_id: str,
    context: str = "",
    system_override: Optional[str] = None,
) -> str:
    """
    Inference with profile-specific LoRA adapter.
    Injects ChromaDB context and learning graph context (same as original ARIA).
    """
    from mlx_lm import generate
    from eval.profiles import PROFILE_MAP

    profile = PROFILE_MAP.get(profile_id)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_id}")

    if system_override:
        system = system_override
    else:
        from lora.data_formatter import _build_system_prompt
        subject = profile.subjects[0] if profile.subjects else "general"
        system = _build_system_prompt(
            name=profile.name,
            diagnosis=profile.diagnosis,
            learning_style=profile.learning_style,
            topic=subject,
            confidence=profile.baseline_confidence.get(subject, 0.5),
            struggles=list(profile.misconceptions.get(subject, ["unknown"]))[:2],
            best_style=profile.preferred_styles.get(subject, profile.learning_style),
        )
        if context:
            system += f"\n\nRelevant prior session context:\n{context}"

    model, tokenizer = load_profile_adapter(profile_id)

    # Build ChatML prompt
    prompt = f"<|system|>\n{system}\n<|user|>\n{student_message}\n<|assistant|>\n"

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=512,
        verbose=False,
    )
    # Strip any continuation of the prompt template
    response = response.split("<|user|>")[0].split("<|system|>")[0].strip()
    return response


def adapter_exists(profile_id: str) -> bool:
    return (ADAPTERS_DIR / profile_id / "adapters.safetensors").exists()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARIA profile-specific LoRA training and inference")
    parser.add_argument("--train-all", action="store_true", help="Train adapters for all 5 profiles")
    parser.add_argument("--train", action="store_true", help="Train adapter for one profile")
    parser.add_argument("--profile", type=str, choices=PROFILE_IDS, help="Profile ID")
    parser.add_argument("--inference", action="store_true", help="Run inference with profile adapter")
    parser.add_argument("--message", type=str, help="Student message for inference")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.train_all:
        results = train_all_adapters(force_retrain=args.force_retrain, verbose=verbose)
        for pid, r in results.items():
            status = "✓" if r["status"] == "ok" else "✗"
            print(f"  {status} {pid}: {r.get('adapter_path', r.get('error', ''))}")

    elif args.train:
        if not args.profile:
            parser.error("--train requires --profile")
        path = train_profile_adapter(args.profile, force_retrain=args.force_retrain, verbose=verbose)
        print(f"Adapter saved to: {path}")

    elif args.inference:
        if not args.profile or not args.message:
            parser.error("--inference requires --profile and --message")
        response = aria_lora_inference(args.message, args.profile)
        print(f"\nARIA ({args.profile}):\n{response}")

    else:
        parser.print_help()
