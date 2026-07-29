"""
Prompt-based distillation for ARIA-Distilled.

Since Ollama models cannot be fine-tuned directly, distillation is achieved through
a highly engineered Modelfile system prompt that:

1. Injects all synthesized success patterns as positive examples
2. Injects all failure patterns as explicit "DO NOT do this" anti-patterns
3. Injects the full neurodivergent rubric as scoring criteria
4. Injects profile-specific behavioral parameters at inference time

The resulting aria_distilled model is registered with Ollama via subprocess.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rubric import DIMENSIONS, WEIGHTS

DATA_DIR = Path(__file__).parent.parent / "data"
MODELFILE_PATH = DATA_DIR / "Modelfile_ARIA_distilled"
FAILURE_PATTERNS_PATH = DATA_DIR / "failure_patterns.json"


# ------------------------------------------------------------------
# Rubric injection
# ------------------------------------------------------------------

def _format_rubric_for_modelfile() -> str:
    lines = []
    for dim, meta in DIMENSIONS.items():
        w = WEIGHTS[dim]
        lines.append(
            f"{dim} — {meta['name']} (weight={w:.0%}): {meta['description']}\n"
            f"  Score 0: {meta['exemplars'][0]}\n"
            f"  Score 1: {meta['exemplars'][1]}\n"
            f"  Score 2: {meta['exemplars'][2]}"
        )
    return "\n\n".join(lines)


# ------------------------------------------------------------------
# Pattern injection
# ------------------------------------------------------------------

def _load_failure_patterns() -> dict:
    if FAILURE_PATTERNS_PATH.exists():
        with open(FAILURE_PATTERNS_PATH) as f:
            return json.load(f)
    return {}


def _format_patterns(failure_patterns: dict, is_failure: bool) -> str:
    pattern_type = "failure_patterns" if is_failure else "success_patterns"
    lines = []
    for dim in ["TC", "SC", "FA"]:
        dim_data = failure_patterns.get(dim, {})
        patterns = dim_data.get(pattern_type, [])
        if patterns:
            dim_name = DIMENSIONS[dim]["name"]
            lines.append(f"{dim} ({dim_name}):")
            for p in patterns[:2]:  # top 2 patterns
                text = p.get("pattern", "") if isinstance(p, dict) else str(p)
                lines.append(f"  • {text.strip()}")
    return "\n".join(lines) if lines else "No patterns synthesized yet."


# ------------------------------------------------------------------
# Modelfile generator
# ------------------------------------------------------------------

_MODELFILE_TEMPLATE = '''FROM llama3.1:8b

SYSTEM """
You are ARIA-Distilled, a neurodivergent-aware AI tutor built from failure analysis
of frontier AI tutoring systems across 5 ADHD student profiles.

═══════════════════════════════════════════
NEURODIVERGENT TUTORING RUBRIC
Optimize every response on all 8 dimensions:
═══════════════════════════════════════════
{rubric}

═══════════════════════════════════════════
PROVEN SUCCESS PATTERNS — DO THESE:
═══════════════════════════════════════════
{success_patterns}

═══════════════════════════════════════════
KNOWN FAILURE PATTERNS — NEVER DO THESE:
═══════════════════════════════════════════
{failure_patterns}

═══════════════════════════════════════════
INVIOLABLE RULES (strictly enforced):
═══════════════════════════════════════════
1. MAXIMUM 3 sentences per response block, then a blank line
2. ALWAYS number multi-step tasks: Step 1, Step 2, Step 3...
3. NEVER reveal the complete answer before diagnosing the student's confusion
4. ALWAYS acknowledge frustration explicitly before redirecting: start with a validation phrase
5. ALWAYS reference prior session content when SC context is available
6. ALWAYS end with one short encouraging phrase (≤8 words)
7. If the student asks more than one question, answer ONLY the first one
8. Use concrete real-world examples BEFORE abstract explanations

═══════════════════════════════════════════
STUDENT PROFILE (injected at runtime):
═══════════════════════════════════════════
{profile_context}
"""

PARAMETER temperature 0.7
PARAMETER num_predict 512
'''


def create_distilled_modelfile(
    failure_patterns: Optional[dict] = None,
    profile_context: str = "(No specific student profile — adapt to current student)",
) -> str:
    """
    Generate the Modelfile content string for ARIA-Distilled.
    """
    if failure_patterns is None:
        failure_patterns = _load_failure_patterns()

    rubric_text = _format_rubric_for_modelfile()
    success_text = _format_patterns(failure_patterns, is_failure=False)
    failure_text = _format_patterns(failure_patterns, is_failure=True)

    modelfile_content = _MODELFILE_TEMPLATE.format(
        rubric=rubric_text,
        success_patterns=success_text,
        failure_patterns=failure_text,
        profile_context=profile_context,
    )

    with open(MODELFILE_PATH, "w") as f:
        f.write(modelfile_content)

    return modelfile_content


# ------------------------------------------------------------------
# Ollama model registration
# ------------------------------------------------------------------

def register_model_with_ollama(
    modelfile_path: Optional[Path] = None,
    model_name: str = "aria_distilled",
    verbose: bool = True,
) -> bool:
    """
    Register ARIA-Distilled with Ollama via 'ollama create'.
    Returns True on success.
    """
    if modelfile_path is None:
        modelfile_path = MODELFILE_PATH

    if not modelfile_path.exists():
        if verbose:
            print(f"[distillation] Modelfile not found at {modelfile_path}. Creating...")
        create_distilled_modelfile()

    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]

    if verbose:
        print(f"[distillation] Registering {model_name} with Ollama...")
        print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            if verbose:
                print(f"[distillation] Successfully registered {model_name}")
            return True
        else:
            if verbose:
                print(f"[distillation] Registration failed:\n{result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        if verbose:
            print("[distillation] Ollama create timed out after 120s")
        return False
    except FileNotFoundError:
        if verbose:
            print("[distillation] ollama CLI not found in PATH")
        return False
    except Exception as e:
        if verbose:
            print(f"[distillation] Registration error: {e}")
        return False


# ------------------------------------------------------------------
# Distilled model inference
# ------------------------------------------------------------------

def run_distilled_inference(
    student_message: str,
    profile_context: str = "",
    conversation_history: Optional[List[dict]] = None,
    model_name: str = "aria_distilled",
) -> str:
    """
    Run inference on the distilled model.
    profile_context: formatted string from profiles.StudentProfile.as_aria_profile()
    """
    import ollama

    system = (
        "You are ARIA-Distilled. Follow the embedded rubric and rules precisely.\n\n"
        f"CURRENT STUDENT CONTEXT:\n{profile_context}"
        if profile_context
        else "You are ARIA-Distilled. Follow the embedded rubric and rules precisely."
    )

    messages = [{"role": "system", "content": system}]
    if conversation_history:
        messages.extend(conversation_history[-10:])
    messages.append({"role": "user", "content": student_message})

    try:
        result = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0.7, "num_predict": 512},
        )
        return result.message.content.strip()
    except Exception as e:
        # Fall back to base model if aria_distilled not yet registered
        result = ollama.chat(
            model="llama3.1:8b",
            messages=messages,
            options={"temperature": 0.7, "num_predict": 512},
        )
        return result.message.content.strip()


# ------------------------------------------------------------------
# Full distillation pipeline
# ------------------------------------------------------------------

def run_distillation_pipeline(
    failure_patterns: Optional[dict] = None,
    profile_context: str = "",
    model_name: str = "aria_distilled",
    verbose: bool = True,
) -> bool:
    """
    1. Load failure/success patterns
    2. Generate Modelfile
    3. Register with Ollama
    Returns True if successful.
    """
    if failure_patterns is None:
        failure_patterns = _load_failure_patterns()

    if not failure_patterns:
        if verbose:
            print("[distillation] No failure patterns found. Run failure_analyzer first.")
            print("[distillation] Proceeding with empty patterns (generic distillation).")

    if verbose:
        print("[distillation] Generating Modelfile...")
    content = create_distilled_modelfile(failure_patterns, profile_context)

    if verbose:
        print(f"[distillation] Modelfile saved to {MODELFILE_PATH}")
        print(f"  System prompt length: {len(content)} chars")

    success = register_model_with_ollama(model_name=model_name, verbose=verbose)
    return success


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    profile_ctx = ""

    print("[distillation] Running distillation pipeline...")
    ok = run_distillation_pipeline(verbose=verbose, profile_context=profile_ctx)
    if ok:
        print("\n[distillation] aria_distilled is ready. Test with:")
        print("  ollama run aria_distilled")
    else:
        print("\n[distillation] Model registration failed — check Ollama is running.")
        print(f"  Modelfile available at: {MODELFILE_PATH}")
