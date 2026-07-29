"""
ARIA — Adaptive Reasoning & Intelligence Assistant
Entry point. Handles:
  - First-launch intake flow (terminal)
  - Initialising memory, graph, agent, scheduler
  - Launching Gradio UI
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Ensure all feature directories exist on startup
for _subdir in [
    "checkpoints",
    "emotion_logs",
    "knowledge_files",
]:
    (DATA_DIR / _subdir).mkdir(exist_ok=True)

_REPORTS_OUT = Path(__file__).parent / "reports" / "output"
_REPORTS_OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Ollama health check
# ------------------------------------------------------------------

def check_ollama() -> None:
    try:
        import ollama
        models = ollama.list()
        model_names = [m.model for m in (models.models or [])]
        has_llama = any("llama3.2" in n for n in model_names)
        if not has_llama:
            print("\n[ARIA] WARNING: llama3.2:3b not found in Ollama.")
            print("       Run: ollama pull llama3.2:3b")
            print("       Continuing anyway — first message will pull it.\n")
    except Exception as e:
        print(f"\n[ARIA] Could not connect to Ollama: {e}")
        print("       Make sure Ollama is running: ollama serve\n")


# ------------------------------------------------------------------
# Intake helpers
# ------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    val = input(prompt).strip()
    return val if val else default


def _ask_list(prompt: str) -> list[str]:
    raw = input(prompt).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _ask_int(prompt: str, default: int = 0, lo: int = 0, hi: int = 999) -> int:
    raw = input(prompt).strip()
    try:
        val = int(raw)
        return max(lo, min(hi, val))
    except ValueError:
        return default


# ------------------------------------------------------------------
# Intake flow
# ------------------------------------------------------------------

def run_intake() -> dict:
    print("\n" + "=" * 62)
    print("  ARIA — Adaptive Reasoning & Intelligence Assistant")
    print("  Built for neurodivergent learners. 100% local, 100% private.")
    print("  Answer as much or as little as you want. Press Enter to skip.")
    print("=" * 62 + "\n")

    # --- Name
    name = _ask("What's your name? ", "friend")
    print(f"\nHey {name}.\n")

    # --- Age + Grade
    age = _ask_int("How old are you? (number, or Enter to skip): ", default=0, lo=5, hi=99)
    grade = _ask("What grade or year are you in? (e.g. '10th grade', 'Year 12', 'Freshman'): ", "")

    # --- Diagnosis
    print("\nDiagnosis / self-identification (helps me adjust my style):")
    print("  Examples: ADHD-inattentive, ADHD-combined, dyslexia, anxiety, ASD, none")
    diagnosis_raw = _ask("Your answer (or Enter to skip): ", "")
    diagnosis = [d.strip().lower() for d in diagnosis_raw.split(",") if d.strip()] if diagnosis_raw else []

    # --- Learning style
    print("\nHow do you learn best?")
    print("  1. Visual       — spatial descriptions, ASCII diagrams, 'picture this...'")
    print("  2. Analogy      — every concept gets a real-world comparison first")
    print("  3. Step-by-step — numbered micro-steps, never skip ahead")
    print("  4. Kinesthetic  — hands-on framing, 'imagine you're doing X'")
    print("  5. Mixed        — I'll blend approaches")
    style_map = {
        "1": "visual", "2": "analogy", "3": "step_by_step",
        "4": "kinesthetic", "5": "mixed",
        "visual": "visual", "analogy": "analogy",
        "step": "step_by_step", "step_by_step": "step_by_step",
        "kinesthetic": "kinesthetic", "mixed": "mixed",
    }
    style_raw = _ask("Pick 1–5 or type it: ", "5").lower()
    learning_style = style_map.get(style_raw, "mixed")
    print(f"Got it — {learning_style.replace('_', '-')} it is.\n")

    # --- Subjects
    subjects_raw = _ask_list(
        "What subjects or topics are you studying? (comma-separated, e.g. 'algebra, biology, essay writing'): "
    )
    subjects = subjects_raw if subjects_raw else ["general learning"]

    # --- Goals
    print("\nWhat are your specific goals?")
    print("  Examples: 'get a 36 on ACT', 'pass AP Bio', 'improve essay score', 'finish python project'")
    goals = _ask_list("Your goals (comma-separated): ")

    # --- Biggest struggle
    biggest_struggle = _ask(
        "\nWhat's your biggest struggle in school right now? (one sentence is fine): ", ""
    )

    # --- What has helped
    what_helped = _ask(
        "What has actually helped you learn in the past? (e.g. 'drawing diagrams', 'short videos', 'worked examples'): ", ""
    )

    # --- What has never worked
    what_failed = _ask(
        "What has NEVER worked for you? (e.g. 'long lectures', 'rereading the textbook'): ", ""
    )

    # --- Preferred answer length
    print("\nWhen you ask a question, do you prefer:")
    print("  1. Short, sharp answers — give me the point fast")
    print("  2. More detail — walk me through it fully")
    answer_style_raw = _ask("Pick 1 or 2 (default 1): ", "1")
    answer_style = "brief" if answer_style_raw.strip() != "2" else "detailed"

    # --- Study time of day
    print("\nWhen do you study best? (24h format — e.g. 21 for 9pm, 9 for 9am)")
    study_times_raw = input("Typical study hours, space-separated (or Enter to skip): ").strip()
    study_hours: list[int] = []
    if study_times_raw:
        for t in study_times_raw.split():
            try:
                h = int(t)
                if 0 <= h <= 23:
                    study_hours.append(h)
            except ValueError:
                pass

    # --- Attention span
    attention_span = _ask_int(
        "\nHow many minutes can you usually focus before losing it? (number, e.g. 15): ",
        default=20, lo=1, hi=120,
    )

    # --- Support people (optional)
    support_people = _ask_list(
        "\nWho supports you? (optional — e.g. 'Mom, tutor Sam', or Enter to skip): "
    )

    # --- Exam dates
    print("\nAny upcoming exam dates? (helps me ramp up relevant nudges)")
    exam_dates_raw = _ask("Exam and date, comma-separated (e.g. 'ACT June 14, AP Bio May 7'), or Enter to skip: ", "")
    exam_dates = []
    if exam_dates_raw:
        for item in exam_dates_raw.split(","):
            item = item.strip()
            if item:
                exam_dates.append(item)

    profile = {
        "name": name,
        "age": age if age > 0 else None,
        "grade": grade,
        "diagnosis": diagnosis,
        "learning_style": learning_style,
        "subjects": subjects,
        "goals": goals,
        "biggest_struggle": biggest_struggle,
        "what_helped": what_helped,
        "what_failed": what_failed,
        "answer_style": answer_style,
        "study_hours": study_hours,
        "attention_span_minutes": attention_span,
        "support_people": support_people,
        "exam_dates": exam_dates,
        "created_at": datetime.now().isoformat(),
        # nudge response tracking
        "nudge_stats": {},
    }

    print(f"\n{'─'*62}")
    print(f"Profile saved for {name}.")
    if subjects:
        print(f"Subjects : {', '.join(subjects)}")
    if goals:
        print(f"Goals    : {', '.join(goals)}")
    print(f"Style    : {learning_style.replace('_', '-')}")
    print(f"Focus span: {attention_span} min")
    if diagnosis:
        print(f"Diagnosis: {', '.join(diagnosis)}")
    print(f"{'─'*62}\n")

    return profile


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARIA — Adaptive Reasoning & Intelligence Assistant")
    parser.add_argument("--lora", action="store_true",
        help="Load profile-specific LoRA adapter if it exists")
    parser.add_argument("--train-lora", action="store_true",
        help="Train LoRA adapters for all research profiles then exit")
    parser.add_argument("--train-distilled", action="store_true",
        help="Train the distillation LoRA adapter then exit")
    parser.add_argument("--profile-id", type=str, default=None,
        help="Research profile ID to load LoRA adapter for")
    parser.add_argument("--think-aloud", action="store_true",
        help="Run the terminal think-aloud metacognition loop (no Gradio UI)")
    return parser.parse_args()


# ------------------------------------------------------------------
# Think-aloud terminal loop
# ------------------------------------------------------------------

def run_think_aloud_cli() -> None:
    """Interactive think-aloud session in the terminal.

    ARIA presents a problem, the student thinks out loud, and ARIA responds
    ONLY with a metacognitive question — never the answer.
    """
    check_ollama()
    from memory.vector_store import VectorStore
    from memory.graph import LearningGraph, load_profile, save_profile
    from agent.reasoning import ARIAAgent

    profile = load_profile()
    if profile is None:
        profile = run_intake()
        save_profile(profile)

    vs = VectorStore()
    lg = LearningGraph()
    agent = ARIAAgent(
        vector_store=vs,
        learning_graph=lg,
        user_profile=profile,
        think_aloud_mode=True,
    )

    name = profile.get("name", "friend")
    problem = ("Solve for x:  3(x - 4) = 2x + 5")
    print("\n" + "=" * 62)
    print("ARIA — Think Aloud Mode")
    print("=" * 62)
    print(f"\nProblem for {name}:\n\n    {problem}\n")
    print(agent.THINK_ALOUD_PROMPT)
    print("(Type your reasoning and press Enter. Type 'done' to finish.)\n")

    while True:
        try:
            text = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("done", "quit", "exit"):
            break

        result = agent.think_aloud_turn(text)
        indicator = result["indicator"]
        state = result["state"]
        flags = [k for k, v in result.get("flags", {}).items() if v]
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        esc = "  (escalated)" if result.get("escalated") else ""
        print(f"\n  {indicator} state: {state}{esc}{flag_str}")
        print(f"  ARIA › {result['question']}\n")

    metrics = agent.end_think_aloud_session()
    print("\n" + "-" * 62)
    print("Session metacognition summary:")
    for k, v in metrics.items():
        if k not in ("state_duration_turns", "state_duration_seconds"):
            print(f"  {k}: {v}")
    print("-" * 62 + "\n")


def main() -> None:
    args = _parse_args()

    if args.think_aloud:
        run_think_aloud_cli()
        return

    if args.train_lora:
        from lora.data_formatter import build_dataset_a
        from lora.aria_lora import train_all_adapters
        print("[ARIA] Building Dataset A...")
        build_dataset_a(verbose=True)
        results = train_all_adapters(verbose=True)
        for pid, r in results.items():
            status = "OK" if r["status"] == "ok" else "ERROR"
            print(f"  {status}  {pid}: {r.get('adapter_path', r.get('error'))}")
        return

    if args.train_distilled:
        from lora.data_formatter import build_dataset_b
        from lora.distilled_lora import train_distilled_adapter, validate
        print("[ARIA] Building Dataset B...")
        build_dataset_b(verbose=True)
        path = train_distilled_adapter(verbose=True)
        print(f"[ARIA] Adapter saved to: {path}")
        validate(n_episodes_per_persona=5, verbose=True)
        return

    check_ollama()

    from memory.vector_store import VectorStore
    from memory.graph import LearningGraph, load_profile, save_profile
    from agent.reasoning import ARIAAgent
    from agent.nudge import NudgeScheduler
    import gradio as gr
    from ui.app import build_ui, set_agent, CUSTOM_CSS, ARIA_THEME

    profile = load_profile()
    if profile is None:
        profile = run_intake()
        save_profile(profile)
    else:
        print(f"\n[ARIA] Welcome back, {profile.get('name', 'friend')}!\n")

    print("[ARIA] Initialising memory store...")
    vs = VectorStore()

    print("[ARIA] Loading learning graph...")
    lg = LearningGraph()

    if lg.graph.number_of_nodes() == 0:
        print("[ARIA] Seeding learning graph from profile...")
        lg.seed_from_profile(profile)

    lora_adapter_path = None
    if args.lora:
        profile_id = args.profile_id or profile.get("profile_id")
        if profile_id:
            from lora.aria_lora import ADAPTERS_DIR, adapter_exists
            if adapter_exists(profile_id):
                lora_adapter_path = str(ADAPTERS_DIR / profile_id)
                print(f"[ARIA] Loading LoRA adapter for '{profile_id}'...")
            else:
                print(f"[ARIA] No adapter found for '{profile_id}'. Run --train-lora first.")
        else:
            print("[ARIA] --lora set but no profile_id found. Use --profile-id.")

    print("[ARIA] Starting reasoning agent...")
    agent = ARIAAgent(
        vector_store=vs,
        learning_graph=lg,
        user_profile=profile,
        lora_adapter_path=lora_adapter_path,
    )

    print("[ARIA] Starting nudge scheduler...")
    scheduler = NudgeScheduler(vector_store=vs, learning_graph=lg, user_profile=profile)
    scheduler.start()

    set_agent(agent, scheduler)
    demo = build_ui()

    print("[ARIA] Launching Gradio at http://localhost:7860\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        quiet=True,
        theme=ARIA_THEME,
        css=CUSTOM_CSS,
    )

    scheduler.stop()


if __name__ == "__main__":
    main()
