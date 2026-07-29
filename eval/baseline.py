"""
Vanilla baseline: llama3.1:8b with zero memory, zero graph, zero ADHD features.
Identical interface to ARIAAgent so runner.py can swap them without changes.
"""

import ollama


_BASELINE_SYSTEM = """You are a helpful tutoring assistant.
Answer the student's questions clearly and accurately.
Be encouraging and patient."""


class BaselineAgent:
    """
    Vanilla Ollama tutor. No personalization, no memory, no graph, no adaptation.
    Maintains only the current session's in-context history (last 12 turns).
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self._history = []

    def chat(self, user_input: str) -> str:
        self._history.append({"role": "user", "content": user_input})

        msgs = [{"role": "system", "content": _BASELINE_SYSTEM}]
        msgs.extend(self._history[-12:])

        result = ollama.chat(
            model=self.model,
            messages=msgs,
            options={"temperature": 0.7, "num_predict": 512},
        )
        response = result.message.content.strip()
        self._history.append({"role": "assistant", "content": response})
        return response

    def reset(self) -> None:
        self._history = []

    def get_stats(self) -> dict:
        return {
            "total_conversations": len(self._history) // 2,
            "topics_tracked": 0,
            "struggling_topics": [],
            "peak_focus_hours": [],
            "topic_details": [],
        }
