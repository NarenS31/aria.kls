"""
Unified model interface for ARIA evaluation.

Supports:
  Local (Ollama):  llama3.1:8b, mistral:7b, gemma2:9b, aria_distilled
  API:             gpt-4o (OpenAI), gemini-2.5-flash (Google)

query_model(model, messages) -> str  works identically for all six.

Rate limiting: 1 s between API calls.
Retry: 3 attempts with exponential backoff (2^n seconds).
Cost tracker: appends to data/api_costs.json.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import ollama

DATA_DIR = Path(__file__).parent.parent / "data"
COST_LOG = DATA_DIR / "api_costs.json"

# ------------------------------------------------------------------
# Optional API SDK imports
# ------------------------------------------------------------------

try:
    from openai import OpenAI as _OpenAI
    _openai_client: Optional[_OpenAI] = None
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    _openai_client = None

try:
    import google.genai as genai
    from google.genai import types as genai_types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# ------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------

OLLAMA_MODELS = {"llama3.1:8b", "mistral:7b", "gemma2:9b", "aria_distilled"}
API_MODELS    = {"gpt-4o", "gemini-2.5-flash"}
ALL_MODELS    = OLLAMA_MODELS | API_MODELS

ALL_EVAL_MODELS: list = [
    "llama3.1:8b", "mistral:7b", "gemma2:9b",
    "gpt-4o", "gemini-2.5-flash",
    "aria_distilled",
]

LOCAL_ONLY_MODELS: list = ["llama3.1:8b", "mistral:7b", "gemma2:9b"]

# Approximate cost per 1 M tokens (USD)
_COSTS_PER_M: Dict[str, Dict[str, float]] = {
    "gpt-4o":             {"input": 5.00,  "output": 15.00},
    "gemini-2.5-flash":   {"input": 0.35,  "output": 1.05},
}

# Running session cost accumulator
_session_costs: Dict[str, float] = {}


def _get_openai() -> Optional["_OpenAI"]:
    global _openai_client
    if not OPENAI_AVAILABLE:
        return None
    if _openai_client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return None
        _openai_client = _OpenAI(api_key=key)
    return _openai_client


_genai_client = None

def _get_genai():
    global _genai_client
    if not GOOGLE_AVAILABLE:
        return None
    if _genai_client is None:
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            return None
        _genai_client = genai.Client(api_key=key)
    return _genai_client


# ------------------------------------------------------------------
# Cost tracking
# ------------------------------------------------------------------

def _log_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = _COSTS_PER_M.get(model)
    if not costs:
        return 0.0
    cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
    _session_costs[model] = _session_costs.get(model, 0.0) + cost

    # Load existing log
    existing: dict = {}
    if COST_LOG.exists():
        try:
            existing = json.loads(COST_LOG.read_text())
        except Exception:
            pass
    existing[model] = round(existing.get(model, 0.0) + cost, 6)
    COST_LOG.write_text(json.dumps(existing, indent=2))
    return cost


def session_cost_summary() -> Dict[str, float]:
    return {m: round(c, 6) for m, c in _session_costs.items()}


# ------------------------------------------------------------------
# Core query with retry + rate limit
# ------------------------------------------------------------------

_last_api_call: float = 0.0
_API_MIN_INTERVAL = 1.0  # seconds between API calls


def _rate_limit() -> None:
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < _API_MIN_INTERVAL:
        time.sleep(_API_MIN_INTERVAL - elapsed)
    _last_api_call = time.time()


def query_model(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
    json_mode: bool = False,
    retries: int = 3,
) -> str:
    """
    Unified inference for any supported model.
    messages: list of {role, content} dicts (OpenAI format).
    Returns the assistant content string.
    """
    for attempt in range(retries):
        try:
            if model in OLLAMA_MODELS:
                return _query_ollama(model, messages, temperature, max_tokens, json_mode)
            elif model == "gpt-4o":
                _rate_limit()
                return _query_openai(model, messages, temperature, max_tokens, json_mode)
            elif model == "gemini-2.5-flash":
                _rate_limit()
                return _query_gemini(messages, temperature, max_tokens, json_mode)
            else:
                raise ValueError(f"Unknown model: {model}")
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            raise RuntimeError(f"Model {model} failed after {retries} retries: {e}") from e
    return ""


def is_available(model: str) -> bool:
    """Check if a model can be used in the current environment."""
    if model in OLLAMA_MODELS:
        try:
            listed = ollama.list()
            names = [m.model for m in (listed.models or [])]
            return any(model in n for n in names)
        except Exception:
            return False
    elif model == "gpt-4o":
        return OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY"))
    elif model == "gemini-2.5-flash":
        return GOOGLE_AVAILABLE and bool(os.environ.get("GOOGLE_API_KEY")) and _get_genai() is not None
    return False


def available_models() -> List[str]:
    return [m for m in ALL_MODELS if is_available(m)]


# ------------------------------------------------------------------
# Backend implementations
# ------------------------------------------------------------------

def _query_ollama(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        kwargs["format"] = "json"
    result = ollama.chat(**kwargs)
    return result.message.content.strip()


def _query_openai(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    client = _get_openai()
    if client is None:
        raise RuntimeError("OpenAI client not available — set OPENAI_API_KEY")

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""

    # Log cost
    usage = response.usage
    if usage:
        _log_cost(model, usage.prompt_tokens, usage.completion_tokens)

    return content.strip()


def _query_gemini(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    client = _get_genai()
    if client is None:
        raise RuntimeError("Gemini not available — set GOOGLE_API_KEY")

    # Convert OpenAI-style messages to Gemini format
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=m["content"])]))

    if system_parts and contents:
        # Prepend system prompt to first user message
        first = contents[0]
        merged = system_parts[0] + "\n\n" + first.parts[0].text
        contents[0] = genai_types.Content(role=first.role, parts=[genai_types.Part(text=merged)])

    if not contents:
        contents = [genai_types.Content(role="user", parts=[genai_types.Part(text="Hello")])]

    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )

    text = response.text or ""

    approx_in = sum(len(m.get("content", "")) for m in messages) // 4
    approx_out = len(text) // 4
    _log_cost("gemini-2.5-flash", approx_in, approx_out)

    return text.strip()
