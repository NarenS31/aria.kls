"""
Cognitive-state detection for the ARIA think-aloud metacognition engine.

The analyzer classifies a student's think-aloud utterance (typed OR
whisper-transcribed audio) into one of seven cognitive states:

    PLANNING, FLOW, CONFUSED, RUSHING, FRUSTRATED, STUCK, INSIGHT

Detection is a three-step cascade:

  1. Keyword heuristics (fast, runs first). Distinctive markers per state are
     matched with weights; a confidence is derived from the winning margin.
  2. LLM classification (ollama llama3.2:3b) — only consulted when the
     heuristics are not confident (< 0.8).
  3. Combine — heuristic wins when confident, otherwise the LLM is trusted
     (it is more context-aware, so it also breaks disagreements).

Alongside the state, four boolean metacognitive flags are extracted directly
from the text:

    planning_detected, self_correction, insight_moment, gave_up

Optional audio features (from listener.ThinkAloudListener) can nudge the final
classification: fast speech boosts RUSHING, many pauses boost CONFUSED/STUCK,
heavy filler use boosts CONFUSED.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import ollama
except ImportError:  # pragma: no cover - depends on local optional install
    ollama = None  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from .transfer import detect_self_initiation
except ImportError:  # Support `python3.11 metacognition/analyzer.py`.
    from metacognition.transfer import detect_self_initiation

COGNITIVE_STATES = [
    "PLANNING", "FLOW", "CONFUSED", "RUSHING",
    "FRUSTRATED", "STUCK", "INSIGHT",
]

LLM_MODEL = os.environ.get("ARIA_META_MODEL", "llama3.2:3b")


# ------------------------------------------------------------------
# Keyword marker banks
# ------------------------------------------------------------------
# Each entry: (compiled_regex, weight).  Distinctive markers carry more weight
# than ambiguous ones (e.g. "wait" appears in both CONFUSED and INSIGHT, so it
# is weighted low; "ohhh" is unmistakably INSIGHT, so it is weighted high).

def _rx(*patterns: str) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_MARKERS: dict[str, list[tuple[re.Pattern, float]]] = {
    "PLANNING": [
        (re.compile(r"\bfirst,?\s+i\s+(need|have|want|should|will|gotta)\b", re.I), 3.0),
        (re.compile(r"\bthe\s+problem\s+is\s+asking\b", re.I), 3.0),
        (re.compile(r"\b(okay|ok|alright|right),?\s+so\b", re.I), 1.5),
        (re.compile(r"\blet\s+me\s+(think|see|start|figure|break|plan)\b", re.I), 2.0),
        (re.compile(r"\bi\s+(know|remember)\s+that\b", re.I), 1.5),
        (re.compile(r"\bmy\s+(plan|strategy|approach)\b", re.I), 2.5),
        (re.compile(r"\bstep\s+(one|1|by\s+step)\b", re.I), 1.5),
        (re.compile(r"\bwhat\s+i\s+need\s+to\s+do\b", re.I), 2.0),
    ],
    "FLOW": [
        (re.compile(r"\bso\s+that\s+means\b", re.I), 2.5),
        (re.compile(r"\btherefore\b", re.I), 2.5),
        (re.compile(r"\bbecause\b", re.I), 1.2),
        (re.compile(r"\b(then|next),?\b", re.I), 1.0),
        (re.compile(r"\bwhich\s+(gives|means|equals|is)\b", re.I), 1.5),
        (re.compile(r"\bnow\s+i\b", re.I), 1.0),
        (re.compile(r"\bafter\s+that\b", re.I), 1.2),
    ],
    "CONFUSED": [
        (re.compile(r"\bi\s+don'?t\s+get\s+it\b", re.I), 3.0),
        (re.compile(r"\bi\s+don'?t\s+understand\b", re.I), 3.0),
        (re.compile(r"\bhuh+\b", re.I), 2.5),
        (re.compile(r"\bconfus(ed|ing)\b", re.I), 2.5),
        (re.compile(r"\bwait,?\s+what\b", re.I), 2.0),
        (re.compile(r"\bwhat\s+(is|does|do|even)\b", re.I), 1.2),
        (re.compile(r"\bi'?m\s+lost\b", re.I), 2.5),
        (re.compile(r"\bhow\s+(do|does|is)\b.*\?", re.I), 1.0),
        (re.compile(r"\bthis\s+is\s+confusing\b", re.I), 3.0),
    ],
    "RUSHING": [
        (re.compile(r"\bwhatever\b", re.I), 2.5),
        (re.compile(r"\bi'?ll\s+just\b", re.I), 2.5),
        (re.compile(r"\bprobably\b", re.I), 1.5),
        (re.compile(r"\bdone[.!]?\s*$", re.I), 2.0),
        (re.compile(r"\bjust\s+(use|do|write|pick|go)\b", re.I), 1.5),
        (re.compile(r"\bthat'?s?\s+(it|good enough)\b", re.I), 1.5),
        (re.compile(r"\btoo\s+easy\b", re.I), 1.5),
        (re.compile(r"\beh,?\b", re.I), 1.0),
    ],
    "FRUSTRATED": [
        (re.compile(r"\bugh+\b", re.I), 3.0),
        (re.compile(r"\bthis\s+makes\s+no\s+sense\b", re.I), 3.0),
        (re.compile(r"\bi\s+hate\s+(this|it)\b", re.I), 3.0),
        (re.compile(r"\bso\s+(annoying|frustrating)\b", re.I), 2.5),
        (re.compile(r"\bargh+\b", re.I), 2.5),
        (re.compile(r"\bwhy\s+(won'?t|doesn'?t|can'?t)\b", re.I), 1.5),
        (re.compile(r"\bstupid\b", re.I), 1.5),
        (re.compile(r"!{2,}", re.I), 1.2),
    ],
    "STUCK": [
        (re.compile(r"\bi\s+(really\s+)?don'?t\s+know\b", re.I), 1.8),
        (re.compile(r"\[pause\]", re.I), 2.5),
        (re.compile(r"\bidk\b", re.I), 1.5),
        (re.compile(r"\bi\s+give\s+up\b", re.I), 2.0),
        (re.compile(r"\bi\s+can'?t\s+(do|figure)\b", re.I), 2.0),
        (re.compile(r"\bno\s+idea\b", re.I), 2.5),
        (re.compile(r"\bi'?m\s+stuck\b", re.I), 3.0),
        (re.compile(r"\bblank\b", re.I), 1.5),
    ],
    "INSIGHT": [
        (re.compile(r"\boh+!*\b", re.I), 1.5),
        (re.compile(r"\bohh+\b", re.I), 3.0),
        (re.compile(r"\bi\s+(see|get\s+it)\b", re.I), 2.5),
        (re.compile(r"\bgot\s+it\b", re.I), 2.5),
        (re.compile(r"\bthat\s+means\b", re.I), 1.8),
        (re.compile(r"\bit\s+click(ed|s)\b", re.I), 3.0),
        (re.compile(r"\bof\s+course\b", re.I), 2.0),
        (re.compile(r"\bthat'?s\s+(it|the answer)\b", re.I), 2.0),
        (re.compile(r"\bnow\s+i\s+(get|see|understand)\b", re.I), 2.5),
    ],
}

# Flag markers -------------------------------------------------------
_PLANNING_FLAG = _MARKERS["PLANNING"]
_INSIGHT_FLAG = _MARKERS["INSIGHT"]

_SELF_CORRECTION = _rx(
    r"\bno,?\s+wait\b",
    r"\bactually\b",
    r"\bhold\s+on\b",
    r"\bscratch\s+that\b",
    r"\bthat\s+can'?t\s+be\s+right\b",
    r"\bwait,?\s+no\b",
    r"\bi\s+meant\b",
    r"\bnever\s?mind\b",
    r"\blet\s+me\s+redo\b",
    r"\bwrong,?\s+it'?s\b",
)

_GAVE_UP = _rx(
    r"\bi\s+give\s+up\b",
    r"\bforget\s+it\b",
    r"\bi\s+can'?t\s+do\s+this\b",
    r"\bi\s+quit\b",
    r"\bwhatever,?\s+i'?m\s+done\b",
    r"\bnot\s+even\s+going\s+to\s+try\b",
)


def _any(patterns: list, text: str) -> bool:
    return any(p.search(text) for p in patterns)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class AnalysisResult:
    state: str
    confidence: float
    evidence: str
    method: str  # "heuristic" | "llm" | "heuristic+audio" | "llm+audio"
    planning_detected: bool = False
    self_correction: bool = False
    insight_moment: bool = False
    gave_up: bool = False
    # Transfer: did the student self-initiate metacognition (unprompted)?
    self_initiated_metacognition: bool = False
    metacognitive_type: str = "none"
    prompted_by_aria: bool = False
    heuristic_state: Optional[str] = None
    heuristic_confidence: float = 0.0
    llm_state: Optional[str] = None
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "method": self.method,
            "planning_detected": self.planning_detected,
            "self_correction": self.self_correction,
            "insight_moment": self.insight_moment,
            "gave_up": self.gave_up,
            "self_initiated_metacognition": self.self_initiated_metacognition,
            "metacognitive_type": self.metacognitive_type,
            "prompted_by_aria": self.prompted_by_aria,
            "heuristic_state": self.heuristic_state,
            "heuristic_confidence": round(self.heuristic_confidence, 3),
            "llm_state": self.llm_state,
        }


# ------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------

class CognitiveStateAnalyzer:
    """Detect a student's cognitive state from a think-aloud utterance."""

    HEURISTIC_TRUST = 0.8  # heuristic wins at or above this confidence

    def __init__(self, model: str = LLM_MODEL, use_llm: bool = True):
        self.model = model
        self.use_llm = use_llm

    # -- public API --------------------------------------------------

    def analyze(
        self,
        text: str,
        audio_features: Optional[dict] = None,
        aria_previous_prompt: str = "",
    ) -> dict[str, Any]:
        """Classify `text` and return a result dict (see AnalysisResult).

        Before state detection, transfer detection checks whether the student
        self-initiated metacognition (planning/monitoring/reflection) without
        ARIA prompting it. `aria_previous_prompt` is ARIA's last message: if it
        was a metacognitive question, metacognition here is prompted, not
        self-initiated. The result is folded into the returned dict so callers
        can acknowledge self-initiation and avoid re-prompting a habit the
        student already showed.
        """
        text = (text or "").strip()
        transfer = detect_self_initiation(text, aria_previous_prompt)
        flags = self._extract_flags(text)

        h_state, h_conf, h_evidence, scores = self._heuristic(text)

        if h_conf >= self.HEURISTIC_TRUST or not self.use_llm:
            state, conf, evidence, method = h_state, h_conf, h_evidence, "heuristic"
            llm_state = None
        else:
            llm = self._llm_classify(text, h_state)
            if llm is None:
                # LLM unavailable — fall back to whatever the heuristic found.
                state, conf, evidence, method = h_state, h_conf, h_evidence, "heuristic"
                llm_state = None
            else:
                llm_state = llm["state"]
                # Step 3: heuristic unclear -> trust LLM (also breaks disagreements)
                state = llm_state
                conf = float(llm["confidence"])
                evidence = llm["evidence"]
                method = "llm"

        # Audio features can override / nudge the classification.
        if audio_features:
            state, conf, evidence, method = self._apply_audio(
                state, conf, evidence, method, scores, audio_features
            )

        result = AnalysisResult(
            state=state,
            confidence=conf,
            evidence=evidence,
            method=method,
            planning_detected=flags["planning_detected"],
            self_correction=flags["self_correction"],
            insight_moment=flags["insight_moment"],
            gave_up=flags["gave_up"],
            self_initiated_metacognition=transfer["self_initiated_metacognition"],
            metacognitive_type=transfer["metacognitive_type"],
            prompted_by_aria=transfer["prompted_by_aria"],
            heuristic_state=h_state,
            heuristic_confidence=h_conf,
            llm_state=llm_state,
            scores=scores,
        )
        return result.to_dict()

    # -- step 1: heuristics -----------------------------------------

    def _heuristic(self, text: str) -> tuple[str, float, str, dict]:
        if not text:
            return "STUCK", 0.5, "empty response", {}

        words = re.findall(r"\w+", text.lower())
        n_words = len(words)

        scores: dict[str, float] = {s: 0.0 for s in COGNITIVE_STATES}
        hit_terms: dict[str, list[str]] = {s: [] for s in COGNITIVE_STATES}

        for state, markers in _MARKERS.items():
            for pat, weight in markers:
                m = pat.search(text)
                if m:
                    scores[state] += weight
                    hit_terms[state].append(m.group(0).strip())

        # --- structural / length signals ---
        # Single-word or one-token responses -> STUCK.
        if n_words <= 1:
            scores["STUCK"] += 3.0
            hit_terms["STUCK"].append("single-word response")
        # Very short response (< 10 words) with no strong other signal -> RUSHING.
        elif n_words < 10:
            scores["RUSHING"] += 1.5
            hit_terms["RUSHING"].append("very short response")

        # Repeated "i don't know" / "idk idk idk" -> STUCK.
        idk_count = len(re.findall(r"\bi\s+don'?t\s+know\b|\bidk\b", text, re.I))
        if idk_count >= 2:
            scores["STUCK"] += 3.0
            hit_terms["STUCK"].append(f"'don't know' x{idk_count}")

        # Circular repetition (a phrase repeated) -> CONFUSED.
        if self._has_circular_repetition(words):
            scores["CONFUSED"] += 2.0
            hit_terms["CONFUSED"].append("circular repetition")

        # Multi-step reasoning (several connectives) -> FLOW.
        connectives = len(re.findall(r"\b(then|next|so|because|therefore|thus)\b", text, re.I))
        if connectives >= 3:
            scores["FLOW"] += 2.0
            hit_terms["FLOW"].append(f"{connectives} reasoning connectives")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_state, top_score = ranked[0]
        second_score = ranked[1][1]

        if top_score <= 0:
            # No markers at all — weakly guess FLOW (neutral narration) and
            # defer to the LLM by returning a low confidence.
            return "FLOW", 0.0, "no distinctive markers", scores

        # Confidence blends absolute evidence with the winning margin.
        margin = top_score - second_score
        confidence = 0.45 + 0.09 * top_score + 0.12 * margin
        confidence = max(0.0, min(0.97, confidence))

        evidence = f"markers: {', '.join(hit_terms[top_state][:4])}"
        return top_state, confidence, evidence, scores

    @staticmethod
    def _has_circular_repetition(words: list[str]) -> bool:
        """Detect a short phrase repeated back-to-back (a confusion signal)."""
        if len(words) < 4:
            return False
        # Repeated single content words (e.g. "what what what").
        counts = Counter(words)
        for w, c in counts.items():
            if len(w) > 2 and c >= 3:
                return True
        # Repeated bigrams.
        bigrams = list(zip(words, words[1:]))
        bg_counts = Counter(bigrams)
        return any(c >= 2 for c in bg_counts.values() if c >= 2) and len(words) < 25

    # -- flags -------------------------------------------------------

    def _extract_flags(self, text: str) -> dict[str, bool]:
        planning = _any([p for p, _ in _PLANNING_FLAG], text)
        insight = _any([p for p, _ in _INSIGHT_FLAG], text)
        # An insight is a form of self-correction only when a prior idea was
        # revised; treat explicit correction markers OR insight-after-doubt.
        self_corr = _any(_SELF_CORRECTION, text)
        if insight and re.search(r"\b(no|but|wait|isn'?t|can'?t\s+be)\b", text, re.I):
            self_corr = True
        gave_up = _any(_GAVE_UP, text)
        return {
            "planning_detected": planning,
            "self_correction": self_corr,
            "insight_moment": insight,
            "gave_up": gave_up,
        }

    # -- step 2: LLM -------------------------------------------------

    def _llm_classify(self, text: str, heuristic_hint: str) -> Optional[dict]:
        if ollama is None:
            return None
        prompt = (
            "Classify this student think-aloud. Output JSON only:\n"
            '{"state": "PLANNING|FLOW|CONFUSED|RUSHING|FRUSTRATED|STUCK|INSIGHT",\n'
            ' "confidence": 0.0-1.0,\n'
            ' "evidence": "one sentence explaining why"}\n\n'
            f"Think-aloud: \"{text}\"\n"
        )
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                        "You classify a student's cognitive state from their think-aloud. "
                        "Respond with JSON only, no prose."},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.0, "num_predict": 150},
            )
            raw = resp.message.content
            data = self._parse_json(raw)
            state = str(data.get("state", "")).strip().upper()
            if state not in COGNITIVE_STATES:
                # Best-effort: pick the first valid state token present.
                for s in COGNITIVE_STATES:
                    if s in state:
                        state = s
                        break
                else:
                    state = heuristic_hint if heuristic_hint in COGNITIVE_STATES else "CONFUSED"
            conf = data.get("confidence", 0.6)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.6
            conf = max(0.0, min(1.0, conf))
            evidence = str(data.get("evidence", "")).strip() or "LLM classification"
            return {"state": state, "confidence": conf, "evidence": evidence}
        except Exception:
            return None

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))

    # -- audio override ---------------------------------------------

    def _apply_audio(
        self,
        state: str,
        conf: float,
        evidence: str,
        method: str,
        scores: dict,
        audio: dict,
    ) -> tuple[str, float, str, str]:
        """Nudge classification using prosodic features from the mic."""
        speech_rate = audio.get("speech_rate", 0.0) or 0.0
        pause_count = audio.get("pause_count", 0) or 0
        filler_rate = audio.get("filler_rate", 0.0) or 0.0

        boosts: dict[str, float] = {}
        notes: list[str] = []
        if speech_rate > 180:
            boosts["RUSHING"] = boosts.get("RUSHING", 0) + 2.0
            notes.append(f"fast speech {speech_rate:.0f}wpm")
        if pause_count > 3:
            boosts["CONFUSED"] = boosts.get("CONFUSED", 0) + 1.0
            boosts["STUCK"] = boosts.get("STUCK", 0) + 1.0
            notes.append(f"{pause_count} pauses")
        if filler_rate > 10:
            boosts["CONFUSED"] = boosts.get("CONFUSED", 0) + 1.5
            notes.append(f"filler {filler_rate:.0f}/min")

        if not boosts:
            return state, conf, evidence, method

        # Combine text scores with audio boosts and re-rank when audio is
        # strong enough to overturn a low-confidence text classification.
        combined = dict(scores) if scores else {s: 0.0 for s in COGNITIVE_STATES}
        combined[state] = combined.get(state, 0.0) + conf * 3.0  # anchor text pick
        for s, b in boosts.items():
            combined[s] = combined.get(s, 0.0) + b

        new_state = max(combined, key=combined.get)
        if new_state != state:
            evidence = f"{evidence}; audio override ({', '.join(notes)})"
            conf = min(0.95, conf + 0.1)
        else:
            evidence = f"{evidence}; audio confirms ({', '.join(notes)})"
            conf = min(0.98, conf + 0.05)
        return new_state, conf, evidence, method + "+audio"


# ------------------------------------------------------------------
# Evaluation on the synthetic dataset
# ------------------------------------------------------------------

def _join_think_aloud(ta: Any) -> str:
    if isinstance(ta, str):
        return ta
    if isinstance(ta, dict):
        parts = [ta.get("pre_attempt", ""), ta.get("during_attempt", ""),
                 ta.get("post_attempt", "")]
        return " ".join(p for p in parts if p)
    return str(ta)


def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def evaluate(dataset_path: str, use_llm: bool = True, limit: int = 0) -> dict:
    """Run the analyzer over a labelled dataset and report per-state metrics.

    Returns a dict with accuracy plus precision/recall/F1 per state and a
    confusion matrix; also prints the confusion matrix to stdout.
    """
    records = _load_jsonl(dataset_path)
    if limit:
        records = records[:limit]
    analyzer = CognitiveStateAnalyzer(use_llm=use_llm)

    y_true: list[str] = []
    y_pred: list[str] = []
    for rec in records:
        text = _join_think_aloud(rec.get("think_aloud", ""))
        gt = rec.get("cognitive_state", "")
        pred = analyzer.analyze(text)["state"]
        y_true.append(gt)
        y_pred.append(pred)

    metrics = compute_classification_metrics(y_true, y_pred, COGNITIVE_STATES)
    print_confusion_matrix(metrics["confusion_matrix"], COGNITIVE_STATES)
    print(f"\nOverall accuracy: {metrics['accuracy']:.1%}  (n={len(records)})")
    return metrics


def compute_classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict:
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1

    correct = sum(cm[i][i] for i in range(n))
    total = sum(sum(row) for row in cm)
    accuracy = correct / total if total else 0.0

    per_state = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_state[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }

    return {
        "accuracy": round(accuracy, 4),
        "n": total,
        "per_state": per_state,
        "confusion_matrix": cm,
        "labels": labels,
    }


def print_confusion_matrix(cm: list[list[int]], labels: list[str]) -> None:
    abbr = [l[:4] for l in labels]
    header = "true\\pred".ljust(11) + "".join(a.rjust(6) for a in abbr)
    print("\n" + header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = label[:10].ljust(11) + "".join(str(cm[i][j]).rjust(6) for j in range(len(labels)))
        print(row)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate CognitiveStateAnalyzer.")
    ap.add_argument("--dataset", default=os.path.join(
        REPO_ROOT, "data", "synthetic_thinkaloud", "test.jsonl"))
    ap.add_argument("--no-llm", action="store_true",
                    help="Heuristics only (skip the LLM fallback).")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--text", type=str, default=None,
                    help="Classify a single string instead of the dataset.")
    args = ap.parse_args()

    if args.text is not None:
        a = CognitiveStateAnalyzer(use_llm=not args.no_llm)
        print(json.dumps(a.analyze(args.text), indent=2))
    else:
        evaluate(args.dataset, use_llm=not args.no_llm, limit=args.limit)
