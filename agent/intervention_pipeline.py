"""Closed-loop intervention selection for ARIA student practice.

The pipeline separates generation from selection. A language model may propose
candidate interventions, but deterministic checks decide what reaches a
student. The selected strategy is then evaluated against the student's next
turn so future selections can use observed outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from agent.learner_model import LearnerModelStore

STRATEGIES = {
    "error_localization",
    "smallest_step",
    "contrast_case",
    "self_explanation",
    "retrieval_cue",
    "verification",
    "planning",
    "reflection",
}

NEGATIVE_STATES = {"CONFUSED", "RUSHING", "FRUSTRATED", "STUCK"}
POSITIVE_STATES = {"PLANNING", "FLOW", "INSIGHT"}
_SIMILARITY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can",
    "could", "did", "do", "does", "for", "from", "how", "i", "if", "in",
    "inside", "is", "it", "must", "of", "on", "or", "should", "that",
    "the", "this", "to", "was", "what", "when", "which", "will", "with",
    "would", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return [
        word for word in re.findall(r"[a-z0-9.]+", (text or "").lower())
        if len(word) >= 2 and word not in _SIMILARITY_STOPWORDS
    ]


def semantic_similarity(left: str, right: str) -> float:
    """Lightweight semantic-overlap score using unigram and bigram cosine."""
    def features(text: str) -> Counter:
        words = _tokens(text)
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return Counter(grams)

    a = features(left)
    b = features(right)
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class InterventionOutcomeStore:
    """Append-only outcome log plus in-memory strategy effectiveness."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._stats = {strategy: {"success": 0, "total": 0} for strategy in STRATEGIES}
        self._student_stats = {}
        self._topic_stats = {}
        self._context_stats = {}
        self._load()

    @staticmethod
    def _increment(stats: dict, key: str, effective: bool) -> None:
        bucket = stats.setdefault(key, {"success": 0, "total": 0})
        bucket["total"] += 1
        bucket["success"] += int(bool(effective))

    def _observe_record(self, record: dict) -> None:
        strategy = record.get("strategy")
        if strategy not in self._stats:
            return
        effective = bool(record.get("effective"))
        self._stats[strategy]["total"] += 1
        self._stats[strategy]["success"] += int(effective)
        signature = record.get("signature", {}) or {}
        student = str(signature.get("student", "")).strip()
        topic = str(signature.get("topic", "")).strip()
        if student:
            self._increment(
                self._student_stats,
                f"{student}::{strategy}",
                effective,
            )
        if topic:
            self._increment(
                self._topic_stats,
                f"{topic}::{strategy}",
                effective,
            )
        context_key = self._context_key(signature, strategy)
        self._increment(self._context_stats, context_key, effective)

    @staticmethod
    def _context_key(signature: dict, strategy: str) -> str:
        return "::".join([
            str(signature.get("state", "UNKNOWN")),
            str(signature.get("mastery_band", "unknown")),
            "misconception" if signature.get("misconception") else "no_misconception",
            strategy,
        ])

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open() as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._observe_record(record)
        except OSError:
            pass

    @staticmethod
    def _smoothed(bucket: dict) -> float:
        return (bucket.get("success", 0) + 1) / (bucket.get("total", 0) + 2)

    def effectiveness(
        self,
        strategy: str,
        *,
        student: str = "",
        topic: str = "",
        signature: Optional[dict] = None,
    ) -> float:
        stats = self._stats.get(strategy, {"success": 0, "total": 0})
        # Beta(1,1) smoothing prevents one early outcome from dominating.
        global_rate = self._smoothed(stats)
        rates = [(global_rate, 0.25)]
        if student:
            rates.append((
                self._smoothed(
                    self._student_stats.get(
                        f"{student}::{strategy}",
                        {"success": 0, "total": 0},
                    )
                ),
                0.5,
            ))
        if topic:
            rates.append((
                self._smoothed(
                    self._topic_stats.get(
                        f"{topic}::{strategy}",
                        {"success": 0, "total": 0},
                    )
                ),
                0.25,
            ))
        if signature:
            context_bucket = self._context_stats.get(
                self._context_key(signature, strategy),
                {"success": 0, "total": 0},
            )
            # Context-specific evidence matters more as observations accrue.
            context_weight = min(0.75, context_bucket.get("total", 0) / 12)
            if context_weight:
                rates.append((self._smoothed(context_bucket), context_weight))
        total_weight = sum(weight for _, weight in rates)
        return sum(rate * weight for rate, weight in rates) / total_weight

    def policy_evidence(self, strategy: str, signature: dict) -> dict:
        bucket = self._context_stats.get(
            self._context_key(signature, strategy),
            {"success": 0, "total": 0},
        )
        total = int(bucket.get("total", 0))
        success = int(bucket.get("success", 0))
        alpha = success + 1
        beta = total - success + 1
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / (
            (alpha + beta) ** 2 * (alpha + beta + 1)
        )
        radius = 1.96 * math.sqrt(variance)
        return {
            "context": self._context_key(signature, strategy),
            "observations": total,
            "posterior_mean": round(mean, 4),
            "ci_95": [
                round(max(0.0, mean - radius), 4),
                round(min(1.0, mean + radius), 4),
            ],
        }

    def record(self, record: dict) -> None:
        strategy = record.get("strategy", "smallest_step")
        if strategy not in self._stats:
            strategy = "smallest_step"
            record["strategy"] = strategy
        self._observe_record(record)
        record = {**record, "recorded_at": datetime.now().isoformat()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass


class ClosedLoopInterventionPipeline:
    """Generate-safe candidates, score them, and learn from the next turn."""

    def __init__(self, outcome_path: Path):
        self.outcomes = InterventionOutcomeStore(outcome_path)
        self.learner_model = LearnerModelStore(
            Path(outcome_path).with_name("learner_model.json")
        )
        self.pending: Optional[dict] = None

    @staticmethod
    def build_signature(
        *,
        student: str,
        task_id: str,
        topic: str,
        problem_step: str,
        student_anchor: str,
        misconception: str,
        state: str,
        style: str,
        prior_outcome: str,
        key_ideas: Optional[list[str]] = None,
        mastery_mean: float = 0.5,
        mastery_band: str = "unknown",
    ) -> dict:
        signature = {
            "student": student,
            "task_id": task_id,
            "topic": topic,
            "problem_step": problem_step,
            "student_anchor": student_anchor,
            "misconception": misconception,
            "state": state,
            "style": style,
            "prior_outcome": prior_outcome,
            "key_ideas": list(key_ideas or []),
            "mastery_mean": round(float(mastery_mean), 4),
            "mastery_band": mastery_band,
        }
        canonical = json.dumps(signature, sort_keys=True, ensure_ascii=False)
        signature["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return signature

    def learner_state(
        self,
        *,
        student: str,
        topic: str,
        key_ideas: list[str],
        misconception: str = "",
    ) -> dict:
        return self.learner_model.state(
            student=student,
            topic=topic,
            skills=key_ideas,
            misconception=misconception,
        )

    @staticmethod
    def parse_model_candidates(raw: str) -> list[dict]:
        cleaned = re.sub(
            r"^```(?:json)?|```$",
            "",
            (raw or "").strip(),
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return []
        items = parsed.get("candidates", []) if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            return []
        candidates = []
        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            strategy = str(item.get("strategy", "")).strip().lower()
            if text and strategy in STRATEGIES:
                candidates.append({"text": text, "strategy": strategy, "source": "model"})
        return candidates

    @staticmethod
    def safe_candidates(
        *,
        anchor: str,
        fallback: str,
        key_idea: str,
        state: str,
        has_prior_turn: bool,
        repeat_anchor: bool = True,
    ) -> list[dict]:
        quoted = f"“{anchor}”"
        idea = key_idea or "the first required operation"
        first = f"You wrote {quoted}. {fallback}" if repeat_anchor else fallback
        if not repeat_anchor:
            second = (
                f"Focus on {idea}: which detail in the original problem gives "
                "you a useful place to begin?"
            )
        elif has_prior_turn:
            second = (
                f"You are revisiting {quoted}. Focus on {idea}: "
                "what changed in your reasoning since the previous attempt?"
            )
        else:
            second = (
                f"Your decision point is {quoted}. Focus on {idea}: "
                "which part of the problem supports that move?"
            )
        if not repeat_anchor:
            third = (
                f"Use {idea} as the starting point. What is one detail you can "
                "identify before calculating?"
            )
        elif state in NEGATIVE_STATES:
            third = (
                f"Your uncertainty is centered on {quoted}. Using only {idea}, "
                "what is one line you can verify before continuing?"
            )
        else:
            third = (
                f"Your current plan is {quoted}. Before carrying it out, "
                f"how will you verify the {idea} step?"
            )
        return [
            {
                "text": first,
                "strategy": "error_localization",
                "source": "verified",
                "grounding_strength": 1.0,
                "verified_step_evidence": True,
            },
            {
                "text": second,
                "strategy": "self_explanation",
                "source": "verified",
                "grounding_strength": 0.65,
            },
            {
                "text": third,
                "strategy": "verification",
                "source": "verified",
                "grounding_strength": 0.65,
            },
        ]

    def score_candidates(
        self,
        *,
        candidates: Iterable[dict],
        student_input: str,
        recent_responses: list[str],
        key_ideas: list[str],
        correct_answer: str,
        validator: Callable[[str, str], bool],
        state: str,
        misconception_present: bool = False,
        signature: Optional[dict] = None,
        preferred_words: int = 55,
        max_similarity: float = 0.78,
    ) -> list[dict]:
        student_tokens = set(_tokens(student_input))
        idea_tokens = set(_tokens(" ".join(key_ideas)))
        answer_norm = " ".join(_tokens(correct_answer))
        student_norm = " ".join(_tokens(student_input))
        ranked = []

        for candidate in candidates:
            text = str(candidate.get("text", "")).strip()
            strategy = str(candidate.get("strategy", "smallest_step"))
            grounding_strength = max(
                0.0, min(1.0, float(candidate.get("grounding_strength", 0.5)))
            )
            verified_step_evidence = bool(
                candidate.get("verified_step_evidence", False)
            )
            valid = validator(text, student_input)
            reasons = []
            score = 0.0

            if not valid:
                reasons.append("failed_grounding")
                score -= 100
            else:
                score += grounding_strength * 2.5
                if verified_step_evidence:
                    score += 3.0

            similarities = [semantic_similarity(text, old) for old in recent_responses]
            closest = max(similarities, default=0.0)
            if closest >= max_similarity:
                valid = False
                reasons.append("semantic_repeat")
                score -= 100
            else:
                score += (1.0 - closest) * 2.5

            text_tokens = set(_tokens(text))
            student_overlap = len(student_tokens & text_tokens) / max(len(student_tokens), 1)
            idea_overlap = len(idea_tokens & text_tokens) / max(len(idea_tokens), 1)
            score += min(student_overlap, 0.5) * 5
            score += min(idea_overlap, 0.6) * 3

            if text.count("?") == 1:
                score += 1.5
            else:
                reasons.append("question_contract")

            candidate_answer = " ".join(_tokens(text))
            if answer_norm and answer_norm in candidate_answer and answer_norm not in student_norm:
                valid = False
                reasons.append("reveals_answer")
                score -= 100

            signature = signature or {}
            historical = self.outcomes.effectiveness(
                strategy,
                student=str(signature.get("student", "")),
                topic=str(signature.get("topic", "")),
                signature=signature,
            )
            policy_evidence = self.outcomes.policy_evidence(strategy, signature)
            score += historical * 2
            if state in NEGATIVE_STATES and strategy in {"error_localization", "smallest_step"}:
                score += 1.5
            elif state == "PLANNING" and strategy in {"planning", "verification"}:
                score += 1.5
            elif state in {"FLOW", "INSIGHT"} and strategy in {"self_explanation", "reflection"}:
                score += 1.5
            if misconception_present and strategy == "error_localization":
                score += 3.0

            mastery_band = str(signature.get("mastery_band", ""))
            if mastery_band == "developing" and strategy in {
                "smallest_step", "retrieval_cue", "error_localization"
            }:
                score += 1.25
            elif mastery_band == "emerging" and strategy in {
                "contrast_case", "self_explanation", "verification"
            }:
                score += 1.25
            elif mastery_band == "secure" and strategy in {
                "reflection", "planning", "self_explanation"
            }:
                score += 1.25

            if len(text.split()) <= preferred_words:
                score += 0.5
            else:
                score -= 1.0

            ranked.append({
                **candidate,
                "valid": valid,
                "score": round(score, 4),
                "closest_prior_similarity": round(closest, 4),
                "historical_effectiveness": round(historical, 4),
                "grounding_strength": round(grounding_strength, 4),
                "verified_step_evidence": verified_step_evidence,
                "policy_evidence": policy_evidence,
                "rejection_reasons": reasons,
            })

        return sorted(ranked, key=lambda item: (item["valid"], item["score"]), reverse=True)

    def select(
        self,
        *,
        candidates: Iterable[dict],
        student_input: str,
        recent_responses: list[str],
        key_ideas: list[str],
        correct_answer: str,
        validator: Callable[[str, str], bool],
        signature: dict,
        state: str,
        preferred_words: int = 55,
    ) -> tuple[Optional[str], dict]:
        ranked = self.score_candidates(
            candidates=candidates,
            student_input=student_input,
            recent_responses=recent_responses,
            key_ideas=key_ideas,
            correct_answer=correct_answer,
            validator=validator,
            state=state,
            misconception_present=bool(signature.get("misconception")),
            signature=signature,
            preferred_words=preferred_words,
        )
        selected = next((candidate for candidate in ranked if candidate["valid"]), None)
        meta = {
            "signature": signature,
            "candidate_count": len(ranked),
            "valid_candidate_count": sum(bool(item["valid"]) for item in ranked),
            "semantic_repeats_blocked": sum(
                "semantic_repeat" in item["rejection_reasons"] for item in ranked
            ),
            "ranked_candidates": ranked,
            "selected_strategy": selected.get("strategy") if selected else None,
            "selected_source": selected.get("source") if selected else None,
        }
        if selected:
            self.pending = {
                "signature": signature,
                "strategy": selected["strategy"],
                "state_before": state,
                "student_input": student_input,
                "response": selected["text"],
            }
            return selected["text"], meta
        return None, meta

    def observe_next_turn(
        self,
        *,
        state: str,
        student_input: str,
        state_confidence: float = 1.0,
        correct: Optional[bool] = None,
        misconception_persisted: Optional[bool] = None,
    ) -> Optional[dict]:
        if not self.pending:
            return None
        previous = self.pending
        prior_state = previous.get("state_before", "")
        lower = (student_input or "").lower()
        correction_language = any(
            phrase in lower
            for phrase in ("wait", "i see", "instead", "actually", "that means", "so now")
        )
        signature = previous.get("signature", {}) or {}
        target_tokens = set(_tokens(
            " ".join(signature.get("key_ideas", []))
            + " "
            + str(signature.get("problem_step", ""))
        ))
        before_tokens = set(_tokens(previous.get("student_input", "")))
        after_tokens = set(_tokens(student_input))
        new_target_tokens = (after_tokens & target_tokens) - before_tokens
        explanatory_turn = len(after_tokens) >= 5
        grounded_progress = bool(new_target_tokens) or (
            not target_tokens and correction_language and explanatory_turn
        )
        self_correction = correction_language and grounded_progress
        recovered = (
            prior_state in NEGATIVE_STATES
            and state in POSITIVE_STATES
            and float(state_confidence) >= 0.65
        )
        remained_stuck = prior_state in NEGATIVE_STATES and state in NEGATIVE_STATES
        effective = bool(correct) or self_correction or (
            recovered and grounded_progress
        )
        evidence_strength = min(
            1.0,
            0.25 * len(new_target_tokens)
            + 0.25 * int(correction_language)
            + 0.25 * int(recovered)
            + 0.25 * int(correct is not None),
        )
        learner_state = self.learner_model.observe(
            student=str(signature.get("student", "student")),
            topic=str(signature.get("topic", "")),
            skills=list(signature.get("key_ideas", [])),
            misconception=str(signature.get("misconception", "")),
            correct=correct,
            grounded_progress=grounded_progress,
            self_correction=self_correction,
            misconception_persisted=misconception_persisted,
            evidence_strength=evidence_strength,
        )
        outcome = {
            **previous,
            "state_after": state,
            "next_student_input": student_input,
            "recovered": recovered,
            "self_correction": self_correction,
            "correction_language": correction_language,
            "grounded_progress": grounded_progress,
            "new_target_tokens": sorted(new_target_tokens),
            "evidence_strength": round(evidence_strength, 4),
            "remained_stuck": remained_stuck,
            "effective": effective,
            "learner_state_after": learner_state,
        }
        self.outcomes.record(outcome)
        self.pending = None
        return outcome
