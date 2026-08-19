#!/usr/bin/env python3.11
"""Validation suite for the RQ2 stimulus set produced by eval/rq2_pipeline.py.

Run after every regeneration. The pairs are simulator-generated, so the file
changes on each run and the checks below are what stands between a bad stimulus
and a rater seeing it.

    python3.11 eval/rq2_validate.py             # checks + full 10-pair preview
    python3.11 eval/rq2_validate.py --quiet     # pass/fail summary only
    python3.11 eval/rq2_validate.py --file data/rq2_pairs_FROZEN.json

The study's canonical stimulus set is the frozen file, not the live pipeline
output, so validate that path explicitly: `--file` reads any pairs file, in
either the bare-list shape the pipeline writes or the frozen shape that wraps
the same list in a header block.

Checks:
  a  schema           every required key present on every pair
  b  display_order    only "scaffolded" / "answer_giving"
  c  quote anchor     >= 3 words, never a lone connective ("but", "and", ...)
  d  answer leakage   answer in 0/10 scaffolded, 10/10 answer-giving
  e  artifacts        no bare leading quote, no simulator stage directions
  f  unblinding key   _hidden.shown_as_A == display_order.shown_as_A
  g  unblinding key   _hidden.shown_as_B is the complement of shown_as_A
  h  response length  mean words per condition + the measured ratio
  i  stimulus         every pair intervened, no weak stimuli
  j  question marks   no response carries more than one "?"

Check (h) never fails: the residual length gap between the two conditions is a
measurement to report in the paper, not a bar to clear. It prints the number so
the write-up can state it instead of assuming it away.

Check (j) duplicates one clause of ARIAAgent._response_passes_grounding on
purpose. That guard is not applied on the path this pipeline takes, so a second
question mark — the student's own, carried in on the tail of a quoted anchor —
reached the stimulus file twice before this check existed. It is enforced here
over every response in the set, independently of the agent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.rq2_pipeline import ANSWER_GIVING, PAIRS_PATH, SCAFFOLDED  # noqa: E402

REQUIRED_KEYS = (
    "problem_id",
    "problem_text",
    "student_input",
    "display_order",
    "response_scaffolded",
    "response_answer_giving",
    "_hidden",
)
REQUIRED_HIDDEN_KEYS = (
    "shown_as_A",
    "shown_as_B",
    "seed",
    "subject",
    "topic",
    "difficulty",
    "task_type",
    "persona",
    "attempt_tries",
    "weak_stimulus",
    "scaffolded_meta",
    "answer_giving_meta",
)

MIN_ANCHOR_WORDS = 3
CONNECTIVES = {
    "but", "and", "so", "then", "or", "if", "because", "since", "also", "well",
}

# Deliberately NOT imported from the pipeline. This check exists to catch what
# the pipeline's cleaner missed, so sharing its pattern would make it agree with
# the bug. Kept broader on purpose: the vocabulary the cleaner knows about, plus
# any parenthetical that reads like a stage direction ("(muttering to self)").
# A false positive here costs one line of manual review; a false negative puts a
# simulator artifact in front of a rater.
_ASIDE_VOCAB = (
    r"probability|persona|hedging|frustration|turn\s*number|response\s*length|"
    r"affect|persistence|trail\w*\s*off|pause[sd]?|pausing|sigh\w*|mumbl\w*|"
    r"mutter\w*|grumbl\w*|groan\w*|whisper\w*|shrug\w*|laugh\w*|silence|"
    r"inaudible|to\s+(?:him|her|them|it)self|under\s+\w+\s+breath|quietly"
)
_ARTIFACT = re.compile(
    rf"\((?=[^)]*\b(?:{_ASIDE_VOCAB})\b)[^)]*\)"
    rf"|\*(?=[^*]{{0,60}}\b(?:{_ASIDE_VOCAB})\b)[^*]{{0,60}}\*",
    re.IGNORECASE,
)
# "(muttering to self)", "(trails off)", "*sighs quietly*" — an aside whose
# first word is a bare verb of manner. "(which felt wrong)", "(x plus x)" and
# the `*` of `x * x` are not.
_STAGE_DIRECTION_SHAPE = re.compile(
    r"\(\s*[a-z]+(?:ing|s)\b[^)0-9]{0,40}\)"
    r"|\*\s*[a-z]+(?:ing|s)\b[^*0-9]{0,40}\*"
)
# The coach quotes with curly marks, so a student who used straight quotes
# ("affect" vs "effect") nests inside rather than terminating the anchor.
_QUOTED_CURLY = re.compile(r"“([^”]+)”")
_QUOTED_STRAIGHT = re.compile(r"\"([^\"]+)\"")
# Two quote marks with nothing but space between them. Nothing a student writes
# looks like this; it is always the seam between two scripted speech segments.
_ADJACENT_QUOTES = re.compile(r"[\"“”]\s*[\"“”]")


def _norm(text: str) -> str:
    """Whitespace- and case-insensitive form, for answer containment."""
    return re.sub(r"\s+", "", (text or "").lower())


def _anchors(response: str) -> list[str]:
    found = [m.group(1) for m in _QUOTED_CURLY.finditer(response or "")]
    if not found:
        found = [m.group(1) for m in _QUOTED_STRAIGHT.finditer(response or "")]
    return [fragment.strip(" .,;:!?") for fragment in found]


def _coach_words_only(response: str) -> str:
    """The response with everything it quoted from the student removed."""
    stripped = _QUOTED_CURLY.sub(" ", response or "")
    if stripped == response:
        stripped = _QUOTED_STRAIGHT.sub(" ", response or "")
    return stripped


# ------------------------------------------------------------------
# Checks. Each returns (passed, headline, [detail lines]).
# ------------------------------------------------------------------

def check_schema(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    problems = []
    for i, pair in enumerate(pairs, 1):
        missing = [k for k in REQUIRED_KEYS if k not in pair]
        if missing:
            problems.append(f"pair {i}: missing {missing}")
            continue
        hidden_missing = [k for k in REQUIRED_HIDDEN_KEYS if k not in pair["_hidden"]]
        if hidden_missing:
            problems.append(f"pair {i}: _hidden missing {hidden_missing}")
        empty = [
            k for k in ("problem_text", "student_input",
                        "response_scaffolded", "response_answer_giving")
            if not str(pair.get(k, "")).strip()
        ]
        if empty:
            problems.append(f"pair {i}: empty {empty}")
        if "shown_as_A" not in pair["display_order"]:
            problems.append(f"pair {i}: display_order missing shown_as_A")
    return (
        not problems,
        f"all required keys present and non-empty on {len(pairs)} pairs",
        problems,
    )


def check_display_order(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    problems = [
        f"pair {i}: display_order.shown_as_A = {pair['display_order']['shown_as_A']!r}"
        for i, pair in enumerate(pairs, 1)
        if pair["display_order"]["shown_as_A"] not in (SCAFFOLDED, ANSWER_GIVING)
    ]
    counts = {
        c: sum(1 for p in pairs if p["display_order"]["shown_as_A"] == c)
        for c in (SCAFFOLDED, ANSWER_GIVING)
    }
    return (
        not problems,
        f"only scaffolded/answer_giving "
        f"({counts[SCAFFOLDED]} / {counts[ANSWER_GIVING]} shown as A)",
        problems,
    )


def check_quote_anchors(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    problems, shortest = [], None
    for i, pair in enumerate(pairs, 1):
        found = _anchors(pair["response_scaffolded"])
        if not found:
            problems.append(
                f"pair {i} ({pair['problem_id']}): scaffolded response quotes "
                "nothing from the student"
            )
            continue
        for anchor in found:
            words = anchor.split()
            if shortest is None or len(words) < len(shortest.split()):
                shortest = anchor
            if len(words) < MIN_ANCHOR_WORDS:
                problems.append(
                    f"pair {i} ({pair['problem_id']}): anchor {anchor!r} is "
                    f"{len(words)} word(s)"
                )
            if len(words) == 1 and words[0].lower() in CONNECTIVES:
                problems.append(
                    f"pair {i} ({pair['problem_id']}): anchor is the bare "
                    f"connective {anchor!r}"
                )
            spoken = " ".join(pair["student_input"].split()).lower()
            if " ".join(anchor.split()).lower() not in spoken:
                problems.append(
                    f"pair {i} ({pair['problem_id']}): anchor {anchor!r} is not "
                    "a literal span of the student's words"
                )
    return (
        not problems,
        f"every scaffolded response quotes >= {MIN_ANCHOR_WORDS} of the "
        f"student's own words (shortest: {shortest!r})",
        problems,
    )


def check_answer_leakage(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    """The scaffolded condition must never state the answer.

    Containment is measured over the coach's own words. A student who already
    wrote the keyed answer (or, for a word-choice item, both candidates) puts it
    inside the quoted anchor, and echoing a student back is not giving them the
    answer. Those cases are still listed, because a rater sees them and a
    reviewer should decide about them by eye rather than by grep.
    """
    problems, notes, leaked, stated = [], [], 0, 0
    for i, pair in enumerate(pairs, 1):
        answer = _norm(_answer_of(pair))
        if not answer:
            problems.append(f"pair {i}: no answer recorded to test against")
            continue
        scaffolded = pair["response_scaffolded"]
        if answer in _norm(_coach_words_only(scaffolded)):
            leaked += 1
            problems.append(
                f"pair {i} ({pair['problem_id']}): scaffolded response STATES "
                "the literal answer outside the student's quoted words"
            )
        elif answer in _norm(scaffolded):
            notes.append(
                f"pair {i} ({pair['problem_id']}): the answer string appears "
                "only inside the quoted student words (the student wrote it) — "
                "not leakage, but check it reads that way"
            )
        if answer in _norm(pair["response_answer_giving"]):
            stated += 1
        else:
            problems.append(
                f"pair {i} ({pair['problem_id']}): answer-giving response does "
                "NOT contain the literal answer"
            )
    return (
        not problems,
        f"answer withheld in {len(pairs) - leaked}/{len(pairs)} scaffolded, "
        f"stated in {stated}/{len(pairs)} answer-giving",
        problems + notes,
    )


def check_artifacts(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    problems = []
    for i, pair in enumerate(pairs, 1):
        student = pair["student_input"]
        if student.lstrip().startswith(('"', "“")):
            problems.append(
                f"pair {i} ({pair['problem_id']}): student_input opens with a "
                "bare quote mark"
            )
        if student.count('"') % 2 or student.count("“") != student.count("”"):
            problems.append(
                f"pair {i} ({pair['problem_id']}): student_input has an "
                "unbalanced quote mark"
            )
        # `... somewhere else?" "Is that right?` — the simulator scripted the
        # turn as two quoted speech segments butted together. The count is even,
        # so the balance test above passes it, but the marks are the simulator's
        # and a rater would see them.
        if _ADJACENT_QUOTES.search(student):
            problems.append(
                f"pair {i} ({pair['problem_id']}): student_input has adjacent "
                "quote marks (simulator speech-scripting residue)"
            )
        for field in ("student_input", "response_scaffolded",
                      "response_answer_giving", "problem_text"):
            value = str(pair.get(field, ""))
            hit = _ARTIFACT.search(value) or _STAGE_DIRECTION_SHAPE.search(value)
            if hit:
                problems.append(
                    f"pair {i} ({pair['problem_id']}): {field} contains "
                    f"simulator artifact {hit.group(0)!r}"
                )
        if re.match(r"^\s*(student|user)\s*:", student, re.IGNORECASE):
            problems.append(
                f"pair {i} ({pair['problem_id']}): student_input keeps a "
                "speaker prefix"
            )
    return (
        not problems,
        "no bare leading quotes, unbalanced quotes, speaker prefixes, or stage "
        "directions in any field",
        problems,
    )


def check_hidden_matches_display(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    problems = [
        f"pair {i} ({pair['problem_id']}): _hidden.shown_as_A "
        f"{pair['_hidden']['shown_as_A']!r} != display_order "
        f"{pair['display_order']['shown_as_A']!r}"
        for i, pair in enumerate(pairs, 1)
        if pair["_hidden"]["shown_as_A"] != pair["display_order"]["shown_as_A"]
    ]
    return not problems, "unblinding key agrees with display_order on A", problems


def check_hidden_complement(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    problems = []
    for i, pair in enumerate(pairs, 1):
        a, b = pair["_hidden"]["shown_as_A"], pair["_hidden"]["shown_as_B"]
        expected = ANSWER_GIVING if a == SCAFFOLDED else SCAFFOLDED
        if b != expected:
            problems.append(
                f"pair {i} ({pair['problem_id']}): shown_as_B {b!r}, expected "
                f"{expected!r}"
            )
    return not problems, "shown_as_B is always the complement of shown_as_A", problems


def check_response_length(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    """Report-only. Measures the residual gap instead of asserting it away."""
    scaffolded = [len(p["response_scaffolded"].split()) for p in pairs]
    answering = [len(p["response_answer_giving"].split()) for p in pairs]
    m_s, m_a = statistics.mean(scaffolded), statistics.mean(answering)
    details = [
        f"scaffolded    mean {m_s:5.1f} words  "
        f"median {statistics.median(scaffolded):5.1f}  "
        f"range {min(scaffolded)}-{max(scaffolded)}",
        f"answer_giving mean {m_a:5.1f} words  "
        f"median {statistics.median(answering):5.1f}  "
        f"range {min(answering)}-{max(answering)}",
        f"answer_giving / scaffolded = {m_a / m_s:.2f}x  "
        f"(difference {m_a - m_s:.1f} words)",
        "Report this ratio in the write-up. The conditions are NOT "
        "length-matched; the gap is measured, not eliminated.",
    ]
    return True, f"measured length ratio {m_a / m_s:.2f}x (report-only)", details


def check_stimulus_strength(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    problems = []
    for i, pair in enumerate(pairs, 1):
        hidden = pair["_hidden"]
        if hidden["weak_stimulus"]:
            problems.append(
                f"pair {i} ({pair['problem_id']}): weak stimulus — the "
                "situation model abstained"
            )
        if not hidden["scaffolded_meta"].get("intervened"):
            problems.append(
                f"pair {i} ({pair['problem_id']}): intervened is falsy "
                f"(action {hidden['scaffolded_meta'].get('policy_action')!r})"
            )
    tries = [p["_hidden"]["attempt_tries"] for p in pairs]
    return (
        not problems,
        f"{len(pairs)}/{len(pairs)} intervened, 0 weak stimuli, "
        f"attempt tries {tries} (mean {statistics.mean(tries):.1f})",
        problems,
    )


def check_question_marks(pairs: list[dict]) -> tuple[bool, str, list[str]]:
    """No response may carry more than one "?".

    A coach response asks exactly one question. A second question mark means the
    student's own question survived into the quoted anchor and is being read
    back at them, which is the defect this check exists to catch — the agent's
    grounding guard tests for it but is not applied on the pipeline's path.
    """
    problems, worst = [], 0
    for i, pair in enumerate(pairs, 1):
        for field in ("response_scaffolded", "response_answer_giving"):
            count = str(pair.get(field, "")).count("?")
            worst = max(worst, count)
            if count > 1:
                problems.append(
                    f"pair {i} ({pair['problem_id']}): {field} has {count} "
                    f"question marks — {str(pair.get(field, ''))!r}"
                )
    return (
        not problems,
        f"no response carries more than one '?' (max seen: {worst})",
        problems,
    )


_ANSWER_KEY: dict[str, str] = {}


def _answer_of(pair: dict) -> str:
    """The keyed answer for a pair, read back from the question bank."""
    if not _ANSWER_KEY:
        from eval.rq2_pipeline import load_question_bank
        _ANSWER_KEY.update(
            {q["id"]: str(q.get("answer", "")) for q in load_question_bank()}
        )
    return _ANSWER_KEY.get(pair["problem_id"], "")


CHECKS = [
    ("a", "schema", check_schema),
    ("b", "display_order values", check_display_order),
    ("c", "quote-anchor sanity", check_quote_anchors),
    ("d", "answer leakage", check_answer_leakage),
    ("e", "stray quotes / stage directions", check_artifacts),
    ("f", "hidden shown_as_A matches display_order", check_hidden_matches_display),
    ("g", "hidden shown_as_B complement", check_hidden_complement),
    ("h", "response length (report-only)", check_response_length),
    ("i", "stimulus strength", check_stimulus_strength),
    ("j", "one question mark per response", check_question_marks),
]


def preview(pairs: list[dict]) -> str:
    lines = []
    for i, pair in enumerate(pairs, 1):
        hidden = pair["_hidden"]
        meta = hidden["scaffolded_meta"]
        lines += [
            "=" * 74,
            f"{i}. {pair['problem_id']}  —  {hidden['subject']} / "
            f"{hidden['topic']}  ({hidden['task_type']}, persona "
            f"{hidden['persona']}, shown as A: {hidden['shown_as_A']})",
            "=" * 74,
            "",
            f"PROBLEM        {pair['problem_text']}",
            "",
            f"STUDENT        {pair['student_input']}",
            "",
            f"SCAFFOLDED     [{meta['policy_action']} · "
            f"\"{meta['situation_label']}\" · conf {meta['aria_confidence']} · "
            f"{len(pair['response_scaffolded'].split())} words]",
            f"               {pair['response_scaffolded']}",
            "",
            f"ANSWER-GIVING  [{len(pair['response_answer_giving'].split())} words]",
            f"               {pair['response_answer_giving']}",
            "",
        ]
    return "\n".join(lines)


def load_pairs(path: str) -> tuple[list[dict], dict]:
    """Read a pairs file in either shape it is written in.

    The pipeline writes a bare list. The frozen study set wraps that same list
    in a header block recording when it was frozen and that it, not the live
    pipeline output, is canonical. Both validate identically; the header is
    returned separately so it can be echoed above the results.
    """
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    if isinstance(loaded, dict):
        pairs = loaded.get("pairs")
        if not isinstance(pairs, list):
            raise ValueError(
                f"{path}: object-shaped pairs file has no 'pairs' list"
            )
        return pairs, {k: v for k, v in loaded.items() if k != "pairs"}
    if not isinstance(loaded, list):
        raise ValueError(f"{path}: expected a list of pairs or a header object")
    return loaded, {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", "--pairs", dest="pairs", default=PAIRS_PATH,
                    help="Pairs file to validate. Defaults to the live "
                         "pipeline output; pass the frozen study set to "
                         "validate the file the raters will actually see.")
    ap.add_argument("--quiet", action="store_true",
                    help="Skip the 10-pair preview.")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    pairs, header = load_pairs(args.pairs)

    print(f"RQ2 stimulus validation — {args.pairs} ({len(pairs)} pairs)")
    if header:
        for key in ("frozen", "frozen_on", "canonical_stimulus_set", "seed"):
            if key in header:
                print(f"  {key}: {header[key]}")
        if header.get("note"):
            print(f"  note: {header['note']}")
    print("-" * 74)
    failures = 0
    for key, name, fn in CHECKS:
        passed, headline, details = fn(pairs)
        failures += 0 if passed else 1
        print(f"[{'PASS' if passed else 'FAIL'}] {key}. {name}: {headline}")
        for line in details:
            print(f"         {line}")
    print("-" * 74)
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} checks passed"
          f"{'' if not failures else f' — {failures} FAILED'}")

    if not args.quiet:
        print("\n" + preview(pairs))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
