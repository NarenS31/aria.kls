#!/usr/bin/env python3.11
"""Build the blind human-annotation study for the inter-rater kappa analysis.

    python3.11 eval/build_annotation_study.py

Samples 150 items from the synthetic think-aloud test split, stratified evenly
across the seven cognitive-state categories, strips every label, and writes two
identical blind sheets plus the rubric raters work from.

    data/annotation/rater1_blind.csv          150 items, no labels
    data/annotation/rater2_blind.csv          byte-identical to rater1
    data/annotation/rubric.md                 the one-page coding guide
    data/annotation/annotation_private_key.json   labels, for scoring afterwards

WHY THE ITEM IDS ARE OPAQUE
The corpus id is "CONFUSED|Python Programming|hard|hyperactive|functions|0.29" —
it opens with the answer. Copying it into a blind sheet would hand raters the
label in the first column. Sheets therefore carry ITEM_001..ITEM_150 in shuffled
order, and the private key maps them back.

A LEXICAL CONFOUND, MEASURED AND LEFT IN PLACE
46% of STUCK items and 34% of CONFUSED items contain their own category word
("why do I always get stuck on this stuff"). No other category exceeds 6%. A
rater who keys on the literal word would score above chance on those two
categories without reading the reasoning, which would inflate kappa for reasons
that have nothing to do with the construct.

The words are not stripped. They are how students actually talk, and blanking
them would both corrupt the stimulus and signal to the rater that something was
removed. Instead the confound is handled where it can be handled honestly: the
rubric tells raters in as many words that the literal token is not evidence, and
the private key flags every affected item so the kappa analysis can report
agreement with and without them. Note the flag is not itself a leak — the word
present is often a DIFFERENT category's ("don't wanna get stuck" inside a
RUSHING item), which is why the flag records which word, not just that one
occurred.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from metacognition.generate import COGNITIVE_STATES, STATE_MARKERS  # noqa: E402

CORPUS_PATH = os.path.join(HERE, "data", "synthetic_thinkaloud", "test.jsonl")
OUT_DIR = os.path.join(HERE, "data", "annotation")
RATER_PATHS = (os.path.join(OUT_DIR, "rater1_blind.csv"),
               os.path.join(OUT_DIR, "rater2_blind.csv"))
RUBRIC_PATH = os.path.join(OUT_DIR, "rubric.md")
KEY_PATH = os.path.join(OUT_DIR, "annotation_private_key.json")

N_ITEMS = 150
SEED = 20260823
CONFIDENCE_SCALE = "1-5"

CSV_COLUMNS = ["item_id", "think_aloud", "predicted_category", "confidence"]

# Inflections a rater could pattern-match on, per category.
LABEL_WORD_FORMS = {
    "PLANNING": ["planning", "plan", "plans", "planned"],
    "FLOW": ["flow", "flowing"],
    "CONFUSED": ["confused", "confusing", "confusion"],
    "RUSHING": ["rushing", "rush", "rushed"],
    "FRUSTRATED": ["frustrated", "frustrating", "frustration"],
    "STUCK": ["stuck"],
    "INSIGHT": ["insight"],
}

# Definitions, taken from the generation prompts that produced the corpus so the
# rubric describes what was actually generated rather than a later gloss of it.
#   `markers`     verbatim from generate.py STATE_MARKERS
#   `definition`  the state as build_prompt / build_hard_negative_prompt frame it
#   `not_this`    the confusion this category is most often lost to
CATEGORY_GUIDE = {
    "PLANNING": {
        "definition": "Deliberate setup before starting — laying out a plan, "
                      "listing what they need, or choosing an approach — rather "
                      "than smooth continuous progress.",
        "markers": STATE_MARKERS["PLANNING"],
        "not_this": "FLOW. Both move forward confidently. PLANNING is arranging "
                    "the approach before working; FLOW is already working.",
    },
    "FLOW": {
        "definition": "Smooth continuous progress. The student is engaged and "
                      "moving through the problem without stalling, "
                      "second-guessing, or stopping to reorganise.",
        "markers": STATE_MARKERS["FLOW"],
        "not_this": "PLANNING and INSIGHT. The corpus deliberately contains "
                    "think-alouds written to LOOK like FLOW that are really one "
                    "of those two, so an engaged surface is not sufficient.",
    },
    "CONFUSED": {
        "definition": "Actively does not understand something and says so — "
                      "circling, second-guessing, contradicting themselves. "
                      "Still engaged with the content, but not tracking it.",
        "markers": STATE_MARKERS["CONFUSED"],
        "not_this": "STUCK. CONFUSED is muddled but still generating ideas; "
                    "STUCK has run out of them.",
    },
    "RUSHING": {
        "definition": "Moving too fast for the work — skipping steps, guessing "
                      "past uncertainty, settling for an answer without "
                      "checking it.",
        "markers": STATE_MARKERS["RUSHING"],
        "not_this": "FLOW. Both are fast. FLOW's steps are there and correct; "
                    "RUSHING's are skipped or waved through.",
    },
    "FRUSTRATED": {
        "definition": "Emotional reaction to the task itself — irritation, "
                      "resentment, wanting to abandon it. The feeling is the "
                      "dominant content, not the reasoning.",
        "markers": STATE_MARKERS["FRUSTRATED"],
        "not_this": "STUCK. STUCK is an absence of next steps; FRUSTRATED is a "
                    "reaction. A student can be either without the other.",
    },
    "STUCK": {
        "definition": "Has no next step and knows it. Progress has stopped, and "
                      "the student is not producing new candidate moves.",
        "markers": STATE_MARKERS["STUCK"],
        "not_this": "CONFUSED and FRUSTRATED. Ask what the student would do "
                    "next: CONFUSED has an idea but it is muddled, FRUSTRATED "
                    "may have one but resents it, STUCK has none.",
    },
    "INSIGHT": {
        "definition": "A sudden realization after difficulty — a click moment "
                      "where it finally makes sense — rather than steady "
                      "progress that was never blocked.",
        "markers": STATE_MARKERS["INSIGHT"],
        "not_this": "FLOW. INSIGHT requires the before-and-after: something was "
                    "not understood, and then it was. Smooth confidence "
                    "throughout is FLOW.",
    },
}

TEXT_PART_LABELS = [
    ("pre_attempt", "BEFORE STARTING"),
    ("during_attempt", "WHILE SOLVING"),
    ("post_attempt", "AFTER FINISHING"),
]


# ------------------------------------------------------------------
# Corpus
# ------------------------------------------------------------------

def load_corpus(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found.")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def think_aloud_block(item: dict) -> str:
    """The three think-aloud parts as one readable block for a spreadsheet cell."""
    parts = []
    for key, label in TEXT_PART_LABELS:
        text = str(item.get("think_aloud", {}).get(key, "")).strip()
        if text:
            parts.append(f"{label}: {text}")
    return "\n\n".join(parts)


def label_words_present(text: str) -> list[str]:
    """Which category words appear literally in the text, if any."""
    lowered = (text or "").lower()
    found = []
    for state, forms in LABEL_WORD_FORMS.items():
        if any(re.search(rf"\b{form}\b", lowered) for form in forms):
            found.append(state)
    return found


# ------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------

def stratified_sample(corpus: list[dict], n: int, rng: random.Random
                      ) -> tuple[list[dict], dict[str, int]]:
    """Evenly spread across the categories, remainder assigned at random.

    150 does not divide by 7, so three categories get 22 items and four get 21.
    Which three is drawn from the seeded RNG rather than taken in list order, so
    the extra item never lands on the same categories run after run by accident
    of how COGNITIVE_STATES happens to be written.
    """
    by_state: dict[str, list[dict]] = defaultdict(list)
    for item in corpus:
        state = item.get("cognitive_state")
        if state in COGNITIVE_STATES:
            by_state[state].append(item)

    missing = [s for s in COGNITIVE_STATES if not by_state.get(s)]
    if missing:
        raise SystemExit(f"corpus has no items for {missing}")

    base, remainder = divmod(n, len(COGNITIVE_STATES))
    quota = {state: base for state in COGNITIVE_STATES}
    for state in rng.sample(list(COGNITIVE_STATES), remainder):
        quota[state] += 1

    sampled: list[dict] = []
    for state in COGNITIVE_STATES:
        pool = sorted(by_state[state], key=lambda r: r["id"])
        if len(pool) < quota[state]:
            raise SystemExit(
                f"{state}: need {quota[state]} items, corpus has {len(pool)}")
        sampled.extend(rng.sample(pool, quota[state]))

    # Shuffle so the sheet does not run in category blocks.
    rng.shuffle(sampled)
    return sampled, quota


# Definitional signatures, used to choose which items illustrate a category.
#
# These are for EXAMPLE SELECTION ONLY — they never label a study item and are
# not a classifier. Drawing rubric examples at random does not work: the corpus
# is LLM-generated and many items are weak exemplars of the label they carry, so
# a random "PLANNING" item can read as CONFUSED. An example that contradicts its
# own heading teaches raters the wrong mapping, and both raters learning the same
# wrong mapping is worse for the study than no examples at all.
#
# `for` phrases are what the category's own definition says it looks like;
# `against` phrases belong to the category it is most often confused with. An
# item scores well by having the first and lacking the second, which selects for
# items that sit clearly inside their category rather than on a boundary.
EXEMPLAR_SIGNATURES = {
    "PLANNING": {
        "for": [r"first i(?:'ll| need| have| should| gotta)", r"let me think",
                r"my plan", r"i'?m gonna start", r"step one", r"i need to figure out",
                r"start by", r"the plan is"],
        "against": [r"\bwait,? no\b", r"i don'?t get", r"\bugh\b", r"\bhuh\b",
                    r"start over", r"that'?s not right", r"i don'?t know"],
    },
    "FLOW": {
        "for": [r"\bso\b.*\bthen\b", r"\bnext\b", r"that gives", r"which means",
                r"got it", r"easy", r"makes sense"],
        "against": [r"\bwait\b", r"\bugh\b", r"i don'?t know", r"i don'?t get",
                    r"\bhuh\b", r"no,? that'?s not", r"start over", r"\[pause",
                    r"probably", r"whatever", r"confus"],
    },
    "CONFUSED": {
        "for": [r"i don'?t get", r"\bhuh\b", r"wait,? what", r"does that even",
                r"i'?m not sure (?:if|what|how)", r"or is it", r"but then again"],
        "against": [r"i don'?t know what to do", r"\[pause", r"forget it",
                    r"give up", r"i give", r"stares blank"],
    },
    "RUSHING": {
        "for": [r"probably", r"whatever", r"close enough", r"just guess",
                r"good enough", r"i'?ll just say", r"skip", r"doesn'?t matter"],
        "against": [r"let me (?:double )?check", r"wait,? let me", r"i don'?t know",
                    r"\[pause", r"\bugh\b"],
    },
    "FRUSTRATED": {
        "for": [r"\bugh\b", r"\bstupid\b", r"i hate", r"forget it", r"so annoying",
                r"why do i", r"this is dumb", r"come on"],
        "against": [r"let me think", r"first i", r"\boh!\b", r"that'?s it",
                    r"makes sense now"],
    },
    "STUCK": {
        "for": [r"i don'?t know", r"no idea", r"\[pause", r"stares blank",
                r"i'?m lost", r"can'?t remember", r"blanking"],
        "against": [r"first i", r"let me think", r"\boh!\b", r"that'?s it",
                    r"got it", r"makes sense now"],
    },
    "INSIGHT": {
        "for": [r"\boh+!?\b", r"ohh+", r"that'?s it", r"makes sense now",
                r"i get it now", r"\bclick", r"now i see", r"of course"],
        "against": [r"i don'?t know", r"\[pause", r"forget it", r"still don'?t",
                    r"no idea"],
    },
}


def exemplar_score(text: str, state: str) -> int:
    """How cleanly an item sits inside its own category. Higher is clearer."""
    lowered = (text or "").lower()
    signature = EXEMPLAR_SIGNATURES[state]
    hits = sum(1 for pattern in signature["for"]
               if re.search(pattern, lowered))
    misses = sum(1 for pattern in signature["against"]
                 if re.search(pattern, lowered))
    return hits - misses


def pick_rubric_examples(corpus: list[dict], sampled_ids: set[str],
                         rng: random.Random, per_category: int = 2
                         ) -> dict[str, list[dict]]:
    """Clearest available examples per category, from items NOT in the study.

    Ranked by `exemplar_score`, with items containing their own category word
    dropped first: an example that says "I'm so stuck" teaches raters to match
    the token, which is the shortcut the rubric is trying to talk them out of.
    Ties break on the corpus id so the choice is reproducible.
    """
    examples: dict[str, list[dict]] = {}
    for state in COGNITIVE_STATES:
        pool = [item for item in corpus
                if item.get("cognitive_state") == state
                and item["id"] not in sampled_ids]
        clean = [item for item in pool
                 if state not in label_words_present(think_aloud_block(item))]
        chosen_pool = clean if len(clean) >= per_category else pool
        # Score the span that will actually be printed, not the whole item. A
        # think-aloud can open with a clean plan and dissolve into confusion
        # three sentences later; what matters is whether the quote a rater reads
        # illustrates the category, so the excerpt is what gets ranked.
        candidates = []
        for item in chosen_pool:
            excerpt = excerpt_for(item, state)
            candidates.append((
                -exemplar_score(excerpt, state),
                -exemplar_score(think_aloud_block(item), state),
                item["id"], item, excerpt,
            ))
        candidates.sort(key=lambda c: c[:3])
        examples[state] = [{
            "source_id": item["id"],
            "excerpt": excerpt,
            "excerpt_score": -neg_excerpt,
            "item_score": -neg_item,
            "uses_own_label_word": state in label_words_present(
                think_aloud_block(item)),
        } for neg_excerpt, neg_item, _id, item, excerpt
            in candidates[:per_category]]
    return examples


def excerpt_for(item: dict, state: str = "", max_chars: int = 300) -> str:
    """A short quotable span that actually contains the category's evidence.

    Taking the first N characters does not work for every category. An INSIGHT
    think-aloud earns its label at the click moment, which is usually near the
    end, so a head-truncated excerpt shows only the confusion that preceded it
    and reads like CONFUSED. The window is therefore centred on the strongest
    definitional match, and cut at sentence boundaries so it does not end
    mid-word.
    """
    think = item.get("think_aloud", {})
    text = " ".join(part for part in (
        str(think.get("during_attempt") or "").strip(),
        str(think.get("post_attempt") or "").strip(),
    ) if part) or str(think.get("pre_attempt") or "").strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    # Find the evidence this item was chosen for, and centre the window on it.
    anchor = 0
    if state in EXEMPLAR_SIGNATURES:
        for pattern in EXEMPLAR_SIGNATURES[state]["for"]:
            match = re.search(pattern, text.lower())
            if match:
                anchor = match.start()
                break

    start = max(0, anchor - max_chars // 2)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)

    window = text[start:end]
    # Prefer to open at a sentence start and close at a sentence end.
    if start > 0:
        boundary = re.search(r"(?<=[.!?…])\s+", window)
        if boundary and boundary.end() < len(window) // 3:
            window = window[boundary.end():]
    tail = re.search(r"[.!?…](?!.*[.!?…])", window, re.DOTALL)
    if tail and tail.end() > len(window) // 2:
        window = window[:tail.end()]
    elif " " in window:
        window = window[:window.rfind(" ")]

    return ("…" if start > 0 else "") + window.strip() + (
        "…" if end < len(text) and not window.rstrip().endswith(("…",)) else "")


# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------

def write_blind_sheets(rows: list[dict]) -> None:
    for path in RATER_PATHS:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "item_id": row["item_id"],
                    "think_aloud": row["think_aloud"],
                    "predicted_category": "",
                    "confidence": "",
                })


def write_key(rows: list[dict], quota: dict[str, int], examples: dict) -> dict:
    key = {
        "_README": (
            "PRIVATE KEY — do not give this to a rater and do not put it in the "
            "same folder you share. Maps the blind item ids back to the corpus "
            "labels for scoring after both raters have returned their sheets."
        ),
        "generated_by": "eval/build_annotation_study.py",
        "seed": SEED,
        "corpus": os.path.relpath(CORPUS_PATH, REPO_ROOT),
        "n_items": len(rows),
        "categories": list(COGNITIVE_STATES),
        "per_category_quota": quota,
        "confidence_scale": CONFIDENCE_SCALE,
        "lexical_confound": {
            "measured_on": "the full 350-item test split",
            "own_label_word_rate": {"STUCK": 0.46, "CONFUSED": 0.34,
                                    "FLOW": 0.06, "FRUSTRATED": 0.06,
                                    "PLANNING": 0.02, "RUSHING": 0.0,
                                    "INSIGHT": 0.0},
            "handling": "Text left intact; rubric instructs raters that the "
                        "literal token is not evidence. Per-item flags below "
                        "let the kappa analysis report agreement with and "
                        "without the affected items.",
        },
        "rubric_examples_are_disjoint_from_study_items": True,
        "rubric_example_source_ids": {
            state: [e["source_id"] for e in items]
            for state, items in examples.items()
        },
        "items": {
            row["item_id"]: {
                "true_category": row["true_category"],
                "source_id": row["source_id"],
                "subject": row["subject"],
                "topic": row["topic"],
                "difficulty": row["difficulty"],
                "adhd_type": row["adhd_type"],
                "label_words_present": row["label_words_present"],
                "contains_own_label_word": row["contains_own_label_word"],
            }
            for row in rows
        },
    }
    with open(KEY_PATH, "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return key


def render_rubric(quota: dict[str, int], examples: dict) -> str:
    total = sum(quota.values())
    lines = [
        "# Think-aloud coding rubric",
        "",
        f"**{total} items · one category per item · "
        f"confidence {CONFIDENCE_SCALE}**",
        "",
        "You will read a student's think-aloud — what they said before, during, "
        "and after attempting a problem — and assign the ONE category that best "
        "describes their cognitive state. Then rate how confident you are, from "
        "1 (guessing) to 5 (certain).",
        "",
        "Work down the sheet in order. Do not go back and revise earlier items "
        "after you have developed a feel for the set: the point of two "
        "independent raters is that neither has been tuned by the other or by "
        "hindsight.",
        "",
        "## Before you start: three rules",
        "",
        "1. **Code what is on the page, not the student.** You are describing "
        "the reasoning in this text, not diagnosing a person.",
        "2. **The literal word is not the category.** A student who writes "
        "\"I don't wanna get stuck\" is not thereby STUCK, and one who says "
        "\"this is confusing\" is not thereby CONFUSED — those words show up "
        "inside several different categories. Judge the reasoning; the "
        "vocabulary is a coincidence of how teenagers talk.",
        "3. **Every item gets exactly one category.** If two fit, pick the one "
        "that governs the passage as a whole, and drop your confidence to 2 or "
        "1 to record that it was close.",
        "",
        "## The seven categories",
        "",
    ]

    for state in COGNITIVE_STATES:
        guide = CATEGORY_GUIDE[state]
        lines += [
            f"### {state}",
            "",
            guide["definition"],
            "",
            f"*Typical markers:* {guide['markers']}",
            "",
            f"*Most often confused with:* {guide['not_this']}",
            "",
        ]
        for example in examples.get(state, []):
            lines.append(f"> {example['excerpt']}")
            lines.append("")

    lines += [
        "## When two categories both fit",
        "",
        "Most disagreement in a set like this comes from four pairs. Use these "
        "tie-breakers rather than inventing your own, so both raters break ties "
        "the same way.",
        "",
        "| If you are torn between | Ask |",
        "|---|---|",
        "| STUCK vs CONFUSED | Do they still have a candidate next step? "
        "An idea, however muddled, is CONFUSED. No idea at all is STUCK. |",
        "| STUCK vs FRUSTRATED | Is the dominant content an absence of steps, "
        "or a reaction to the task? Absence is STUCK, reaction is FRUSTRATED. |",
        "| FLOW vs PLANNING | Are they working, or arranging how to work? "
        "Arranging is PLANNING even when it sounds confident. |",
        "| FLOW vs INSIGHT | Was anything unclear earlier in the passage? "
        "INSIGHT needs a before-and-after. Uninterrupted confidence is FLOW. |",
        "| FLOW vs RUSHING | Are the steps present and sound, or skipped? "
        "Fast and correct is FLOW; fast and waved through is RUSHING. |",
        "",
        "## Filling in the sheet",
        "",
        "`predicted_category` — one of: "
        + ", ".join(f"`{s}`" for s in COGNITIVE_STATES) + ".",
        "",
        "`confidence` — 1 guessing · 2 leaning · 3 reasonably sure · "
        "4 confident · 5 certain. Use the low end freely; a well-placed 1 is "
        "more useful to the analysis than a polite 3.",
        "",
        "Leave nothing blank. If an item seems uncodable, choose the closest "
        "category, mark confidence 1, and carry on.",
        "",
        "---",
        "",
        f"*Category definitions and markers are reproduced from the generation "
        f"prompts in `eval/metacognition/generate.py` that produced these items. "
        f"Every example above is drawn from items that are NOT in your "
        f"{total}-item sheet.*",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------

def verify(rows: list[dict], quota: dict[str, int], examples: dict) -> list[str]:
    problems = []

    counts = Counter(row["true_category"] for row in rows)
    for state in COGNITIVE_STATES:
        if counts[state] != quota[state]:
            problems.append(
                f"{state}: {counts[state]} items, quota was {quota[state]}")
    if len(rows) != N_ITEMS:
        problems.append(f"{len(rows)} items, expected {N_ITEMS}")
    if len({row["item_id"] for row in rows}) != len(rows):
        problems.append("duplicate item_id")
    if len({row["source_id"] for row in rows}) != len(rows):
        problems.append("the same corpus item was sampled twice")

    # The blind sheet must not carry the answer anywhere.
    for row in rows:
        for state in COGNITIVE_STATES:
            if state.lower() in row["item_id"].lower():
                problems.append(f"{row['item_id']}: item id names a category")
        if row["source_id"] in row["think_aloud"]:
            problems.append(f"{row['item_id']}: source id leaked into the text")
        if not row["think_aloud"].strip():
            problems.append(f"{row['item_id']}: empty think-aloud")

    # Rubric examples must be disjoint from the study items.
    sampled_sources = {row["source_id"] for row in rows}
    for state, items in examples.items():
        for example in items:
            if example["source_id"] in sampled_sources:
                problems.append(
                    f"rubric example for {state} is also a study item "
                    f"({example['source_id']})")
        if not items:
            problems.append(f"no rubric example for {state}")

    # The two sheets must be identical.
    contents = []
    for path in RATER_PATHS:
        if not os.path.exists(path):
            problems.append(f"{path} was not written")
            continue
        with open(path, "rb") as fh:
            contents.append(fh.read())
    if len(contents) == 2 and contents[0] != contents[1]:
        problems.append("rater1 and rater2 sheets differ")

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=N_ITEMS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    rng = random.Random(args.seed)
    corpus = load_corpus(CORPUS_PATH)
    print(f"corpus            {os.path.relpath(CORPUS_PATH, REPO_ROOT)} "
          f"({len(corpus)} items)")

    sampled, quota = stratified_sample(corpus, args.n, rng)
    sampled_ids = {item["id"] for item in sampled}
    examples = pick_rubric_examples(corpus, sampled_ids, rng)

    rows = []
    for i, item in enumerate(sampled, 1):
        block = think_aloud_block(item)
        present = label_words_present(block)
        state = item["cognitive_state"]
        problem = item.get("problem", {}) or {}
        profile = item.get("student_profile", {}) or {}
        rows.append({
            "item_id": f"ITEM_{i:03d}",
            "think_aloud": block,
            "true_category": state,
            "source_id": item["id"],
            "subject": problem.get("subject", ""),
            "topic": problem.get("topic", ""),
            "difficulty": problem.get("difficulty", ""),
            "adhd_type": profile.get("adhd_type", ""),
            "label_words_present": present,
            "contains_own_label_word": state in present,
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    write_blind_sheets(rows)
    with open(RUBRIC_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_rubric(quota, examples))
    write_key(rows, quota, examples)

    problems = verify(rows, quota, examples)

    print(f"seed              {args.seed}")
    print(f"sampled           {len(rows)} items  "
          f"({', '.join(f'{s} {quota[s]}' for s in COGNITIVE_STATES)})")
    flagged = sum(1 for r in rows if r["contains_own_label_word"])
    print(f"lexical confound  {flagged}/{len(rows)} items contain their own "
          f"category word (flagged in the key, text left intact)")
    print()
    for path in RATER_PATHS + (RUBRIC_PATH, KEY_PATH):
        print(f"wrote             {os.path.relpath(path, REPO_ROOT)}")
    print()
    if problems:
        print(f"{len(problems)} CHECK(S) FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("all checks passed: even strata, opaque ids, no label in any sheet, "
          "rubric examples disjoint from study items, both sheets identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
