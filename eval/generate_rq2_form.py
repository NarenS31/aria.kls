#!/usr/bin/env python3.11
"""Generate the RQ2 rating form script and its hidden answer key.

    python3.11 eval/generate_rq2_form.py

Reads the frozen stimulus set and writes two files:

    eval/build_rq2_form.gs     Apps Script that builds the Google Form
    data/rq2_answer_key.json   the unblinding key, never seen by a rater

The point of generating rather than hand-writing the .gs is the blinding. Each
item shows its two responses under the neutral labels "Response A" and
"Response B", and which condition sits under which letter varies per item —
five items lead with the scaffolded response, five with the answer-giving one.
Transcribing that by hand is exactly the kind of work that produces one silently
swapped item, and a swapped item does not look wrong in the form; it looks fine
and inverts that item's result. So the letters are assigned here, from
`display_order.shown_as_A` in the frozen file, and asserted before anything is
written.

The generated .gs carries no condition names, no `_hidden` block, and no
reference to the answer key. It cannot tell you which response is which, which
is what makes it safe to open in a browser next to a rater.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data")
FROZEN_PATH = os.path.join(DATA_DIR, "rq2_pairs_FROZEN.json")
GS_PATH = os.path.join(HERE, "build_rq2_form.gs")
ANSWER_KEY_PATH = os.path.join(DATA_DIR, "rq2_answer_key.json")

GENERATED_ON = "2026-08-22"

SCAFFOLDED = "scaffolded"
ANSWER_GIVING = "answer_giving"

# ------------------------------------------------------------------
# Fixed form copy. Wording is the study's, reproduced verbatim.
# ------------------------------------------------------------------

FORM_TITLE = "Comparing Two Tutoring Response Styles (10 items, ~10 min)"

FORM_DESCRIPTION = (
    "We're comparing two different styles of AI tutoring responses to the same "
    "math, English, science, and coding problems. For each item you'll see a "
    "problem, what the student had already tried, and two possible responses "
    "(A and B) — please read both and answer a few quick questions. There are "
    "no wrong answers. Your responses are anonymous and used only for a student "
    "research project. You can stop at any time."
)

CONSENT_TITLE = (
    "By continuing, you confirm you're voluntarily participating and understand "
    "your responses are anonymous."
)
CONSENT_CHOICE = "I understand and agree to continue"

STUDENT_LABEL = "What the student had tried so far:"
RESPONSE_A_LABEL = "Response A:"
RESPONSE_B_LABEL = "Response B:"

Q_WHICH_HELPS = (
    "Which response would actually help you understand how to solve this, not "
    "just get the answer?"
)
Q_RATE_A = "Rate Response A — how helpful is this for actually learning the concept?"
Q_RATE_B = "Rate Response B — how helpful is this for actually learning the concept?"
Q_WHEN_STUCK = (
    "If you were stuck and stressed, which response would you rather receive "
    "right now?"
)
Q_FREE_TEXT = "Anything either response got wrong, confusing, or annoying?"

AB_CHOICES = ["A", "B", "About the same"]
SCALE_LOW_LABEL = "Not helpful at all"
SCALE_HIGH_LABEL = "Extremely helpful"

CLOSING_SECTION_TITLE = "A few last questions"
Q_OVERALL = (
    "Across all the items you just saw, which style did you generally prefer?"
)
OVERALL_CHOICES = ["Consistently A", "Consistently B", "Depended on the item"]
Q_GRADE = "What grade or year are you in?"
Q_MATH_RELATIONSHIP = "How would you describe your relationship with math?"
# Same five options as the earlier formative survey. That survey is not in this
# repo, so these are written from its five descriptors; if the original wording
# differs, edit here and regenerate rather than editing the .gs.
MATH_RELATIONSHIP_CHOICES = [
    "I'm fine at it, but I don't really enjoy it",
    "I struggle with specific areas",
    "I struggle with it a lot",
    "I actively avoid it",
    "I actually enjoy it",
]

# The item where the student's own writing contains the keyed answer, so a rater
# does not read the echo as one response having given the answer away.
NOTE_PROBLEM_ID = "english-revision-04"
ITEM_NOTE = (
    "Note: in this item, the student's own writing already includes the word "
    "'effect' — that's not something either response revealed to them."
)


def confirmation_message(ratio: float) -> str:
    """Closing message, with the length gap measured from the frozen set.

    Deliberately NOT the wording the spec gave, which named "Response B style"
    as the longer one. The letters are randomised per item — the answer-giving
    response is B on five items and A on the other five — so that sentence is
    false for half the set and implies a consistency the design does not have.
    The gap is described by style instead of by letter, and the measured ratio
    is derived from the frozen file so it cannot drift from the data.
    """
    percent = round((ratio - 1) * 100 / 10) * 10
    return (
        "Thank you — this is genuinely useful. One honest note: the two response "
        "styles in this survey are not perfectly matched in length. The style "
        f"that walks through the full explanation runs about {percent}% longer "
        f"on average (measured {ratio}× in this set), and it wasn't always the "
        "one labelled A or B — the labels were shuffled from item to item. Keep "
        "that in mind when you think back on your ratings."
    )


# ------------------------------------------------------------------
# Blinding
# ------------------------------------------------------------------

def assign_letters(pair: dict) -> tuple[str, str, str]:
    """Return (response under A, response under B, condition shown as A).

    The only place the mapping is decided. Everything downstream — the .gs and
    the answer key — reads this, so the form and the key cannot disagree.
    """
    shown_as_a = pair["display_order"]["shown_as_A"]
    if shown_as_a == SCAFFOLDED:
        return pair["response_scaffolded"], pair["response_answer_giving"], SCAFFOLDED
    if shown_as_a == ANSWER_GIVING:
        return pair["response_answer_giving"], pair["response_scaffolded"], ANSWER_GIVING
    raise ValueError(f"{pair['problem_id']}: unknown shown_as_A {shown_as_a!r}")


def check_blinding(pairs: list[dict], items: list[dict]) -> None:
    """Re-derive the mapping from the frozen file and compare, item by item."""
    assert len(pairs) == len(items), "item count does not match the frozen set"
    for pair, item in zip(pairs, items):
        assert item["problemId"] == pair["problem_id"]
        expected_a = pair["display_order"]["shown_as_A"]
        true_a = (
            SCAFFOLDED if item["responseA"] == pair["response_scaffolded"]
            else ANSWER_GIVING if item["responseA"] == pair["response_answer_giving"]
            else None
        )
        true_b = (
            SCAFFOLDED if item["responseB"] == pair["response_scaffolded"]
            else ANSWER_GIVING if item["responseB"] == pair["response_answer_giving"]
            else None
        )
        assert true_a is not None, f"{item['problemId']}: A text is in neither condition"
        assert true_b is not None, f"{item['problemId']}: B text is in neither condition"
        assert true_a == expected_a, (
            f"{item['problemId']}: A holds {true_a}, frozen file says {expected_a}"
        )
        assert true_b != true_a, f"{item['problemId']}: A and B hold the same condition"
        assert item["responseA"] != item["responseB"]
    # Both orders must actually occur, or the "randomised" order is a constant.
    leading = {i["problemId"]: p["display_order"]["shown_as_A"]
               for i, p in zip(items, pairs)}
    assert set(leading.values()) == {SCAFFOLDED, ANSWER_GIVING}, (
        "every item leads with the same condition — the blinding is a constant"
    )


def build_items(pairs: list[dict]) -> list[dict]:
    items = []
    for pair in pairs:
        response_a, response_b, _ = assign_letters(pair)
        item = {
            "problemId": pair["problem_id"],
            "problemText": pair["problem_text"],
            "studentInput": pair["student_input"],
            "responseA": response_a,
            "responseB": response_b,
        }
        if pair["problem_id"] == NOTE_PROBLEM_ID:
            item["note"] = ITEM_NOTE
        items.append(item)
    return items


# ------------------------------------------------------------------
# Emit the Apps Script
# ------------------------------------------------------------------

def js(value) -> str:
    """A Python value as a JavaScript literal. JSON is a subset of JS."""
    return json.dumps(value, ensure_ascii=False)


def render_gs(items: list[dict], ratio: float, frozen_on: str) -> str:
    stimuli = json.dumps(items, ensure_ascii=False, indent=2)

    return f'''/**
 * Builds the RQ2 rating form: scaffolded vs answer-giving tutoring responses.
 *
 * GENERATED FILE — do not hand-edit. Regenerate with:
 *     python3.11 eval/generate_rq2_form.py
 * Source of truth: data/rq2_pairs_FROZEN.json (frozen {frozen_on}).
 *
 * HOW TO RUN
 *   1. Go to script.google.com and start a new project.
 *   2. Replace the contents of Code.gs with this whole file.
 *   3. Run buildRq2Form(). Authorise it when Google asks — it needs permission
 *      to create a form in your Drive.
 *   4. The published (share this) and edit URLs are printed to the execution
 *      log, and also written into the function's return value.
 *
 * BLINDING
 *   Each item shows its two responses as "Response A" and "Response B". Which
 *   condition holds which letter is fixed per item in the frozen stimulus file
 *   and was applied when this file was generated. Five items lead with the
 *   scaffolded response and five with the answer-giving one, so a rater cannot
 *   learn the pattern partway through and start rating the letter.
 *
 *   Nothing in this file records which is which. The words "scaffolded" and
 *   "answer_giving" do not appear below, and the answer key that does hold the
 *   mapping (data/rq2_answer_key.json) is never referenced here or linked from
 *   the form. If you need to unblind, use that file after collection.
 *
 * ONE DEVIATION FROM THE SPEC, FOR CORRECTNESS
 *   The closing message was specified as saying "Response B style runs about
 *   30% longer on average". B is the answer-giving response on only five of the
 *   ten items, so that sentence is false for the other five and implies the
 *   letters are stable across the form. The message below states the same
 *   measured gap ({ratio}x) by style rather than by letter, and says the labels
 *   were shuffled. To ship the original wording instead, edit
 *   confirmation_message() in eval/generate_rq2_form.py and regenerate.
 *
 * FORMATTING
 *   The Forms service has no rich text for display items, so there is no way to
 *   italicise the per-item note. It is set as help text, which Google renders
 *   smaller and lighter than a title — the closest the API gets.
 */

var FORM_TITLE = {js(FORM_TITLE)};

var FORM_DESCRIPTION = {js(FORM_DESCRIPTION)};

var CONSENT_TITLE = {js(CONSENT_TITLE)};

var CONSENT_CHOICE = {js(CONSENT_CHOICE)};

var STUDENT_LABEL = {js(STUDENT_LABEL)};
var RESPONSE_A_LABEL = {js(RESPONSE_A_LABEL)};
var RESPONSE_B_LABEL = {js(RESPONSE_B_LABEL)};

var Q_WHICH_HELPS = {js(Q_WHICH_HELPS)};
var Q_RATE_A = {js(Q_RATE_A)};
var Q_RATE_B = {js(Q_RATE_B)};
var Q_WHEN_STUCK = {js(Q_WHEN_STUCK)};
var Q_FREE_TEXT = {js(Q_FREE_TEXT)};

var AB_CHOICES = {js(AB_CHOICES)};
var SCALE_LOW_LABEL = {js(SCALE_LOW_LABEL)};
var SCALE_HIGH_LABEL = {js(SCALE_HIGH_LABEL)};

var CLOSING_SECTION_TITLE = {js(CLOSING_SECTION_TITLE)};
var Q_OVERALL = {js(Q_OVERALL)};
var OVERALL_CHOICES = {js(OVERALL_CHOICES)};
var Q_GRADE = {js(Q_GRADE)};
var Q_MATH_RELATIONSHIP = {js(Q_MATH_RELATIONSHIP)};
var MATH_RELATIONSHIP_CHOICES = {js(MATH_RELATIONSHIP_CHOICES)};

var CONFIRMATION_MESSAGE = {js(confirmation_message(ratio))};

/**
 * The ten items, in the order they are shown. `responseA` / `responseB` are
 * already in display order; this file does not know, and must not know, which
 * condition either of them came from.
 */
var STIMULI = {stimuli};

/**
 * Build the form. Returns {{ publishedUrl, editUrl, itemCount }}.
 */
function buildRq2Form() {{
  var form = FormApp.create(FORM_TITLE);
  form.setDescription(FORM_DESCRIPTION);
  form.setProgressBar(true);
  form.setAllowResponseEdits(false);
  form.setShowLinkToRespondAgain(false);
  form.setConfirmationMessage(CONFIRMATION_MESSAGE);

  // Anonymity. These are the settings that differ between account types and
  // across Apps Script versions, so a missing one must not abort the build —
  // every one of them is already the default for a newly created form.
  try {{ form.setCollectEmail(false); }} catch (e) {{}}
  try {{ form.setLimitOneResponsePerUser(false); }} catch (e) {{}}
  try {{ form.setRequireLogin(false); }} catch (e) {{}}

  form.addCheckboxItem()
      .setTitle(CONSENT_TITLE)
      .setChoiceValues([CONSENT_CHOICE])
      .setRequired(true);

  for (var i = 0; i < STIMULI.length; i++) {{
    addItemSection(form, STIMULI[i]);
  }}

  addClosingSection(form);

  // Newer Apps Script requires an explicit publish step; older versions have no
  // such method and forms are reachable as soon as they exist.
  try {{ form.setPublished(true); }} catch (e) {{}}

  var result = {{
    publishedUrl: form.getPublishedUrl(),
    editUrl: form.getEditUrl(),
    itemCount: STIMULI.length
  }};
  Logger.log('Built "%s" with %s items.', FORM_TITLE, result.itemCount);
  Logger.log("SHARE THIS URL: %s", result.publishedUrl);
  Logger.log("EDIT URL:       %s", result.editUrl);
  return result;
}}

/** One page per item: the problem, the attempt, both responses, five questions. */
function addItemSection(form, item) {{
  form.addPageBreakItem().setTitle(item.problemText);

  form.addSectionHeaderItem()
      .setTitle(STUDENT_LABEL)
      .setHelpText(item.studentInput);
  form.addSectionHeaderItem()
      .setTitle(RESPONSE_A_LABEL)
      .setHelpText(item.responseA);
  form.addSectionHeaderItem()
      .setTitle(RESPONSE_B_LABEL)
      .setHelpText(item.responseB);

  // Item-specific footnote, shown before the questions so it is read first.
  // Untitled on purpose: help text alone is the smallest, lightest register the
  // Forms service offers.
  if (item.note) {{
    form.addSectionHeaderItem().setHelpText(item.note);
  }}

  form.addMultipleChoiceItem()
      .setTitle(Q_WHICH_HELPS)
      .setChoiceValues(AB_CHOICES)
      .setRequired(true);

  form.addScaleItem()
      .setTitle(Q_RATE_A)
      .setBounds(1, 5)
      .setLabels(SCALE_LOW_LABEL, SCALE_HIGH_LABEL)
      .setRequired(true);

  form.addScaleItem()
      .setTitle(Q_RATE_B)
      .setBounds(1, 5)
      .setLabels(SCALE_LOW_LABEL, SCALE_HIGH_LABEL)
      .setRequired(true);

  form.addMultipleChoiceItem()
      .setTitle(Q_WHEN_STUCK)
      .setChoiceValues(AB_CHOICES)
      .setRequired(true);

  form.addParagraphTextItem()
      .setTitle(Q_FREE_TEXT)
      .setRequired(false);
}}

/** Overall preference plus the two background questions. */
function addClosingSection(form) {{
  form.addPageBreakItem().setTitle(CLOSING_SECTION_TITLE);

  form.addMultipleChoiceItem()
      .setTitle(Q_OVERALL)
      .setChoiceValues(OVERALL_CHOICES)
      .setRequired(true);

  // Background questions are optional: a rater who skips them still gives a
  // usable set of ratings, and requiring them only costs completions.
  form.addTextItem()
      .setTitle(Q_GRADE)
      .setRequired(false);

  form.addMultipleChoiceItem()
      .setTitle(Q_MATH_RELATIONSHIP)
      .setChoiceValues(MATH_RELATIONSHIP_CHOICES)
      .setRequired(false);
}}
'''


# ------------------------------------------------------------------
# Emit the answer key
# ------------------------------------------------------------------

def render_answer_key(pairs: list[dict], frozen: dict) -> dict:
    key = {}
    for number, pair in enumerate(pairs, 1):
        _, _, shown_as_a = assign_letters(pair)
        shown_as_b = ANSWER_GIVING if shown_as_a == SCAFFOLDED else SCAFFOLDED
        key[pair["problem_id"]] = {
            "form_item_number": number,
            "A": shown_as_a,
            "B": shown_as_b,
            "scaffolded_letter": "A" if shown_as_a == SCAFFOLDED else "B",
            "answer_giving_letter": "B" if shown_as_a == SCAFFOLDED else "A",
            "subject": pair["_hidden"]["subject"],
            "topic": pair["_hidden"]["topic"],
            "task_type": pair["_hidden"]["task_type"],
        }
    counts = {
        SCAFFOLDED: sum(1 for v in key.values() if v["A"] == SCAFFOLDED),
        ANSWER_GIVING: sum(1 for v in key.values() if v["A"] == ANSWER_GIVING),
    }
    return {
        "_README": (
            "HIDDEN UNBLINDING KEY for the RQ2 rating form. This file maps each "
            "item to the condition behind its A and B labels. It is not needed "
            "to run the form and must never be linked from it, pasted into it, "
            "or sent to a rater — it is read only after responses are collected, "
            "by eval/rq2_analysis.py. `form_item_number` is the item's position "
            "in the form, which is how a response CSV joins back to it: the "
            "per-item questions are worded identically across items, so the "
            "columns are told apart by order, not by name."
        ),
        "generated_on": GENERATED_ON,
        "generated_by": "eval/generate_rq2_form.py",
        "form_title": FORM_TITLE,
        "source": "data/rq2_pairs_FROZEN.json",
        "frozen_on": frozen["frozen_on"],
        "measured_length_ratio_answer_giving_over_scaffolded": frozen[
            "measured_length_ratio_answer_giving_over_scaffolded"
        ],
        "length_ratio_note": frozen["note"],
        "post_freeze_edits": frozen.get("post_freeze_edits", []),
        "n_items": len(pairs),
        "shown_as_A_counts": counts,
        "questions": {
            "which_helps": Q_WHICH_HELPS,
            "rate_a": Q_RATE_A,
            "rate_b": Q_RATE_B,
            "when_stuck": Q_WHEN_STUCK,
            "free_text": Q_FREE_TEXT,
            "overall": Q_OVERALL,
            "grade": Q_GRADE,
            "math_relationship": Q_MATH_RELATIONSHIP,
        },
        "ab_choices": AB_CHOICES,
        "scale": {"low": 1, "high": 5,
                  "low_label": SCALE_LOW_LABEL, "high_label": SCALE_HIGH_LABEL},
        "key": key,
    }


def main() -> int:
    with open(FROZEN_PATH, encoding="utf-8") as fh:
        frozen = json.load(fh)
    pairs = frozen["pairs"]
    ratio = frozen["measured_length_ratio_answer_giving_over_scaffolded"]

    items = build_items(pairs)
    check_blinding(pairs, items)

    gs = render_gs(items, ratio, frozen["frozen_on"])
    for forbidden in (SCAFFOLDED, ANSWER_GIVING, "_hidden", "rq2_answer_key"):
        # The header comment explains the blinding, so allow it there only.
        body = gs.split("*/", 1)[1]
        assert forbidden not in body, (
            f"generated .gs leaks {forbidden!r} outside its header comment"
        )
    with open(GS_PATH, "w", encoding="utf-8") as fh:
        fh.write(gs)

    answer_key = render_answer_key(pairs, frozen)
    with open(ANSWER_KEY_PATH, "w", encoding="utf-8") as fh:
        json.dump(answer_key, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"Wrote {os.path.relpath(GS_PATH, REPO_ROOT)}")
    print(f"Wrote {os.path.relpath(ANSWER_KEY_PATH, REPO_ROOT)}")
    print(f"  {len(items)} items, "
          f"{answer_key['shown_as_A_counts'][SCAFFOLDED]} lead scaffolded / "
          f"{answer_key['shown_as_A_counts'][ANSWER_GIVING]} lead answer-giving")
    print(f"  length ratio in closing message: {ratio}x")
    noted = [i["problemId"] for i in items if "note" in i]
    print(f"  per-item note on: {noted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
