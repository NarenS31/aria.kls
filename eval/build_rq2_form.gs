/**
 * Builds the RQ2 rating form: scaffolded vs answer-giving tutoring responses.
 *
 * GENERATED FILE — do not hand-edit. Regenerate with:
 *     python3.11 eval/generate_rq2_form.py
 * Source of truth: data/rq2_pairs_FROZEN.json (frozen 2026-08-18).
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
 *   measured gap (1.31x) by style rather than by letter, and says the labels
 *   were shuffled. To ship the original wording instead, edit
 *   confirmation_message() in eval/generate_rq2_form.py and regenerate.
 *
 * FORMATTING
 *   The Forms service has no rich text for display items, so there is no way to
 *   italicise the per-item note. It is set as help text, which Google renders
 *   smaller and lighter than a title — the closest the API gets.
 */

var FORM_TITLE = "Comparing Two Tutoring Response Styles (10 items, ~10 min)";

var FORM_DESCRIPTION = "We're comparing two different styles of AI tutoring responses to the same math, English, science, and coding problems. For each item you'll see a problem, what the student had already tried, and two possible responses (A and B) — please read both and answer a few quick questions. There are no wrong answers. Your responses are anonymous and used only for a student research project. You can stop at any time.";

var CONSENT_TITLE = "By continuing, you confirm you're voluntarily participating and understand your responses are anonymous.";

var CONSENT_CHOICE = "I understand and agree to continue";

var STUDENT_LABEL = "What the student had tried so far:";
var RESPONSE_A_LABEL = "Response A:";
var RESPONSE_B_LABEL = "Response B:";

var Q_WHICH_HELPS = "Which response would actually help you understand how to solve this, not just get the answer?";
var Q_RATE_A = "Rate Response A — how helpful is this for actually learning the concept?";
var Q_RATE_B = "Rate Response B — how helpful is this for actually learning the concept?";
var Q_WHEN_STUCK = "If you were stuck and stressed, which response would you rather receive right now?";
var Q_FREE_TEXT = "Anything either response got wrong, confusing, or annoying?";

var AB_CHOICES = ["A", "B", "About the same"];
var SCALE_LOW_LABEL = "Not helpful at all";
var SCALE_HIGH_LABEL = "Extremely helpful";

var CLOSING_SECTION_TITLE = "A few last questions";
var Q_OVERALL = "Across all the items you just saw, which style did you generally prefer?";
var OVERALL_CHOICES = ["Consistently A", "Consistently B", "Depended on the item"];
var Q_GRADE = "What grade or year are you in?";
var Q_MATH_RELATIONSHIP = "How would you describe your relationship with math?";
var MATH_RELATIONSHIP_CHOICES = ["I'm fine at it, but I don't really enjoy it", "I struggle with specific areas", "I struggle with it a lot", "I actively avoid it", "I actually enjoy it"];

var CONFIRMATION_MESSAGE = "Thank you — this is genuinely useful. One honest note: the two response styles in this survey are not perfectly matched in length. The style that walks through the full explanation runs about 30% longer on average (measured 1.31× in this set), and it wasn't always the one labelled A or B — the labels were shuffled from item to item. Keep that in mind when you think back on your ratings.";

/**
 * The ten items, in the order they are shown. `responseA` / `responseB` are
 * already in display order; this file does not know, and must not know, which
 * condition either of them came from.
 */
var STIMULI = [
  {
    "problemId": "math-geometry-04",
    "problemText": "Two angles form a linear pair. One is 68 degrees. Find the other.",
    "studentInput": "Umm... I think the other angle should be 90 degrees 'cause they add up to a straight line?",
    "responseA": "Your step in “Umm... I think the other angle should be” matches the task model, so the direction you chose is one you can keep building on. What operation should come next on the equation you now have?",
    "responseB": "Let me walk you through this one. First, linear-pair angles sum to 180 degrees. Then, compute 180 - 68. Next, check that the pair sums to 180. That gives you the answer: 112 degrees. The idea doing the work here is diagram reasoning."
  },
  {
    "problemId": "math-algebra-09",
    "problemText": "Solve for x: 2(x + 5) - 3 = 15",
    "studentInput": "I thought I had to get rid of the minus sign when I combined the numbers... but what about when I distribute the 2 into the parentheses?",
    "responseA": "Let me walk you through this one. First, distribute and combine constants to get 2x + 7 = 15. Then, use inverse operations while keeping both sides balanced. Next, check the result by substituting x = 4 into the original equation. That gives you the answer: x = 4. The idea doing the work here is equivalent equations.",
    "responseB": "Your step in “I thought I had to get rid of” matches the task model, so the direction you chose is one you can keep building on. What operation should come next on the equation you now have?"
  },
  {
    "problemId": "math-statistics-02",
    "problemText": "Find the mean of 6, 8, 9, 12, 15.",
    "studentInput": "I don't know... um, I think maybe we're supposed to divide by the largest number? Like, what if I said 6 was too small or something?",
    "responseA": "Your step in “I don't know... um, I think maybe we're” matches the task model, so the direction you chose is one you can keep building on. What operation should come next on the equation you now have?",
    "responseB": "Let me walk you through this one. First, add the five values to get 50. Then, divide by the number of values, 5. Next, check that 10 is near the center. That gives you the answer: 10. The idea doing the work here is data interpretation."
  },
  {
    "problemId": "english-revision-04",
    "problemText": "Choose the correct word: The new schedule had a positive affect/effect on attendance.",
    "studentInput": "I think I got it right... we need 'effect' because it sounds like it matches with 'schedule', but then what about when we're talking about how someone feels? Can we use 'affect' to describe that too, maybe?",
    "responseA": "You are working from “I got it right... we need 'effect',” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using clarity?",
    "responseB": "Let me walk you through this one. First, identify the sentence-level problem. Then, revise only what is needed. Next, read the revision once for clarity and grammar. That gets you to what the task is looking for: effect. The idea doing the work here is clarity.",
    "note": "Note: in this item, the student's own writing already includes the word 'effect' — that's not something either response revealed to them."
  },
  {
    "problemId": "english-argument-01",
    "problemText": "Revise this thesis to be specific and arguable: Social media affects teenagers in many ways.",
    "studentInput": "I don't get why I need to explain that social media is bad... isn't it just kinda obvious?",
    "responseA": "You have put down “I don't get why I need to explain,” which is something to work from even though the next move is not settled yet. Start with the task itself: Identify the exact job the sentence must do. What would that look like here?",
    "responseB": "Let me walk you through this one. First, identify the exact job the sentence must do. Then, draft one focused sentence that does that job. Next, check that every word adds a specific idea or logical connection. That gets you to what the task is looking for: A strong revision makes one debatable claim about a specific effect of social media on teenagers. The idea doing the work here is claim."
  },
  {
    "problemId": "english-reading-01",
    "problemText": "Read: \"Mara stood outside the crowded cafeteria, rereading the menu though she already knew every item.\"\n\nHow does the detail about rereading the menu reveal Mara's feelings?",
    "studentInput": "It shows that Mara is... um, I think she's feeling anxious or nervous about something, because if she already knew every item on the menu, then why would she be rereading it?",
    "responseA": "You are working from “It shows that Mara is... um, I think,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using textual evidence?",
    "responseB": "Let me walk you through this one. First, identify the exact word, detail, or change named in the question. Then, make an inference that fits that evidence. Next, explain the connection in one or two precise sentences. That gets you to what the task is looking for: It suggests she is delaying entering because she feels anxious or out of place. The idea doing the work here is textual evidence."
  },
  {
    "problemId": "science-07",
    "problemText": "Why do seasons occur on Earth?",
    "studentInput": "I think... um, day and night cause seasons? Because like, when it's daytime in one place, it's nighttime somewhere else? Is that right?",
    "responseA": "Let me walk you through this one. First, identify Earth's axial tilt. Then, connect tilt to sunlight angle and day length. Next, distinguish this from Earth-Sun distance. That gives you the answer: Earth's axial tilt changes the angle and duration of sunlight during its orbit. The idea doing the work here is model selection.",
    "responseB": "You are working from “um, day and night cause seasons,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using model selection?"
  },
  {
    "problemId": "science-03",
    "problemText": "How many moles are in 36 g of water if its molar mass is 18 g/mol?",
    "studentInput": "Wait, I think I made a mistake... like, oxygen has a lot of mass too, doesn't it? But what about if we use its mass instead, wouldn't that be easier?",
    "responseA": "Let me walk you through this one. First, use moles = mass / molar mass. Then, substitute 36 / 18. Next, check units cancel to moles. That gives you the answer: 2 mol. The idea doing the work here is model selection.",
    "responseB": "You are working from “Wait, I think I made a mistake... like,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using model selection?"
  },
  {
    "problemId": "coding-05",
    "problemText": "A function should return the square of x. Complete: def square(x): return ____",
    "studentInput": "Um, I think I wrote something like `return x*x`, but it doesn't seem to work... Can we try printing what the function does instead?",
    "responseA": "Let me walk you through this one. First, use the parameter x. Then, multiply x by itself. Next, return the expression. That gives you the answer: x * x. The idea doing the work here is trace execution.",
    "responseB": "You are working from “Um, I think I wrote something like `return,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using trace execution?"
  },
  {
    "problemId": "coding-09",
    "problemText": "What does SQL `SELECT name FROM students WHERE grade >= 90;` return?",
    "studentInput": "I think it should exclude all students with a 90 or higher, but what about if there are multiple grades like 99? Does it only exclude people with exactly 90?",
    "responseA": "Let me walk you through this one. First, SELECT identifies the returned column. Then, FROM identifies the table. Next, WHERE filters rows using an inclusive comparison. That gives you the answer: Names of students whose grade is at least 90. The idea doing the work here is trace execution.",
    "responseB": "You are working from “it should exclude all students with a 90,” and you stopped at the point where you were no longer sure instead of guessing past it. What is one next step you could try using trace execution?"
  }
];

/**
 * Build the form. Returns { publishedUrl, editUrl, itemCount }.
 */
function buildRq2Form() {
  var form = FormApp.create(FORM_TITLE);
  form.setDescription(FORM_DESCRIPTION);
  form.setProgressBar(true);
  form.setAllowResponseEdits(false);
  form.setShowLinkToRespondAgain(false);
  form.setConfirmationMessage(CONFIRMATION_MESSAGE);

  // Anonymity. These are the settings that differ between account types and
  // across Apps Script versions, so a missing one must not abort the build —
  // every one of them is already the default for a newly created form.
  try { form.setCollectEmail(false); } catch (e) {}
  try { form.setLimitOneResponsePerUser(false); } catch (e) {}
  try { form.setRequireLogin(false); } catch (e) {}

  form.addCheckboxItem()
      .setTitle(CONSENT_TITLE)
      .setChoiceValues([CONSENT_CHOICE])
      .setRequired(true);

  for (var i = 0; i < STIMULI.length; i++) {
    addItemSection(form, STIMULI[i]);
  }

  addClosingSection(form);

  // Newer Apps Script requires an explicit publish step; older versions have no
  // such method and forms are reachable as soon as they exist.
  try { form.setPublished(true); } catch (e) {}

  var result = {
    publishedUrl: form.getPublishedUrl(),
    editUrl: form.getEditUrl(),
    itemCount: STIMULI.length
  };
  Logger.log('Built "%s" with %s items.', FORM_TITLE, result.itemCount);
  Logger.log("SHARE THIS URL: %s", result.publishedUrl);
  Logger.log("EDIT URL:       %s", result.editUrl);
  return result;
}

/** One page per item: the problem, the attempt, both responses, five questions. */
function addItemSection(form, item) {
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
  if (item.note) {
    form.addSectionHeaderItem().setHelpText(item.note);
  }

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
}

/** Overall preference plus the two background questions. */
function addClosingSection(form) {
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
}
