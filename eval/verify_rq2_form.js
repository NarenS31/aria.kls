/**
 * Verify eval/build_rq2_form.gs without touching Google.
 *
 *     node eval/verify_rq2_form.js
 *
 * Runs the generated Apps Script against a mock Forms service, records every
 * item it creates, and checks the result against data/rq2_pairs_FROZEN.json and
 * data/rq2_answer_key.json.
 *
 * The check that matters is the blinding. Whether the text under "Response A"
 * is the one the frozen file assigned to A is invisible on inspection — a
 * swapped item looks like a perfectly good item and silently inverts that
 * item's result. So every item is checked, not a sample, and the answer key is
 * checked to agree with the form rather than being trusted alongside it.
 *
 * What this cannot check: that Google accepts the calls. The mock implements
 * the Forms service as documented, and deliberately omits setRequireLogin and
 * setPublished so the script's defensive wrapping around the methods that vary
 * by account type is actually exercised.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const REPO_ROOT = path.dirname(__dirname);
const GS_PATH = path.join(REPO_ROOT, "eval", "build_rq2_form.gs");
const FROZEN_PATH = path.join(REPO_ROOT, "data", "rq2_pairs_FROZEN.json");
const KEY_PATH = path.join(REPO_ROOT, "data", "rq2_answer_key.json");

const failures = [];
const checks = [];

function check(name, condition, detail) {
  checks.push({ name, ok: !!condition, detail });
  if (!condition) failures.push(`${name}${detail ? " — " + detail : ""}`);
}

// ------------------------------------------------------------------
// Mock Forms service
// ------------------------------------------------------------------

function makeForm(title) {
  const form = {
    title,
    description: null,
    confirmationMessage: null,
    settings: {},
    items: [],
    calledMissing: [],
  };

  function record(type) {
    const item = { type, title: null, helpText: null, choices: null,
                   required: false, bounds: null, labels: null };
    form.items.push(item);
    const api = {
      setTitle(v) { item.title = v; return api; },
      setHelpText(v) { item.helpText = v; return api; },
      setChoiceValues(v) { item.choices = v.slice(); return api; },
      setRequired(v) { item.required = v; return api; },
      setBounds(lo, hi) { item.bounds = [lo, hi]; return api; },
      setLabels(lo, hi) { item.labels = [lo, hi]; return api; },
    };
    return api;
  }

  form.setDescription = (v) => { form.description = v; return form; };
  form.setConfirmationMessage = (v) => { form.confirmationMessage = v; return form; };
  form.setProgressBar = (v) => { form.settings.progressBar = v; return form; };
  form.setAllowResponseEdits = (v) => { form.settings.allowEdits = v; return form; };
  form.setShowLinkToRespondAgain = (v) => { form.settings.respondAgain = v; return form; };
  form.setCollectEmail = (v) => { form.settings.collectEmail = v; return form; };
  form.setLimitOneResponsePerUser = (v) => { form.settings.limitOne = v; return form; };
  // setRequireLogin and setPublished are intentionally NOT defined.

  form.addPageBreakItem = () => record("PAGE_BREAK");
  form.addSectionHeaderItem = () => record("SECTION_HEADER");
  form.addCheckboxItem = () => record("CHECKBOX");
  form.addMultipleChoiceItem = () => record("MULTIPLE_CHOICE");
  form.addScaleItem = () => record("SCALE");
  form.addParagraphTextItem = () => record("PARAGRAPH_TEXT");
  form.addTextItem = () => record("TEXT");

  form.getPublishedUrl = () => "https://docs.google.com/forms/d/e/MOCK/viewform";
  form.getEditUrl = () => "https://docs.google.com/forms/d/MOCK/edit";
  return form;
}

let createdForm = null;
const logLines = [];
const sandbox = {
  FormApp: {
    create(title) { createdForm = makeForm(title); return createdForm; },
  },
  Logger: { log: (...args) => logLines.push(args.join(" ")) },
};

// ------------------------------------------------------------------
// Run the script
// ------------------------------------------------------------------

const source = fs.readFileSync(GS_PATH, "utf8");
const context = vm.createContext(sandbox);
new vm.Script(source, { filename: "build_rq2_form.gs" }).runInContext(context);

let result;
try {
  result = context.buildRq2Form();
} catch (err) {
  console.error("build_rq2_form.gs threw while building the form:\n", err);
  process.exit(1);
}

const form = createdForm;
const frozen = JSON.parse(fs.readFileSync(FROZEN_PATH, "utf8"));
const answerKey = JSON.parse(fs.readFileSync(KEY_PATH, "utf8"));
const pairs = frozen.pairs;

// ------------------------------------------------------------------
// Form-level
// ------------------------------------------------------------------

check("form is created with the specified title",
  form.title === "Comparing Two Tutoring Response Styles (10 items, ~10 min)",
  form.title);
check("description mentions anonymity and stopping at any time",
  /anonymous/.test(form.description) && /stop at any time/.test(form.description));
check("script survives Forms methods that do not exist on this account type",
  result && result.itemCount === 10,
  "setRequireLogin/setPublished are absent from the mock and must not abort");

// ------------------------------------------------------------------
// Structure: consent, then 10 item pages, then the closing page
// ------------------------------------------------------------------

const first = form.items[0];
check("consent is the first item, a required checkbox with one option",
  first.type === "CHECKBOX" && first.required === true &&
  first.choices.length === 1 &&
  first.choices[0] === "I understand and agree to continue" &&
  /voluntarily participating/.test(first.title));

const pageBreaks = form.items.filter((i) => i.type === "PAGE_BREAK");
check("11 sections: one per item plus the closing section",
  pageBreaks.length === 11, `found ${pageBreaks.length}`);

// Split the item stream into sections at each page break.
const sections = [];
for (const item of form.items) {
  if (item.type === "PAGE_BREAK") sections.push({ page: item, items: [] });
  else if (sections.length) sections[sections.length - 1].items.push(item);
}
const itemSections = sections.slice(0, 10);
const closing = sections[10];

check("10 item sections", itemSections.length === 10);

const Q_WHICH = "Which response would actually help you understand how to solve this, not just get the answer?";
const Q_RATE_A = "Rate Response A — how helpful is this for actually learning the concept?";
const Q_RATE_B = "Rate Response B — how helpful is this for actually learning the concept?";
const Q_STUCK = "If you were stuck and stressed, which response would you rather receive right now?";
const Q_FREE = "Anything either response got wrong, confusing, or annoying?";

let blindingOk = 0;
let notedItems = [];

itemSections.forEach((section, idx) => {
  const pair = pairs[idx];
  const id = pair.problem_id;
  const label = `item ${idx + 1} (${id})`;

  check(`${label}: section header is the problem text`,
    section.page.title === pair.problem_text);

  const headers = section.items.filter((i) => i.type === "SECTION_HEADER");
  const studentBlock = headers.find((h) => h.title === "What the student had tried so far:");
  const aBlock = headers.find((h) => h.title === "Response A:");
  const bBlock = headers.find((h) => h.title === "Response B:");

  check(`${label}: shows the student's attempt`,
    studentBlock && studentBlock.helpText === pair.student_input);

  // The blinding check.
  const shownAsA = pair.display_order.shown_as_A;
  const expectedA = shownAsA === "scaffolded"
    ? pair.response_scaffolded : pair.response_answer_giving;
  const expectedB = shownAsA === "scaffolded"
    ? pair.response_answer_giving : pair.response_scaffolded;

  const aOk = aBlock && aBlock.helpText === expectedA;
  const bOk = bBlock && bBlock.helpText === expectedB;
  check(`${label}: Response A holds the condition the frozen file assigned to A`,
    aOk, aOk ? `${shownAsA} (correct)` : "MISMATCH");
  check(`${label}: Response B holds the other condition`, bOk);
  if (aOk && bOk) blindingOk += 1;

  // The answer key must describe the same arrangement the form built.
  const keyed = answerKey.key[id];
  check(`${label}: answer key agrees with the form`,
    keyed && keyed.A === shownAsA && keyed.form_item_number === idx + 1);

  // Question block, in order.
  const questions = section.items.filter((i) => i.type !== "SECTION_HEADER");
  const titles = questions.map((q) => q.title);
  check(`${label}: five questions in the specified order`,
    titles.length === 5 &&
    titles[0] === Q_WHICH && titles[1] === Q_RATE_A &&
    titles[2] === Q_RATE_B && titles[3] === Q_STUCK && titles[4] === Q_FREE,
    titles.length !== 5 ? `found ${titles.length}` : undefined);

  check(`${label}: A/B choices on both preference questions`,
    JSON.stringify(questions[0].choices) === JSON.stringify(["A", "B", "About the same"]) &&
    JSON.stringify(questions[3].choices) === JSON.stringify(["A", "B", "About the same"]));
  check(`${label}: 1-5 scales with the specified end labels`,
    JSON.stringify(questions[1].bounds) === "[1,5]" &&
    JSON.stringify(questions[2].bounds) === "[1,5]" &&
    JSON.stringify(questions[1].labels) === JSON.stringify(["Not helpful at all", "Extremely helpful"]) &&
    JSON.stringify(questions[2].labels) === JSON.stringify(["Not helpful at all", "Extremely helpful"]));
  check(`${label}: the free-text question is the only optional one`,
    questions.slice(0, 4).every((q) => q.required === true) &&
    questions[4].required === false);

  // The per-item note: present on exactly one item, and placed before the
  // questions so it is read before anything is answered.
  const note = headers.find((h) => !h.title && h.helpText);
  if (note) {
    notedItems.push(id);
    const noteIndex = section.items.indexOf(note);
    const firstQuestionIndex = section.items.indexOf(questions[0]);
    check(`${label}: the note comes before the first question`,
      noteIndex < firstQuestionIndex);
    check(`${label}: note text is the specified footnote`,
      note.helpText === "Note: in this item, the student's own writing already " +
      "includes the word 'effect' — that's not something either response " +
      "revealed to them.");
  }
});

check("blinding correct on all 10 items", blindingOk === 10, `${blindingOk}/10`);
check("the per-item note appears on english-revision-04 and nowhere else",
  notedItems.length === 1 && notedItems[0] === "english-revision-04",
  notedItems.join(", ") || "none");

// Both orders must occur, or the labels are a constant and the blinding is fake.
const leadCounts = pairs.reduce((acc, p) => {
  acc[p.display_order.shown_as_A] = (acc[p.display_order.shown_as_A] || 0) + 1;
  return acc;
}, {});
check("A is not always the same condition",
  leadCounts.scaffolded > 0 && leadCounts.answer_giving > 0,
  `${leadCounts.scaffolded} lead scaffolded / ${leadCounts.answer_giving} lead answer-giving`);

// ------------------------------------------------------------------
// Closing section
// ------------------------------------------------------------------

check("closing section asks overall preference, grade, and math relationship",
  closing && closing.items.length === 3 &&
  closing.items[0].title === "Across all the items you just saw, which style did you generally prefer?" &&
  JSON.stringify(closing.items[0].choices) === JSON.stringify(["Consistently A", "Consistently B", "Depended on the item"]) &&
  closing.items[1].title === "What grade or year are you in?" &&
  closing.items[2].title === "How would you describe your relationship with math?" &&
  closing.items[2].choices.length === 5);

// ------------------------------------------------------------------
// Confirmation message: the real measured ratio, no placeholder
// ------------------------------------------------------------------

const ratio = frozen.measured_length_ratio_answer_giving_over_scaffolded;
const msg = form.confirmationMessage || "";
check("closing message states the measured ratio from the frozen file",
  msg.includes(String(ratio) + "×"), `expected ${ratio}× in the message`);
check("closing message carries no placeholder",
  !/TODO|TBD|X\.XX|\{\{|<[a-z_]+>/i.test(msg));
check("closing message does not bind the longer style to a fixed letter",
  !/Response B style/.test(msg) && /shuffled/.test(msg));

// ------------------------------------------------------------------
// Leakage: nothing in the script body names a condition or the answer key
// ------------------------------------------------------------------

const body = source.split("*/").slice(1).join("*/");
for (const word of ["scaffolded", "answer_giving", "_hidden", "rq2_answer_key"]) {
  check(`the script body never mentions ${word}`, !body.includes(word));
}
check("no answer-key path appears anywhere in the form's own text",
  !JSON.stringify([form.title, form.description, form.confirmationMessage,
                   form.items]).includes("answer_key"));

// ------------------------------------------------------------------
// Report
// ------------------------------------------------------------------

const passed = checks.filter((c) => c.ok).length;
console.log(`Verifying ${path.relative(REPO_ROOT, GS_PATH)} against the frozen set`);
console.log("-".repeat(74));
for (const c of checks) {
  if (!c.ok) console.log(`[FAIL] ${c.name}${c.detail ? " — " + c.detail : ""}`);
}
console.log(`${passed}/${checks.length} checks passed`);
console.log("-".repeat(74));
console.log("Blinding, item by item (form vs frozen file):");
itemSections.forEach((section, idx) => {
  const pair = pairs[idx];
  const shownAsA = pair.display_order.shown_as_A;
  const aBlock = section.items.find((i) => i.title === "Response A:");
  const ok = aBlock && aBlock.helpText ===
    (shownAsA === "scaffolded" ? pair.response_scaffolded : pair.response_answer_giving);
  console.log(
    `  ${String(idx + 1).padStart(2)}. ${pair.problem_id.padEnd(20)} ` +
    `A = ${shownAsA.padEnd(13)} ${ok ? "OK" : "MISMATCH"}`
  );
});
console.log("-".repeat(74));
console.log("Form log output on build:");
for (const line of logLines) console.log("  " + line);

process.exit(failures.length ? 1 : 0);
