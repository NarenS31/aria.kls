# Ethics, safeguarding, and privacy plan

Status: **planning document; not an IRB determination or legal advice**

## Before any recruitment

1. Identify an institution willing to serve as the responsible research
   organization and determine whether the activity is human-subjects research.
2. Obtain IRB or equivalent written determination before interacting with
   participants or accessing identifiable records.
3. Obtain school/district approval and execute required data-use agreements.
4. Use only institution-approved parental permission and student assent forms.
5. Register the protocol and analysis plan before outcome data are examined.
6. Train all staff in privacy, assent, safeguarding, and adverse-event handling.

HHS guidance notes that research involving children commonly requires parental
permission and affirmative child assent, with the reviewing IRB determining
the applicable category and any waiver. Mere failure to object is not assent:
<https://www.hhs.gov/ohrp/regulations-and-policy/guidance/faq/children-research/index.html>

## Minimal-risk design

- ARIA is a learning-support prototype, not a diagnostic, therapeutic, grading,
  disciplinary, or emergency system.
- Do not infer or display a disability, mental-health condition, or diagnosis.
- Avoid collecting diagnosis, medication, accommodation, or school-record data
  unless essential to an approved question.
- Students may skip any prompt, request a human, or stop without penalty.
- A qualified adult supervises prospective student sessions.
- The study does not replace instruction or withhold ordinary help.
- Outcome tests are brief and appropriate to current instruction.

## Assent

- Explain the activity in age-appropriate language.
- Ask for affirmative agreement, not passive compliance.
- Reconfirm willingness when a session resumes.
- Respect a student's refusal even when a parent permitted participation.
- Avoid recruitment by anyone who controls the student's grade when feasible.

## Safeguarding

Before deployment, the reviewing institution defines:

- who monitors sessions;
- the response to distress or a request to stop;
- the human-help escalation route;
- mandatory-reporting obligations and limits of confidentiality;
- the response to inappropriate model output;
- pause rules for repeated factual, privacy, or safety failures;
- emergency contacts shown to participants through approved materials.

The model does not independently manage disclosures of self-harm, abuse, or
imminent danger. A human follows the institution's approved procedure.

## FERPA and school records

Prefer study-created measures and random study IDs. If a school discloses
personally identifiable education records under FERPA's studies exception, the
study must qualify as being conducted for or on behalf of the school for an
allowed purpose and use a written agreement specifying purpose, access, use,
and destruction:
<https://studentprivacy.ed.gov/faq/may-educational-agency-or-institution-disclose-personally-identifiable-information-students>

Do not copy grades, IEP/504 information, disability status, or disciplinary
records into ARIA's learner model without explicit institutional authorization
and a necessity analysis.

## AI-specific disclosure

Participants are told:

- responses are produced by an automated research system;
- it can misunderstand reasoning and make factual mistakes;
- a human educator remains available;
- interaction text and specified logs are recorded for research;
- whether any third-party or cloud model receives their data;
- how long data are retained and who can access them.

If the approved study promises on-device processing, every production
dependency must be audited to ensure no content leaves the device.

## Fairness

- Test speech transcription and language interpretation separately.
- Report error rates by prespecified groups only with adequate sample,
  participant protections, and privacy-preserving reporting.
- Accessibility support must not silently alter condition assignment.
- A subgroup result cannot justify a disability-specific efficacy claim unless
  the study was designed and powered for it.

## Incident log

Record timestamp, study ID, system version, task, output, incident category,
severity, immediate action, reviewer, resolution, and whether enrollment
paused. Remove direct identifiers from the analytic incident file.

