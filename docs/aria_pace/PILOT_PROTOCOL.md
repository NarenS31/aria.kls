# Pilot protocol (future, 2 weeks, low-stakes)

> **DRAFT, NOT APPROVED. Nothing in this document has been submitted to, reviewed by or approved by a school, a teacher, parents or guardians, an ethics board (IRB) or any other body. No student data is being collected. No pilot is scheduled or open.** This is a planning draft for the people who would have to approve it. It is not legal advice.

## 1. What this pilot would be

A small, low-stakes, two-week classroom pilot of the ARIA tutor on typed math problems (algebra, trigonometry, calculus; 30 problems in the bank), run by a teacher in class time, to learn whether the tutor is understandable, usable and safe and what teachers and students think of when it speaks. It is **a feasibility and usability pilot, not an effectiveness study.** It would make no claim about learning gains, and results would be reported as a pilot.

Proposed shape (to be agreed with the school, **draft**): 2 to 3 sessions of 30 to 45 minutes over two weeks, in class, on school-approved devices, with the teacher present. This matches the proposal on the ARIA website (2 to 3 sessions, 30 to 45 minutes, format agreed with the school).

## 2. What must exist before anything starts (all draft, none exist yet)

| Requirement | Status |
|---|---|
| A **school and teacher sponsor** who wants to host it and can stop it at any time | not found: none identified |
| **Parent or guardian consent**, and student assent, in forms the school and the ethics reviewer approve | not drafted |
| An **ethics review**: IRB approval, or a written exempt determination from the body with authority over the school | not requested |
| An **adult responsible for the data** (a named school staff member or an adult investigator of record), who holds the linking key and answers data requests | not identified |
| An adult faculty or research sponsor for the student researchers | open (no advisor confirmed) |
| Completion of the project's own release checks: offline run verified on the venue device, teacher-summary tests, frozen build, consent review | not complete (see `docs/DEMO_RUNBOOK.md`) |

**Random tokens are not enough for FERPA.** A random study code that we generate does not make records anonymous if the school can link the code back to a student (by roster, login, device, time or class period). Under the US student-records law (FERPA), records that identify a student are protected, so this plan assumes that any pilot data are **education records under the school's control** until the school's privacy officer or the IRB decides otherwise in writing. Other laws that may apply (children's online privacy rules, state student-privacy laws) are **to verify with the school**; they are not assessed here.

## 3. What would be collected (proposed; to be set by the approval process)

- **From the tutor's per-turn log only**, as already implemented: numbers and short identifiers, **no student text and no tutor text** (`agent/demo_log.py:19-42`). Fields: a random session id, attempt number, problem id, turn number, the SymPy verdict, whether ARIA replied and which type, the hint counter, seconds since the attempt started, and the pace-gate numbers.
- **A study code per student**, assigned by the data steward (the responsible adult). The key linking codes to names is held by that adult at the school and **never given to the researchers.** The tutor does not ask for a name, email or any account.
- **Optional short instruments** (section 5), if approved; see the caveat in section 6.
- **Not collected:** student names, free-text answers, audio, video, screenshots, grades, medical or disability information, or anything from outside the session. If the team ever wants free text, that is a separate approval.
- Where the data lives: the tutor runs locally and writes its log to the device (`docs/DEMO_RUNBOOK.md` sections 3 and 8). The pilot would keep it that way: no cloud storage, no network transfer. **The tutor's storage and network behaviour must still be verified on the school's devices before any use**; the ARIA privacy page says it is not yet promised.

## 4. Retention, access, withdrawal (proposed, not approved)

| Topic | Proposal |
|---|---|
| **Retention** | Raw logs stay on the school-controlled device or drive. The linking key is destroyed at the end of the pilot unless the school's rules require otherwise. De-identified analysis files kept only as long as the ethics approval says (not decided: **not found**). Everything else deleted at the end. |
| **Who can see it** | The teacher and the data steward can see the logs and the Teacher summary for their own class. The student researchers see only de-identified study-code data the data steward releases. No one else. No third-party service, no analytics, no advertising. |
| **Withdrawal** | A student or a parent or guardian can withdraw at any time without penalty and without telling us why; the teacher can stop the pilot at any time. On withdrawal the data steward deletes that code's data from the working files. Once data have been de-identified and combined into summaries, they can no longer be removed; the consent form must say so. Declining or withdrawing never affects grades or class standing. |
| **Safety stops** | The teacher can stop a session immediately. Any reply that gives the student's answer, states wrong math, or is inappropriate is logged by the teacher as an incident and the pilot pauses until reviewed. |

## 5. Measures (all draft)

**Primary, from the log** (defined and computed by `docs/aria_pace/PILOT_MEASURES.md` and `scripts/pilot_measures.py`): correction after a hint; hints per problem and `hint_level` reached (**not a three-level engine**); time before the first hint as a **proxy** for productive struggle; the **exploratory** friction index. These describe what happened in sessions. They are not outcomes.

**Optional short instruments, only if the ethics review approves them** (section 6 below explains the caveats):
- A school-age **metacognitive awareness** measure: the Junior Metacognitive Awareness Inventory, Sperling, R. A., Howard, B. C., Miller, L. A., & Murphy, C. (2002), *Measures of children's knowledge and regulation of cognition*, Contemporary Educational Psychology 27(1), 51-79 (citation verified in Crossref, 2026-10-21). Which version fits which grade, its licence and its administration time are **not verified here** and must be checked in the source.
- **NASA-TLX** for perceived workload: Hart, S. G., & Staveland, L. E. (1988), *Development of NASA-TLX (Task Load Index): Results of empirical and theoretical research*, Advances in Psychology 52, 139-183 (title and pages verified in Crossref, 2026-10-21; the author fields were not returned there).

## 6. Caveat on the instruments

Both instruments were developed for particular populations (NASA-TLX for adult task performance). **Adapting wording or dropping or merging subscales for school-age students changes what is measured and needs its own validation.** A shortened or reworded version must not be reported as the original instrument, and no cut-off or norm from the originals may be applied to it. If an instrument is used unchanged, its permitted age range and licence still have to be confirmed. If that cannot be done, the pilot should use plain questions about what students found helpful, confusing or annoying instead, and say so.

## 7. Proposed A/B study (requires approval and real participants; proves nothing alone)

**Not part of the two-week pilot unless separately approved.** A comparison of ARIA against a **direct-answer LLM** (a chat model that gives answers), with an **unassisted transfer problem** as the outcome.

- **Design (proposed):** students in one or more classes are randomly assigned (by the data steward, using study codes) to ARIA or to the direct-answer condition for the same practice problems and the same time on task. Afterwards every student solves a **new problem set alone** (no tool, no help), scored with a written rubric by someone who does not know the condition (blinded). A delayed repeat after a set interval would test retention.
- **Outcomes (to be pre-registered before any data):** unassisted transfer score; time to a correct answer; abandonment; plus the section 5 log measures.
- **Needs before it can run:** everything in section 2; a safe, approved comparator (a model that gives answers must be configured and reviewed so it cannot expose students to unsafe content); a pre-registered analysis plan; and a sample size from a power calculation (**not found: not done**; no pilot data exist to base it on).
- **What it could and could not show:** at best, a difference in unassisted performance between two tools in one setting. It could not show why, would not generalise beyond its students, tasks and time frame, and cannot by itself show that ARIA improves learning, since one study, one site and one outcome prove nothing alone. A null or negative result would be reported with equal prominence.

## 8. Reporting rules

Pilot results, if any, are reported as a feasibility pilot with limits; no learning-gain claim; no claim about any group of learners; the quiet / balanced / supportive settings stay cost weights, not learner types; and nothing from the pilot is shared outside the approved people without the approvals above.

## 9. Open items (not found, not decided)

School and teacher sponsor; ethics body and its requirements; consent and assent forms; the data steward and adult sponsor; retention periods; devices and their verified offline behaviour; instrument versions, licences and age ranges; legal review of student-privacy rules; sample size for any comparison; the comparator model and its safety review; the pre-registration.

## 10. Design fix: two different costs must not be conflated (appended 2026-10-22; DRAFT)

**Everything in this section is draft. Any use with students requires an ethics review (IRB or exempt determination), a school sponsor, parent/guardian consent (and student assent where applicable), and an adult responsible for the data (section 2).** Nothing here has been approved or run.

The offline cost J = w_I x (unnecessary interventions) + w_D x (delayed help) (`aria_pace/SPEC.md:135`) mixes two different quantities, and one pilot cannot estimate both from the same comparison. They need separate questions:

- **(a) Cost of delayed help (w_D).** Compare **immediate** hints with **delayed** hints (for example after one more attempt, or after a fixed wait) for the same kind of error. The outcome is what the delay costs the student: time to a correct answer, number of further errors, abandonment, correction after the hint. Students are in the comparison only because they would receive help in either arm.
- **(b) Cost of interruption (w_I).** Compare **hint on error** with **no hint** among students who **would have succeeded alone**. The key difficulty is that "would have succeeded alone" is only known afterwards. A defensible design assigns the no-hint arm by randomisation at the error (so both arms contain the same mix of students), with a safe fallback hint after a fixed number of further errors, and analyses the subset who succeeded without help in the no-hint arm. A within-student crossover on matched problems is an alternative. Outcomes: success without help, time to success, abandonment, and a short self-report that is itself validated for the age group (section 6).
- **Recovery latency** (how quickly a student fixes the step after an intervention) can **inform** these comparisons but **cannot derive the weights**: a faster recovery is not by itself a benefit that outweighs an interruption. The weights remain a **stated value judgment** (the three profiles are cost weights, not learner types), and the pilot at most shows how sensitive the conclusions are to them.
- **What neither design shows:** learning gains or retention. Those would need an unassisted transfer outcome (section 7) and a pre-registered analysis.
