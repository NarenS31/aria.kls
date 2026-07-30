# External actions required for real evidence

Code and documents cannot complete these gates.

## 1. Responsible institution and mentor

- Identify a university, research organization, or school research office
  willing to supervise the study.
- Confirm the principal investigator, student-researcher role, and data owner.
- Obtain written human-subjects/IRB determination before recruitment or access
  to identifiable student data.
- Have a statistician review the final design and power simulation.

Suggested mentor outreach:

> Subject: Student research collaboration on evidence-based AI tutoring  
>  
> We built ARIA, a local research prototype that asks problem-grounded
> metacognitive questions. Current results are explicitly synthetic or
> software-level. We have prepared an observable reasoning-move codebook,
> blinded educator protocol, task validation packet, data plan, power tooling,
> and a draft active-controlled student study. We are looking for qualified
> supervision to review the theory, obtain an institutional determination, and
> prevent unsupported claims. Would you be willing to review the materials or
> connect us with a learning-science/educational-technology researcher?

## 2. Independent task reviewers

- Recruit at least two subject-qualified educators per task domain.
- Complete `research/packets/task_review.csv` independently.
- Preserve original reviews and use a third reviewer for unresolved
  correctness disagreements.
- Do not mark tasks approved in code before reviews are complete.

## 3. Blinded intervention raters

- Obtain the appropriate institutional determination for adult educator
  participants.
- Give each rater a separately randomized packet and the locked codebook.
- Keep the condition key inaccessible until every file is submitted.
- Run `eval/analyze_educator_ratings.py`; publish agreement and failures.

## 4. Real-language annotators

- Approve the licensed/prospective corpus and deidentification procedure.
- Train two annotators on the calibration set.
- Freeze the codebook, then independently label the confirmatory set.
- Keep model predictions hidden.
- Run `eval/analyze_reasoning_move_annotations.py`.

## 5. Preregistration

- Replace every `[pending]` field.
- Freeze task versions, hypotheses, outcomes, exclusions, randomization,
  sample-size assumptions, stop rules, and analysis.
- Register on OSF or the institution's accepted registry before inspecting
  confirmatory outcomes.
- Commit the timestamped registered version and record all deviations.

## 6. Student feasibility study

- Obtain school approval, parental permission, and affirmative student assent.
- Train the supervising adult.
- Run the small feasibility protocol before an efficacy study.
- Repair critical failures and repeat verification when the intervention
  changes materially.

## 7. Controlled learning study

- Use independently authored outcome tasks.
- Conceal allocation and blind outcome scorers.
- Preserve intention-to-treat assignment.
- Track attrition, fidelity, contamination, and adverse events.
- Report null and negative results.

## Evidence that must remain pending today

- educator-validated task count;
- educator-rated response advantage;
- real-human reasoning-move accuracy;
- usability and safety in students;
- causal learning effect;
- delayed retention;
- unprompted transfer;
- ADHD- or neurodivergence-specific effectiveness.

