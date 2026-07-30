# ARIA data-management plan

Status: draft for institutional review.

## Data minimization

| Data | Purpose | Default handling |
|---|---|---|
| Random study ID | Link sessions without names | Stored in research records |
| Parent permission and assent | Document authorization | Stored separately from research data |
| Task responses and tutor outputs | Primary process data | Pseudonymized |
| Timestamps and latency | Fidelity and usability | Rounded or minimized when exact time is unnecessary |
| Pre/post/delayed scores | Learning outcomes | Pseudonymized |
| Confidence ratings | Calibration outcome | Pseudonymized |
| Audio | Speech transcription, only if approved | Encrypt; delete after verified transcription on the approved schedule |
| Transcript | Language and move coding | Remove names and incidental identifiers |
| School grades/records | Not required by default | Do not collect |
| Diagnosis/accommodation data | Not required for the initial study | Do not collect |

## Separation

- The identity key is encrypted and stored separately from interaction and
  outcome data.
- Analysts receive study IDs, not names, email addresses, or student numbers.
- Blinded outcome scorers do not receive condition assignments.
- The condition key is released only after ratings are locked.

## Access

Access is role-based and limited to named, trained personnel approved by the
reviewing institution. Repository collaborators do not automatically receive
participant data. Public Git repositories contain code, schemas, synthetic
examples, and aggregate results only.

## Security

- Encrypt data in transit and at rest.
- Use institution-managed storage when available.
- Do not send identifiable student text to unapproved model APIs.
- Keep secrets outside source control.
- Maintain access and export logs.
- Test backups and deletion procedures before recruitment.

## Retention and destruction

The reviewing institution supplies exact dates. The final protocol must state:

- identity-key destruction date;
- raw-audio destruction date;
- identifiable-data destruction date;
- retention period for deidentified analytic data;
- whether consent permits controlled future reuse.

Deletion includes working copies, exports, and backups according to the
institution's policy. “Keep indefinitely” is not an acceptable placeholder.

## Deidentification

- Automatically flag names, contacts, schools, locations, and account IDs.
- A trained reviewer checks transcripts before analysis or sharing.
- Preserve slang and spelling only when scientifically necessary.
- Aggregate small cells and suppress potentially identifying quotations.
- Never publish raw minor-participant conversations in the repository.

## Reproducibility

Publish:

- versioned code and task IDs;
- schemas and codebooks;
- preregistration and deviations;
- synthetic fixtures;
- aggregate descriptive statistics;
- analysis scripts and environment versions.

Do not publish participant-level data unless the consent, institutional
approval, legal basis, disclosure-risk review, and repository access controls
explicitly permit it.

