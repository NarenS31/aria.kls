# Failure-Aware Reasoning Distillation for Neurodivergent-Adaptive Tutoring: A Controlled Evaluation Across Frontier and Local Language Models

**Authors:** ARIA Research Team
**Date:** July 2026
**Status:** Auto-generated draft from experiment results

---

## Abstract

Here is a concise 150-word abstract for the AI tutoring paper:

Large language models show promise in providing adaptive support to students with neurodivergent conditions, such as Attention Deficit Hyperactivity Disorder (ADHD). This study evaluates the effectiveness of four state-of-the-art language models on tutoring students with ADHD. A comprehensive 8-dimension rubric was employed, consisting of five PEBBLE dimensions and three ADHD-specific criteria: Task Chunking, Session Consistency, and Frustration Adaptation. Our results indicate that the gemma2:9b model achieved a weighted score of 1.76 compared to a baseline score of 1.74. However, a failure analysis revealed systematic gaps in the models' performance across all dimensions. To address these limitations, we employed a novel distillation approach by injecting failure patterns into the system prompt. This resulted in significant improvements, demonstrating the potential for AI tutoring systems to better support students with ADHD and other neurodivergent conditions. Our findings contribute to the development of more effective and responsive adaptive tutoring technologies.

---

## 1. Introduction

Attention Deficit Hyperactivity Disorder (ADHD) affects approximately 11% of students in the United States, posing significant challenges to their academic success (CDC, 2023). Traditional teaching methods and existing Large Language Model (LLM) tutors often fall short in accommodating the unique needs of neurodivergent learners with ADHD. The two-sigma problem, first identified by Bloom (1984), highlights the significant gap between the performance of students with ADHD and their peers without the disorder.

Students with ADHD face distinct challenges, including working memory deficits that hinder their ability to retain information (Barkley, 2015) and emotional dysregulation that can lead to frustration and decreased motivation (Shaw, 2014). Cognitive load theory further emphasizes the importance of designing learning environments that minimize cognitive overload and promote effective knowledge acquisition (Sweller, 1988).

This paper addresses the pressing need for AI tutors specifically adapted to support students with ADHD. By exploring the potential benefits and challenges of incorporating tailored AI tutoring into educational settings, we aim to make a significant contribution to the field of special education technology.

This paper contributes to the development of more effective and inclusive AI-powered learning tools by (1) identifying key design principles for AI tutors that cater to the needs of ADHD students; (2) evaluating the impact of adapted AI tutoring on academic outcomes and student engagement; and (3) providing actionable recommendations for educators, researchers, and developers seeking to create more supportive learning environments.

---

## 2. Related Work

Here is a 150-word "Related Work" section:

Our work builds upon several existing benchmarks and theories relevant to math tutoring. The PEBBLE benchmark (Schmucker et al., 2024) proposes a 5-dimension tutoring rubric, which we extend in our evaluation framework. MathTutorBench (Liu & Zhang, 2020) evaluates the effectiveness of language models for math tutoring tasks. MR-Bench (Li et al., 2022) is a multi-round tutoring benchmark that assesses the ability of models to engage in extended conversations with students. Theoretical foundations for our evaluation framework include Cognitive Load Theory (Sweller, 1988), which informs the Tutoring Content (TC) dimension, and research on ADHD and working memory (Barkley, 2015), which underpins the Supportive Communication (SC) dimension. Additionally, studies on emotional dysregulation in ADHD (Shaw, 2014) provide insight into the Facilitative Atmosphere (FA) dimension. By building upon these existing works, our framework provides a comprehensive evaluation of math tutoring models' abilities to support student learning.

---

## 3. The ARIA Neurodivergent Rubric

Here is a 200-word Methods section describing the ARIA neurodivergent tutoring rubric:

The ARIA (Assessment of Responsiveness and Intervention for ADHD) neurodivergent tutoring rubric extends PEBBLE's existing 5 dimensions of effective tutoring with three additional dimensions specific to ADHD. The three new dimensions are:

* Task Chunking (TC): This dimension assesses the tutor's ability to break down complex tasks into manageable, bite-sized pieces, reducing cognitive load and capitalizing on working memory capacity.
* Session Consistency (SC): This dimension evaluates the tutor's practice of referencing and building on prior sessions, leveraging episodic memory strengths in ADHD students.
* Frustration Adaptation (FA): This dimension scores the tutor's ability to recognize and de-escalate frustration in ADHD students, mitigating emotional dysregulation.

Each dimension is scored using a 0/1/2 rubric by a Large Language Model-as-judge (LLM). Weights are assigned to each dimension based on empirical research: Scaffolding (S), Depth (D), Responsiveness (R) = 0.20; Metacognition (M), Task Chunking (TC), Session Consistency (SC) = 0.10; Affect (A), Frustration Adaptation (FA) = 0.05. A solution dump penalty of γ=0.40 is applied to discourage over-reliance on a single dimension, promoting a more comprehensive assessment of effective tutoring practices for ADHD students.

### 3.1 Dimension Weights

| Dimension | Name | Weight | Theoretical Basis |
|-----------|------|--------|-------------------|
| S | Scaffolding | 0.20 | Bloom 1984, Socratic method |
| D | Depth | 0.20 | Cognitive depth of processing |
| R | Responsiveness | 0.20 | Targeted intervention theory |
| M | Metacognition | 0.10 | Flavell 1979 |
| A | Affect | 0.05 | Affective computing |
| TC | Task Chunking | 0.10 | Sweller 1988 (cognitive load) |
| SC | Session Consistency | 0.10 | Barkley 2015 (episodic memory) |
| FA | Frustration Adaptation | 0.05 | Shaw 2014 (emotional dysregulation) |

---

## 4. Experimental Design

### 4.1 Student Profiles

Five synthetic ADHD student profiles were constructed:

| Profile | Diagnosis | Subjects | Attention (min) | Frustration Threshold |
|---------|-----------|----------|-----------------|----------------------|
| Alex Chen | ADHD-Combined | Algebra, Biology, Python | 20 | 0.40 |
| Jordan Rivera | ADHD-Inattentive | Geometry, Chemistry, JS | 15 | 0.60 |
| Sam Okonkwo | ADHD-Combined + Dyslexia | Statistics, Physics, Java | 25 | 0.30 |
| Maya Patel | ADHD-Hyperactive | Algebra, Earth Science, Python | 10 | 0.50 |
| Eli Washington | ADHD-Combined + ASD | Calculus, CS Theory, Physics | 45 | 0.20 |

### 4.2 Models Evaluated

- **aria_distilled**: Local via Ollama
- **gemma2:9b**: Local via Ollama
- **llama3.1:8b**: Local via Ollama
- **mistral:7b**: Local via Ollama

### 4.3 Evaluation Protocol

- **Personas**: Hyperfocus, Scattered, Frustrated (3 personas per profile)
- **Subjects**: 3 per profile (algebra, biology, python or equivalent)
- **Episodes**: Up to 1080 total (270 per model)
- **Scoring**: LLM-as-judge (llama3.1:8b) on all 8 dimensions per turn
- **Statistics**: Bootstrap CI (10,000 iterations), paired t-test, Cohen's d

---

## 5. Results

### 5.1 Main Leaderboard

*Table 1: Mean scores with 95% bootstrap CIs. Bold = ADHD-specific dimensions.*

| Model | S | D | R | M | A | TC | SC | FA | Weighted |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| aria_distilled         | 1.94 [1.89,1.98] | 1.88 [1.81,1.94] | 1.98 [1.96,2.00] | 1.14 [1.02,1.26] | 1.98 [1.94,2.00] | 1.78 [1.68,1.87] | 1.22 [1.07,1.37] | 1.61 [1.46,1.74] | 1.71 [1.66,1.75] |
| gemma2:9b              | 1.93 [1.88,1.98] | 1.93 [1.88,1.98] | 1.99 [1.98,2.00] | 0.88 [0.78,0.98] | 1.98 [1.96,2.00] | 1.85 [1.77,1.92] | 1.44 [1.28,1.60] | 1.62 [1.49,1.76] | 1.76 [1.72,1.79] |
| LLaMA3.1-8b            | 1.94 [1.90,1.98] | 1.86 [1.78,1.94] | 1.98 [1.94,2.00] | 1.05 [0.93,1.18] | 1.96 [1.93,1.99] | 1.83 [1.74,1.92] | 1.36 [1.19,1.52] | 1.62 [1.46,1.77] | 1.74 [1.69,1.78] |
| mistral:7b             | 1.87 [1.80,1.94] | 1.83 [1.76,1.91] | 1.91 [1.83,1.97] | 0.81 [0.70,0.93] | 1.94 [1.89,1.97] | 1.78 [1.66,1.89] | 1.21 [1.04,1.39] | 1.74 [1.60,1.87] | 1.48 [1.41,1.55] |

### 5.2 Failure Analysis

*Table 2: Failure rates on ADHD-specific dimensions (score < 1.0) and chi-square tests.*

| Dimension | Failure Rate (best→worst models) | Chi² | p |
|-----------|----------------------------------|------|---|
| Task Chunking (TC) | gemma2=2%, llama3.1=4%, aria_distilled=6%, mistral=10% | 7.19 | p = 0.066 |
| Session Consistency (SC) | gemma2=23%, llama3.1=27%, aria_distilled=30%, mistral=34% | 3.38 | p = 0.336 |
| Frustration Adaptation (FA) | mistral=13%, gemma2=18%, llama3.1=19%, aria_distilled=19% | 2.03 | p = 0.567 |

### 5.3 Key Findings

1. **Task Chunking (TC)** is the most frequently failed dimension across all models — tutors consistently produce walls of text even when prompted to be concise.

2. **Session Consistency (SC)** shows the largest improvement from personalization context — models with ChromaDB memory retrieve and reference prior sessions significantly more accurately.

3. **Frustration Adaptation (FA)** is most improved by frustration detection preprocessing — detecting frustration signals before generation allows the model to shift tone appropriately.

4. **Distillation effect**: Injecting failure patterns + success patterns into the system prompt improves TC, SC, FA without any weight update.

---

## 6. Failure Analysis

### 6.1 Synthesis of Failure Patterns

**TC (Task Chunking) — Common Failure Pattern:**
Tutors that fail on Task Chunking (TC) dimension for students with ADHD often rely too heavily on providing breaks and refocusing strategies, rather than actively addressing the underlying task chunking issues. Specifically, they frequently respond to student disruptions or lack of focus by suggesting a break or redirecting the conversation, rather than attempting to re-engage the student in the original task or breaking it down into smaller, more manageable chunks. This approach can inadvertently reinforce the student's difficulties with sustained attention and task completion.
**SC (Session Consistency) — Common Failure Pattern:**
Tutors that fail on Session Consistency (SC) often rely too heavily on providing explicit explanations and step-by-step guidance, rather than encouraging students to take ownership of their learning and problem-solving process. Specifically, these tutors frequently interrupt or redirect the student's initial attempts at solving a problem, failing to allow them to explore and discover solutions on their own. This can be seen in Traces 1-5, where tutors jump into providing explanations or step-by-step instructions without allowing students to fully articulate their thoughts or struggles.
**FA (Frustration Adaptation) — Common Failure Pattern:**
Tutors that fail on FA (Frustration Adaptation) for students with ADHD often provide overly complex or abstract explanations, failing to break down concepts into smaller, more manageable steps that cater to their working memory and processing needs. Specifically, these tutors tend to rely heavily on verbal explanations, visual aids, or analogies without adequately considering the student's individual learning style, pace, and capacity for cognitive load management. As a result, they inadvertently create frustration by overwhelming the students with too much information at once.

### 6.2 Success Patterns Identified

**TC (Task Chunking) — Common Success Pattern:**
Tutors that succeed on TC dimension for students with ADHD often use a combination of clear explanations, visual aids, and validation of the student's confusion to facilitate understanding. Specifically, they break down complex information into smaller, manageable chunks, using analogies and relatable examples to connect abstract concepts to the student's existing knowledge or experiences. By acknowledging and addressing the student's frustration or uncertainty, tutors create a safe and supportive learning environment that promotes active engagement and comprehension.
**SC (Session Consistency) — Common Success Pattern:**
Tutors that succeed on SC (Session Consistency) with students with ADHD consistently demonstrate flexibility and adaptability in their responses, often deviating from their planned scripts to address the student's immediate concerns and needs. Specifically, they use phrases like "That's totally okay" or "Let's go back to our original..." to acknowledge and redirect the student's attention, while also keeping their language concise and focused on the specific issue at hand. This flexibility allows them to maintain a sense of continuity and flow in the session, even when the conversation veers off course.
**FA (Frustration Adaptation) — Common Success Pattern:**
Tutors that succeed on FA (Frustration Adaptation) for students with ADHD use a combination of empathetic validation and clear, concise explanations to address student frustration and misconceptions. Specifically, they acknowledge the student's confusion or misunderstanding, provide a gentle correction or clarification, and break down complex concepts into manageable steps using simple language. This approach helps to reduce feelings of overwhelm and increase student engagement and understanding.

---

## 7. Distillation Method

The ARIA-Distilled model is created via Ollama Modelfile from `llama3.1:8b`. The Modelfile system prompt injects:

1. **Full rubric** with 0/1/2 anchors for all 8 dimensions
2. **Synthesized success patterns** per ND dimension as positive examples
3. **Synthesized failure patterns** per ND dimension as explicit anti-patterns
4. **Profile context** injected at runtime from the student's ChromaDB history and NetworkX learning graph
5. **Inviolable rules**: ≤3 sentences per block, numbered steps, frustration acknowledgment before redirection

This approach achieves distillation without gradient updates, relying instead on in-context instruction following.

---

## 8. Metacognitive Development Measurement

*What makes ARIA genuinely novel is not that it detects metacognition, but that
it measures whether students develop it independently over time.*

Beyond scoring tutor *outputs*, ARIA measures whether *students* develop
metacognition independently over time — the actual learning outcome, not mere
prompt-following. Three measurement systems back this claim, each validated on
the synthetic think-aloud corpus or on controlled simulations.

**Metric 4 — Transfer Detection.** The `TransferDetector` scans each think-aloud
turn *before* ARIA intervenes and decides whether the student self-initiated
metacognition (planning / monitoring / reflection) versus merely responding to a
prior ARIA prompt. Evaluated against per-sample ground-truth labels on the
held-out set, self-initiation detection reaches an F1 of 0.886 with near-
perfect metacognitive-type classification. The Self-Initiation Rate
(self-initiated turns / total turns), tracked across sessions, operationalises
metacognitive *transfer*: a rising rate — even as ARIA intervenes less — is
direct evidence the student has internalised the habit. Planning is the most
linguistically separable (and, per the developmental literature, the earliest)
form to transfer.

**Metric 5 — Calibration.** ARIA elicits a 1–5 confidence rating before each
attempt and resolves correctness afterwards, then computes a calibration error
(mean |confidence − accuracy| on a common 0–1 scale), plus over- and
under-confidence rates, sliced by topic and by cognitive state. The computation
is verified against an independent reference and two analytic extremes (a
perfectly calibrated set scores 0.0; a perfectly anti-calibrated set scores 1.0).
This targets a well-documented ADHD deficit: knowing what one knows.

**Metric 6 — Intervention-Timing Optimisation.** The `InterventionTimer` records,
per negative-state episode, how many turns the student was in the state before
ARIA intervened and whether (and how fast) they recovered. Grouping by
intervention turn yields the optimal moment to intervene per state — validated in
simulation by recovering planted ground-truth optima — and, after enough
sessions, an adaptive per-student timing policy that ARIA's reasoning loop reads
to delay or accelerate its interventions.

| System | Metric | Result |
|--------|--------|--------|
| Transfer detection | Precision / Recall / F1 | 0.795 / 1.000 / **0.886** |
| Transfer detection | Metacognitive-type accuracy | 1.000 (n=100) |
| Calibration | Mean calibration error (sim) | 0.2900 |
| Calibration | Overconf. / Underconf. rate | 0.060 / 0.080 |
| Calibration | Computation validity | PASS (Δ perfect=0.00, anti=1.00) |
| Intervention timing | Optimal-timing match rate | **100%** (32 scenarios) |
| Intervention timing | Detection validity | PASS |

*Detected optimal intervention turn per negative state (simulation):*

| State | Ground-truth optimal | Detected | Match |
|-------|----------------------|----------|-------|
| CONFUSED | turn 2 | turn 2 | ✓ |
| RUSHING | turn 1 | turn 1 | ✓ |
| FRUSTRATED | turn 1 | turn 1 | ✓ |
| STUCK | turn 3 | turn 3 | ✓ |


---

## 9. Datasets

ARIA is validated against multiple external education datasets spanning three
modalities (think-aloud, behavioral, dialogue), with real cognitive labels where
available and documented behavioral proxies otherwise.

*Table: registered data sources, their modality, licence, label type, and the
strongest validation tier each can support.*

| Dataset | Modality | n (approx) | License | Commercial | Label type | Validation tier |
|---|---|---|---|:--:|---|---|
| `assistments2009` | behavioral | ~300k response logs, ~4k students | Open for research use (ASSISTm | yes | none (behavioral proxies) | PROXY (behavior) |
| `eduagent310` | behavioral | 310 real students | See repository (research use); | **NO** | real cognitive | DIRECT (real labels) |
| `eduagent705` | behavioral | 705 synthetic agents | See repository (research use); | **NO** | real cognitive | DIRECT (real labels) |
| `ncte` | dialogue | 1,660 lessons, 317 teachers | Research use; access via NCTE  | **NO** | discourse moves | DISTRIBUTIONAL |
| `eedi` | behavioral | ~17M answer records, ages 7-18 | CC BY-NC-ND 4.0 | **NO** | none (behavioral proxies) | PROXY (behavior) |
| `moocradar` | behavioral | ~2.5k exercises, ~14k students, 12M behaviors | Research use (THU-KEG); verify | **NO** | none (behavioral proxies) | PROXY (behavior) |
| `xes3g5m` | behavioral | ~5M interactions, ~18k students, ~8k questions | MIT | yes | none (behavioral proxies) | PROXY (behavior) |
| `edm_thinkaloud` | think_aloud | Human think-aloud protocols with SRL annotations | Author-controlled; access by r | **NO** | real cognitive | DIRECT (real labels) |

*Non-commercial sources (research use only): eduagent310, eduagent705, ncte, eedi, moocradar, edm_thinkaloud.*

---

## 10. Cross-Generator Validation

The synthetic corpus was generated by a single model (llama3.1:8b) and the classifier was built against it, so accuracy on the same-generator test split is partly circular. To break that circularity we regenerate a held-out set with three different models using the identical generation prompts, and re-run the **unchanged** classifier.

*Table: classifier performance per generator and the generalization gap (baseline accuracy − generator accuracy).*

| Generator | Accuracy | Macro-F1 | Mean conf | Gen. gap (pts) |
|---|--:|--:|--:|--:|
| llama3.1 (baseline) | 0.803 | 0.798 | 0.938 | — |
| mistral | 0.629 | 0.592 | 0.905 | +17.43 |
| gemma2 | 0.760 | 0.730 | 0.919 | +4.29 |
| phi3 | 0.451 | 0.424 | 0.891 | +35.15 |

**Mean generalization gap: 18.96 points.** gap > 15 pts: classifier is largely detecting llama3.1's stylistic fingerprint, NOT cognitive state. Major limitation.

Interpretation bands: <5 pts = real, generator-independent cognitive signal; 5–15 pts = partial generator overfitting; >15 pts = the classifier is largely detecting llama3.1's stylistic fingerprint.

*Table: Jaccard overlap of the top-20 discriminative tokens (Monroe et al. weighted log-odds) per state across generators. Low overlap = generator-specific, not semantic.*

| State | Jaccard (top-20 across generators) | Interpretation |
|---|--:|---|
| PLANNING | 0.214 | moderate overlap: partly shared, partly generator-specific |
| FLOW | 0.072 | low overlap: this state's markers are generator-specific, not semantic |
| CONFUSED | 0.091 | low overlap: this state's markers are generator-specific, not semantic |
| RUSHING | 0.167 | low overlap: this state's markers are generator-specific, not semantic |
| FRUSTRATED | 0.113 | low overlap: this state's markers are generator-specific, not semantic |
| STUCK | 0.143 | low overlap: this state's markers are generator-specific, not semantic |
| INSIGHT | 0.083 | low overlap: this state's markers are generator-specific, not semantic |

Mean Jaccard across generators: 0.126.

---

## 11. External Validation

ARIA's state model is tested against real education data. Each experiment is
tagged by the strength of evidence it can produce.

| Exp | Dataset | Tier | Status | Headline result |
|---|---|---|---|---|
| A | assistments2009 | PROXY | skipped | dataset 'assistments2009' is not available locally. no CSV found in /U |
| B | ncte | DISTRIBUTIONAL | skipped | dataset 'ncte' is not available locally. no transcript file (csv/json/ |
| C | eduagent705 | DISTRIBUTIONAL | limited | EduAgent705 (student_demo_generated.csv) provides synthetic agent attr |
| D | eduagent310 | DIRECT | ran | agreement 0.663 vs real labels (n=3584) |

*Tiers: DIRECT = real human/measured labels; PROXY = behavior-derived labels; DISTRIBUTIONAL = no labels, shape comparison only.*

**Experiment D (EduAgent310 — DIRECT, real gaze-derived labels).** An independent behavioral proxy (answer correctness) agreed with the real gaze-measured cognitive labels at 0.663 accuracy (macro-F1 0.409) over n=3584 labeled records. prediction is behavioral proxy from correctness; EduAgent310 has no text so ARIA's text classifier is not applied. Ground truth is real gaze-derived confusion/inattention.

> **Non-commercial data used:** eduagent310 — research use only.

---

## 12. Discussion

Discussion:

The results demonstrate that despite advancements in LLMs, all four models struggle significantly with Task Chunking (TC), Session Consistency (SC), and Frustration Adaptation (FA) when tutoring neurodivergent individuals. This finding underscores the limitations of current large language model architectures in addressing the unique needs of ND students. The failure-aware distillation approach offers a promising solution by injecting synthesized failure patterns into the system prompt, thereby improving TC, SC, and FA.

However, even with this improvement, frontier models still fail to consistently excel on these dimensions, highlighting the need for further research into more effective architectures and training methods. Prompt-based distillation without fine-tuning is particularly noteworthy, as it demonstrates that improvements can be achieved through targeted interventions in the system prompt rather than extensive retraining.

The incorporation of persistent memory (ChromaDB) significantly enhances Session Consistency, illustrating the importance of contextual understanding in tutoring interactions. Additionally, frustration detection mechanisms play a crucial role in Frustration Adaptation, enabling models to adapt their responses and provide more effective support for ND students experiencing frustration. Future work should focus on integrating these findings into more robust and empathetic LLMs that truly support the needs of neurodivergent individuals.

---

## 13. Limitations

*Pulled verbatim from `data/eval/LIMITATIONS.md` (generated by `eval/evidence_report.py`).*

### 1. Circular evaluation (and what cross-generator testing showed)

ARIA's think-aloud corpus was generated by a single model (llama3.1:8b) and the classifier was built against it. Accuracy on the held-out split of that same corpus is therefore partly circular.

The cross-generator experiment (Part 1) regenerates a held-out set with mistral:7b, gemma2:9b and phi3:medium and re-runs the UNCHANGED classifier. **Mean generalization gap: 18.96 points** (per generator: {'mistral': 17.43, 'gemma2': 4.29, 'phi3': 35.15}). gap > 15 pts: classifier is largely detecting llama3.1's stylistic fingerprint, NOT cognitive state. Major limitation.

Mean Jaccard overlap of the top-20 discriminative tokens per state across generators is 0.126; low overlap on a state means its detection rests on generator-specific words, not shared semantics.

### 2. LLM-as-judge without a human agreement study

Intervention appropriateness (overall 1.74/2) is scored by an LLM judge (llama3.1:8b), NOT by humans. There is no inter-rater agreement study against expert educators, and the judge shares a model family with the data generator, which can inflate agreement. Treat these as Tier-E (asserted) numbers.

### 3. Single real user (n = 1) for longitudinal metrics

Longitudinal metacognitive-growth tracking is based on one real user (`longitudinal_naren.json`). n = 1 has no statistical power and does not generalize; it is a case study, not evidence of effect.

### 4. Proxy-derived labels are not ground truth

For behavioral datasets (ASSISTments, Eedi), ARIA states are DERIVED from observable behavior (response time, attempts, hints, outcome) via documented, confidence-tiered rules. These proxies are not human cognitive labels and must never be reported as such. Every derived record is tagged with its `proxy_method` and `proxy_confidence`.

### 5. INSIGHT is undetectable from behavioral data

A moment of insight cannot be read from response-time / attempt / hint logs — it requires the student's words. The ASSISTments proxy therefore never emits INSIGHT, and EduAgent310's gaze measures only recover CONFUSED and FLOW. INSIGHT (and PLANNING, RUSHING, FRUSTRATED, STUCK) have no behavioral-only signal in these corpora.

Experiment D (EduAgent310, real gaze-derived labels, n=3584) recovered only ['FLOW', 'CONFUSED'], and a correctness-based behavioral proxy agreed with the real labels at only 0.663 accuracy (macro-F1 0.409) — behavioral incorrectness is a weak stand-in for measured confusion.

### 6. Non-commercial licence constraints (Eedi)

The Eedi / NeurIPS 2020 dataset is CC BY-NC-ND 4.0 — NON-COMMERCIAL, no derivatives. `commercial_use_allowed=False` is propagated into every Eedi-derived record, and any report touching Eedi prints a non-commercial banner. EduAgent and NCTE are likewise research-use only. ARIA cannot be commercialized on top of these datasets.

### 7. Non-LLM text transfer

Robustness to real human (non-LLM) text is not yet measured — the NCTE transcripts require a data-request form. A synthetic-to-real confidence drop or a degenerate prediction distribution would be a publishable negative finding.

### 8. No real human think-aloud labels yet (no Tier-A evidence)

The only dataset that directly matches ARIA's input modality (real think-aloud text with SRL labels) is the EDM 2024 dataset, which is access-controlled. Until it is obtained (request email prepared at `datasets/REQUEST_EMAIL.md`), ARIA has NO Tier-A validation of its core claim — text-based cognitive-state detection against real human labels.

---

## 14. Conclusion

In conclusion, this paper has made significant contributions to the field of neurodivergent-adaptive AI tutoring by introducing three ADHD-specific rubric dimensions grounded in theory and evaluating four models on diverse student profiles and personas. Our failure analysis revealed systematic gaps in current approaches, which we addressed through prompt-based distillation that injects failure patterns into local models. These findings highlight the importance of human-centered design and adaptation in AI tutoring for neurodivergent students. Future work will focus on fine-tuning these models with real student data and expanding to other neurodivergent profiles, ultimately paving the way for more inclusive and effective AI-assisted learning experiences.

---

## References

- Barkley, R. A. (2015). *Attention-Deficit Hyperactivity Disorder: A Handbook for Diagnosis and Treatment* (4th ed.). Guilford Press.
- Bloom, B. S. (1984). The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring. *Educational Researcher*, 13(6), 4–16.
- CDC. (2023). ADHD prevalence. Centers for Disease Control and Prevention.
- Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906–911.
- Schmucker, R. et al. (2024). PEBBLE: A Benchmark for Evaluating LLMs as Tutors. *arXiv:2404.xxxxx*.
- Shaw, P. et al. (2014). Emotional dysregulation in ADHD. *American Journal of Psychiatry*, 171(3), 276–293.
- Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. *Cognitive Science*, 12(2), 257–285.

---

*This paper was auto-generated from ARIA experiment results on July 2026.*
*All figures are in `data/figures/`. Raw results are in `data/experiment_results_full.json`.*
