# ARIA-Pace — Literature Review (Stage 1)

Branch: `aria-pace` (from `summit-demo`). Written 2026-10-15. No code in this stage.

## Verification labels

| Label | Meaning |
|---|---|
| **VERIFIED** | Bibliographic record confirmed AND the abstract or text was opened and read for the claims made here. |
| **VERIFIED (metadata only)** | Existence, authors, venue, volume, pages, year confirmed through the Crossref record for the DOI. The abstract/full text was **not** opened (paywall, 403, or publisher-elided abstract). Content claims below are limited to what the title and secondary sources support, and are marked. |
| **UNVERIFIED** | Not confirmed against any source. |

Search limits: the prior-work searches were a handful of general web searches (US-only web search tool), not a systematic review and not Google Scholar / ACL Anthology / EDM proceedings indexes. "Not found" below means "not found by these searches", not "does not exist".

---

## 1. Foundational papers requested

### 1.1 Sweller — cognitive load
- Sweller, J. (1988). Cognitive Load During Problem Solving: Effects on Learning. *Cognitive Science* 12(2), 257–285. doi:10.1207/s15516709cog1202_4
- **VERIFIED (metadata only)** — Crossref returned title, author, journal, vol 12 issue 2, pp. 257–285, 1988. Wiley page returned 403; abstract not opened.
- Role here: motivates why extra interruptions can cost the learner working memory. The repo's own gloss is "working-memory load theory and the rationale for reducing a task to smaller steps" (`eval/data/research/citations.json:29`). I have not independently confirmed that gloss against the abstract.

### 1.2 Expertise-reversal effect
- Kalyuga, S., Ayres, P., Chandler, P., & Sweller, J. (2003). The Expertise Reversal Effect. *Educational Psychologist* 38(1), 23–31. doi:10.1207/s15326985ep3801_4
- **VERIFIED (metadata only)** — Crossref confirmed. Taylor & Francis page returned 403.
- Role here: the standard citation for the claim that guidance helpful to novices can be unhelpful or harmful to more expert learners. That is the reason a single fixed help policy is a weak baseline, and it motivates (but does not by itself justify) per-profile cost settings. Content description is background knowledge, **not verified in this session**.

### 1.3 Koedinger & Aleven 2007 — Assistance Dilemma
- Koedinger, K. R., & Aleven, V. (2007). Exploring the Assistance Dilemma in Experiments with Cognitive Tutors. *Educational Psychology Review* 19(3), 239–264. doi:10.1007/s10648-007-9049-0
- **VERIFIED** — Crossref record confirmed; abstract read from the authors' **submitted manuscript** (CMU PSLC copy, header "Manuscript submitted for publication, Feb 25, 2007"), not the published version. Wording may differ in print.
- Verified content: the paper reviews Cognitive Tutor experiments and asks "How should learning environments balance information or assistance giving and withholding to achieve optimal student learning?", calls this the "assistance dilemma", and calls for "specific conditions and parameters that indicate when and to what extent to use information giving versus information withholding."
- Role here: ARIA-Pace is one concrete, parameterised answer to *when* to give help. It does not resolve the dilemma.

### 1.4 Corbett & Anderson 1995 — Bayesian Knowledge Tracing
- Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction* 4(4), 253–278. doi:10.1007/BF01099821
- **VERIFIED (metadata only)** — Crossref confirmed. Springer page redirected to a login handshake; abstract not opened.
- Secondary-source support only: a search result describes the ACT Programming Tutor keeping a running probability that a student has learned each rule, updated at each opportunity. The four-parameter (prior / learn / guess / slip) formulation is standard background knowledge, **not verified here**.
- Role here: Stage 3 baseline.

### 1.5 Pavlik, Cen & Koedinger — PFA
- Pavlik, P. I., Cen, H., & Koedinger, K. R. (2009). Performance Factors Analysis – A New Alternative to Knowledge Tracing. In *Artificial Intelligence in Education (AIED 2009)*, Frontiers in AI and Applications vol. 200, pp. 531–538. doi:10.3233/978-1-60750-028-5-531
- **VERIFIED** — Crossref record and the Memphis repository abstract page were both read. Abstract: modifies an existing educational data-mining model (a Learning Factors Analysis variant) so it supports adaptive practice selection, and compares PFA to LFA variants and to standard knowledge tracing.
- **Equation verified against the paper PDF** (pact.cs.cmu.edu copy, opened and text-extracted): Equation 3 is m(i, j∈KCs, s, f) = Σ_j (β_j + γ_j·s_{i,j} + ρ_j·f_{i,j}), where s and f are the student's prior successes and failures on knowledge component j, γ and ρ scale their effects, and Equation 2 (the logistic link) converts the logit m to a probability. The paper notes the model can sum these terms for items needing several KCs. The Stage 3 baseline may therefore be called PFA and cite this equation; any deviation (regularisation, per-skill pooling, feature clipping) must be stated as a deviation.

### 1.6 Beck & Gong 2013 — wheel-spinning
- Beck, J. E., & Gong, Y. (2013). Wheel-Spinning: Students Who Fail to Master a Skill. *AIED 2013*, LNCS 7926, pp. 431–440. doi:10.1007/978-3-642-39112-5_44
- **VERIFIED (metadata only)** — Crossref confirmed. Springer paywalled and Semantic Scholar elides the abstract. **The primary text was not opened.**
- **The definition is reported inconsistently by the secondary sources I did open:**
  - Wang, Kai & Baker (AIED/Springer; PDF opened): Beck & Gong (2013) defined wheel-spinning as "not achieving mastery even after attempting 10 or more problems within a problem set"; a different work ("[10]" in their reference list) used "attempting more than 10 problems but failing to achieve three consecutive correct responses in a row or demonstrate later retention."
  - Kai et al., JEDM (PDF opened): Beck & Gong identified wheel-spinning as "10 or more skill opportunities without mastering"; later says "any student who completes more than 10 problems without getting 3 correct in a row and achieving mastery is wheel-spinning." It reports ~38% of students not mastering after 10 opportunities in Beck & Gong, and that under this definition 12.3% of 287,093 student-problem-set pairs are wheel-spinning in their data.
  - Web search snippets (unopened pages) give a third variant (">8 attempts with mean correctness below 0.35"), attributed to Beck & Gong in a source I did not open. Treat as **UNVERIFIED**.
- Consequence: the "exactly per Beck & Gong" label requested in Stage 3 cannot be certified from the primary text in this session. The 3-correct-in-a-row mastery criterion and the 10-opportunity horizon are supported by two opened secondary sources; whether Beck & Gong themselves state the threshold as ">=10" or ">10" is unresolved. See Section 5 (decisions).

### 1.7 Selective prediction / reject option
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory* 16(1), 41–46. doi:10.1109/TIT.1970.1054406
  - **VERIFIED (metadata only)** — Crossref confirmed. IEEE page returned no abstract text. That it introduces the error/reject trade-off is consistent with the title and with a secondary source that cites "Chow's rule" (reject when expected cost of deciding is high); the derivation is **not verified here**.
- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. arXiv:1705.08500 (v2, 1 Jun 2017).
  - **VERIFIED** — arXiv abstract page read. Builds a selective classifier from a trained network; user sets a target risk; classifier rejects inputs so error on accepted inputs stays at that level with high probability; trade-off is coverage. Example reported: 2% top-5 ImageNet error guaranteed with probability 99.9% at ~60% coverage. The arXiv page lists no conference; I am not asserting a venue.
- Role here: ARIA-Pace's "intervene only when error and risk >= tau" is a risk/coverage-style abstention rule. This borrows the framing, not the guarantees. These are classification results; no claim transfers automatically to tutoring.

### 1.8 Pane et al. 2014 — Cognitive Tutor RAND study
- Pane, J. F., Griffin, B. A., McCaffrey, D. F., & Karam, R. (2014). Effectiveness of Cognitive Tutor Algebra I at Scale. *Educational Evaluation and Policy Analysis* 36(2), 127–144. doi:10.3102/0162373713507480
- **VERIFIED (abstract via ERIC EJ1024233; journal page returned 403)**. Cluster randomized trial; schools matched in pairs and randomly assigned to keep their algebra curriculum or adopt Cognitive Tutor Algebra I for two years. Year 1: no effect on the algebra proficiency posttest. Year 2: positive effects; statistically significant in high schools, not significant in middle schools; estimated magnitude described as raising the median student by about eight percentile points. Other search results state 147 schools in seven states and 17,000+ students (EdSurge/IES); counts vary by source (Carnegie Learning cites 19,000+), so I use none of them as a fact.
- A standardized effect size (e.g. 0.20 SD) was **not** found in any source I opened. Do not quote one.
- Role here: evidence that a Cognitive-Tutor-style curriculum can improve outcomes at scale in a randomized trial. It says nothing about intervention *timing*, and offline replay on logs cannot reproduce learning gains.

---

## 2. Citations in `eval/metacognition/jitai.py` `RULE_CITATIONS`

`RULE_CITATIONS` is defined at [jitai.py:60-71](../../eval/metacognition/jitai.py#L60-L71); `TIEBREAK_CITATIONS = ["le_tallec_2025"]` at [jitai.py:73](../../eval/metacognition/jitai.py#L73). Full records live in `eval/data/research/citations.json` (file status `"metadata_verified"`, `verified_on` 2026-07-29, lines 3 and 5). I re-checked each independently.

| Key (citations.json line) | Used by rules | Status | Notes |
|---|---|---|---|
| `zimmerman_2002` (:7) | insight_transfer, no_interrupt_positive, no_action | **VERIFIED (metadata only)** | Zimmerman, B. J. (2002). Becoming a Self-Regulated Learner: An Overview. *Theory Into Practice* 41(2), 64–70. Crossref confirmed. |
| `hattie_timperley_2007` (:83) | insight_transfer, initial_negative_support | **VERIFIED (metadata only)** | Hattie & Timperley (2007). The Power of Feedback. *Rev. Educ. Res.* 77(1), 81–112. Crossref confirmed. |
| `sweller_1988` (:29) | frustration_reset, repeated_confusion, rushing_slowdown, high_struggle_topic | **VERIFIED (metadata only)** | See 1.1. |
| `black_wiliam_1998` (:72) | high_struggle_topic | **VERIFIED (metadata only)** | Black & Wiliam (1998). Assessment and Classroom Learning. *Assess. Educ.* 5(1), 7–74. Crossref confirmed. |
| `le_tallec_2025` | `TIEBREAK_CITATIONS` ([jitai.py:73](../../eval/metacognition/jitai.py#L73)); docstring at [jitai.py:303](../../eval/metacognition/jitai.py#L303) | **UNVERIFIED — not found** | There is **no entry** for this key in `citations.json` (it lists 28 keys; this is not one). `grep` over the repo finds only the two jitai.py lines. Two web searches (including an extended one) for the paper found nothing. The docstring attributes a specific claim to it ("when the situation model is unreliable, a directive prompt costs more if it lands on a misread affective state…"). I cannot confirm the paper exists or says this. |

Scope note: two of the seven keys in `RULE_CITATIONS` are clinical (ADHD) sources. They are **excluded from ARIA-Pace** and are neither cited nor summarised in these documents; ARIA-Pace profiles are cost settings, not clinical categories. `le_tallec_2025` is also **not cited anywhere in ARIA-Pace**; its two usage sites in the demo path are listed above for a separate decision on `summit-demo`.

---

## 3. Prior work searched

### 3.1 Wheel-spinning prediction (most relevant overlap)
- **Wang, Kai & Baker. Early Detection of Wheel-Spinning in ASSISTments** (AIED 2020 volume; Springer chapter 10.1007/978-3-030-52237-7_46 per search result). **VERIFIED** (PDF opened). Builds detectors distinguishing non-persistence, productive persistence and wheel-spinning from the first n practice opportunities. Best model uses the first nine; wheel-spinning vs productive persistence is separable from the first three; wheel-spinning vs non-persistence from the first five. Key features: hint requests, correctness of prior opportunities, amount of practice and time. **This directly overlaps the Stage 3 "predict wheel-spinning after k = 3, 5, 8 attempts" task. That part is not novel.** I did not open the AUC tables; do not quote numbers from it.
- **Kai et al., Decision Tree Modeling of Wheel-Spinning and Productive Persistence in Skill Builders.** *JEDM* (article 210). **VERIFIED** (PDF opened, definitions only). ASSISTments Skill Builder; mastery = three correct in a row.
- **Mu, Jetten & Brunskill (2020). Towards Suggesting Actionable Interventions for Wheel-Spinning Students.** EDM 2020, pp. 183–193. **VERIFIED** (PDF opened; abstract + intro). Predicts wheel-spinning and uses Shapley values to choose *which* intervention. States there has been "much work for modelling and predicting wheel-spinning, but little work in developing interventions." Decides *what*, not *whether to stay quiet*.

### 3.2 Tutor intervention timing / proactive help
- **Maniktala, Cody, Isvik, Lytle, Chi & Barnes. Extending the Hint Factory for the assistance dilemma: A novel, data-driven HelpNeed Predictor for proactive problem-solving help.** arXiv:2010.04124 (8 Oct 2020). **VERIFIED (abstract page only).** A binary productive/unproductive step classifier triggers proactive hints at the start of each step in a logic tutor; controlled study reports lower help avoidance, higher help appropriateness, better posttest for the Adaptive group. The abstract page does not describe abstention, cost weights, or tunable thresholds; the full text was not read, so I cannot rule them out.
- **Razzaq & Heffernan (2010), Hints: Is it Better to Give or Wait to be Asked?** ITS 2010. **UNVERIFIED** — seen only as a search-result snippet (claimed: on-demand beat proactive hints, largest gap for heavy help requesters). Not opened.
- **An et al., Revisiting the Hint Button** (LAK 2026, doi:10.1145/3785022.3785040). **UNVERIFIED** — snippet only (claimed: unproductive hint use is negatively associated with learning).
- **Learning to Give Useful Hints: Assistance Action Evaluation and Policy Improvements** (Springer 10.1007/978-3-031-42682-7_26). **UNVERIFIED** — snippet only (claimed: bandit policies over assistance actions, offline logs, A/B test).
- **Examining Variation in How Guided AI Tutors Resolve Student Impasses.** arXiv:2609.38346 (29 Sep 2026; Ahtisham et al.). **VERIFIED (abstract page).** 1,260 real LLM chemistry-tutor sessions; impasse depth/type proposed as signals for graduated assistance. No abstention mechanism, no cost model, no learner profiles on the abstract page.

### 3.3 Abstention in tutoring
- **Pal, Ji & Srihari. DICE: Decoupling Capability from Intervention Necessity in LLM Tutoring.** arXiv:2610.04825 (4 Oct 2026; stated on the page as accepted to NeurIPS 2026). **VERIFIED (arXiv HTML text read, except the middle of Appendix C); closest conceptual prior work found.** Separates *whether to intervene* from *what to say*; calibrates an action policy with a rollout-based "Intervention Value" against not intervening; finds many prescribed interventions have little or no marginal benefit; reports near-zero over-intervention in **simulated** sessions on a new benchmark. Per the text: not framed as selective prediction, no Pareto frontier, no learner profiles, no real student logs, no wheel-spinning. It does use a validation-tuned cost lambda and threshold tau (see the comparison above).
- No tutoring paper framing the intervene/stay-quiet decision as selective prediction with a reject option was found by my searches (one search result reports the Chow-rule framing for time-series forecasting, arXiv 2606.23448, unrelated to tutoring; **UNVERIFIED**, snippet only).

### 3.4 Learner-configurable help frequency
- No paper was found in which the learner (or a teacher/parent) sets an explicit help-frequency or interruption-cost parameter that moves an operating point on a risk-based policy. Related but different: student control of hints is studied (e.g. the Aleven & Koedinger 2000 "Do students know when they need help?" line, known to me only as a reference-list title in a search result, **UNVERIFIED**), and hint-depth adaptation to proficiency (snippet only). Unverified encyclopedia-style claims about learner-controlled hints are not used.

### 3.5 Interruption cost, outside tutoring
- A proactive-intervention paper (Optimizing Interventions via Offline Policy Evaluation, AAAI; PDF returned as unreadable binary and **not opened**) was surfaced in a search; the description I have is from a search summary only and is **UNVERIFIED**. Not relied on.

---

## 4. Is "profile-conditioned abstention frontier" already published?

**Not found as a combined, published idea, but nearly all its parts are.**

| Component | Prior art found | Status |
|---|---|---|
| Predict wheel-spinning early from the first few attempts | Wang/Kai/Baker (3, 5, 7, 9 opportunities, ASSISTments); Kai et al.; Mu et al. | Published. **Not novel.** |
| Use a risk prediction to trigger proactive help | HelpNeed Predictor (Maniktala et al.), Mu et al. | Published. **Not novel.** |
| Decide *whether* to intervene as a separate decision; most interventions add little value | DICE (Pal et al., Oct 2026) | Published recently, in simulation with LLM tutors. Overlaps the motivation. |
| Selective prediction / reject option with a cost trade-off | Chow 1970; Geifman & El-Yaniv 2017 | Published, in classification. Not applied to tutoring in anything I found. |
| Explicit user/learner "profiles" as cost weights over interruption vs delayed help | None found | Not found. |
| Interruptions-vs-help-delay Pareto frontier evaluated offline on real student logs, one curve per profile | None found | Not found. |

Statement: I did **not** find a paper that publishes a profile-conditioned abstention frontier for tutor interventions on real student logs. I cannot say none exists: the search was limited (see top), DICE is days old (4 Oct 2026) and I read only its abstract, and the HelpNeed, Mu et al., and AAAI offline-evaluation papers were not read in full.

### How ARIA-Pace differs from DICE (full paper read, arXiv:2610.04825)

Read: main text and Appendix C.3 and H via the arXiv HTML version; the extractor truncated the middle of Appendix C, so details of K and the margin epsilon are **not** known to me.

What DICE does (verified from the text): a policy picks one of six pedagogical actions, one of which (a_0) is "do not intervene". Intervention Value is IV(x,a) = mu(x,a) - mu(x,a_0), estimated by simulating K continuations from a Qwen-2.5-1.5B-Instruct student and labelling them with an LLM judge. Training weights each action by max(IV, 0) with a KL penalty to a supervised policy. Inference uses either a cost-penalised argmax (lambda, 0.25 in the tables) or a probability threshold tau on total intervention probability (0.8 in the tables), both tuned on validation. The benchmark, DICE-Bench, is derived from the synthetic SocraTeach corpus (GSM8K/MAWPS problems; 10,273 matched easy/hard triplets; 20,477 turn-level decision points labelled by GPT-5). Metrics: action accuracy and macro-F1, success rate, turns, over-intervention (interrupting a student who is already correct), false acceptance, unsafe final-answer acceptance.

Where ARIA-Pace differs, and where it does not:

| | DICE | ARIA-Pace |
|---|---|---|
| Data | Synthetic corpus, LLM-generated variants, simulated students | Real logged student interactions from a public tutoring dataset |
| Decision | Which pedagogical action, including "none", for an LLM tutor's next utterance | Whether to interrupt at all, given a verified error and a wheel-spinning risk score |
| Signal | Counterfactual value from rollouts with a simulator and an LLM judge | Predicted probability of wheel-spinning from the student's own response history |
| Threshold | tau / lambda tuned on one validation composite | tau chosen on validation students separately for each of three profiles |
| Learner profiles | None; only simulator size (0.5B/1.5B/3B) is varied | Three explicit cost-weight profiles (quiet / balanced / supportive) |
| Trade-off reporting | Sweeps and over-intervention vs accuracy/turns plots; no frontier or multi-objective analysis reported | Interruptions vs help-delay curve per profile, with bootstrap CIs over students |
| Outcome | Simulated tutoring success | Offline replay only: interruptions, missed help, lead time; no learning outcome |

What is **shared**, and must not be presented as new: treating "do nothing" as a first-class decision; a validation-tuned cost or threshold that suppresses unnecessary interventions; and reporting over-intervention. DICE does not use real student logs, wheel-spinning, or selective-prediction framing, per its text. ARIA-Pace is much narrower: a tabular risk model on one public dataset, with no generative tutor. DICE's limitation section (Appendix H) says its results show improved simulated tutoring and not human learning gains; ARIA-Pace's offline replay has the same limit, and additionally cannot test what a student would have done had they been interrupted.

### Narrowed claim (what ARIA-Pace may defensibly say)
Claim **only**:
> We evaluate a simple abstention rule (intervene only when a verified error coincides with predicted wheel-spinning risk >= tau) on public student logs, and show the interruption-versus-help-delay trade-off it produces under three explicit cost weightings (quiet / balanced / supportive), with student-level held-out evaluation and bootstrap CIs.

Do **not** claim: novelty of early wheel-spinning prediction; novelty of risk-triggered help; that abstention improves learning; that profiles map to ADHD, math anxiety, or any clinical or demographic group; that results transfer to live students.

---

## 5. Decisions recorded (from the owner, 2026-10-15)

1. **Wheel-spinning label.** Label = fewer than 3 consecutive correct responses within the first 10 opportunities on a skill, written as "following Wang et al. / Kai et al., attributed to Beck & Gong". Both ">=10" and ">10" opportunities are run as a sensitivity check. (Beck & Gong's primary text was never opened; do not write "exactly per Beck & Gong".)
2. **PFA.** Verified against the Pavlik et al. PDF (see 1.5), so the baseline is PFA as published, with any deviation stated.
3. **`le_tallec_2025`.** Not cited in ARIA-Pace. `eval/` is untouched. Usage sites in the demo path: [jitai.py:73](../../eval/metacognition/jitai.py#L73) (`TIEBREAK_CITATIONS`) and [jitai.py:303](../../eval/metacognition/jitai.py#L303) (docstring of `_affect_tiebreak`). The key has no entry in `eval/data/research/citations.json`.
4. **ADHD-related RULE_CITATIONS keys** are not cited in ARIA-Pace documents.

## 6. Source list (opened)

- Crossref records (api.crossref.org) for all DOIs above.
- Koedinger & Aleven manuscript PDF: pslcdatashop.web.cmu.edu/KDDCup/FAQ/Koedinger-Aleven-EPR-07.pdf
- Pavlik et al. abstract: digitalcommons.memphis.edu/facpubs/8350
- Pane et al. abstract: eric.ed.gov/?id=EJ1024233
- Geifman & El-Yaniv: arxiv.org/abs/1705.08500
- Wang, Kai & Baker PDF: learninganalytics.upenn.edu/ryanbaker/paper_45a.pdf
- Kai et al. JEDM PDF: jedm.educationaldatamining.org/index.php/JEDM/article/download/210/88
- Mu et al. PDF: educationaldatamining.org/files/conferences/EDM2020/papers/paper_201.pdf
- Maniktala et al.: arxiv.org/abs/2010.04124
- Pavlik et al. PDF: pact.cs.cmu.edu/koedinger/pubs/AIED 2009 final Pavlik Cen Keodinger corrected.pdf
- DICE full text: arxiv.org/html/2610.04825
- DICE: arxiv.org/abs/2610.04825
- Ahtisham et al.: arxiv.org/abs/2609.38346

---

## Addendum (Stage 6, 2026-10-16): knowledge-tracing models used as candidates

- **Piech, C., Spencer, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J. (2015). Deep Knowledge Tracing. arXiv:1506.05908** (submitted 19 June 2015; the arXiv page names no venue). **VERIFIED**
  (arXiv abstract page opened). Paraphrase: recurrent neural networks model how a student's knowledge evolves and are reported to predict better than earlier methods on several datasets without hand-encoded domain knowledge.
- **Pandey, S., & Karypis, G. (2019). A Self-Attentive model for Knowledge Tracing. arXiv:1907.06837** (submitted 16 July 2019; the arXiv listing names the International Conference on Educational Data Mining). **VERIFIED**
  (arXiv abstract page opened). Paraphrase: SAKT uses self-attention to weight the most relevant past activities, and is reported to handle sparse data better than RNN-based trackers, with about 4.4% average AUC improvement across datasets.
  Venue and exact published form beyond the arXiv page: **UNVERIFIED**. The full papers were not read.
- **How they are used here, and what that does not claim.** `aria_pace/dkt.py` and `aria_pace/sakt.py` are small, simplified versions written for this study (every step of a sequence uses the sequence's single KC; one LSTM layer or one attention block; first 100
  opportunities; architectures fixed in SPEC Amendments 3 and 3a). They are **not reproductions** of the published models, and a result here says nothing about the published models' performance on their own datasets.
