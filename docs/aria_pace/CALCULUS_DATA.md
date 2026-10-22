# Calculus-type student logs: what is publicly available (draft research note)

Date checked: 2026-10-22. Method: fetched each source page (WebFetch) or ran a web search. Nothing was downloaded and no account was created. Every item says whether a page was actually fetched; "not found" means it could not be verified, not that it does not exist.

## Bottom line

- **Public student-level step logs for calculus (differentiation or integration): not found.** No DataShop dataset, KDD Cup set, or other open log set that was checked is described as calculus. The DataShop dataset catalog itself could not be browsed (`https://pslcdatashop.web.cmu.edu/DatasetList` returned HTTP 404 to the fetcher), so a calculus dataset inside DataShop is **not ruled out**.
- **The 2009 ASSISTments skill-builder math logs used here are not calculus** (grade level not stated: `aria_pace/SPEC.md:94`). This note does not change that: the risk model has no calculus calibration.

## Sources checked

| Source | What was verified (by fetching) | Access terms, size | Calculus? |
|---|---|---|---|
| **PSLC DataShop** (`pslcdatashop.web.cmu.edu`; FAQ mirror at `datashop.phzh.ch/about/faq.html`) | A repository of student-software interaction data with exports as tab-delimited files. The FAQ names PSLC courses in Algebra, Chemistry, Chinese, English, French, Geometry and Physics, and notes middle-school math data from ASSISTments. | "DataShop access is free." The FAQ says "You do not need permission to view or use public data sets"; private sets need a "Request Access" to the project's PI and data provider; publications must acknowledge the data source; no data use agreement is described. The CMU page (`cmu.edu/datalab/tools/datashop.html`) says you "need to create an account" to view public datasets. **The two pages differ on whether an account is needed for public data; not resolved.** Size per the CMU page: 1,466 datasets, 358,000 students, about 250.7 million student actions. | Not found (no calculus category on the FAQ; catalog not browsable by the fetcher) |
| **KDD Cup 2010** (`pslcdatashop.web.cmu.edu/KDDCup/`) | Five data sets (3 development, 2 challenge), each split into training and test; test labels withheld for the challenge sets; an account is the first step to "join the challenge". | Terms are on a rules page that was not retrieved. | The fetched page does not name the subject; a search result describes the sets as Carnegie Learning Cognitive Tutor algebra-type data. **Not verified.** |
| **DataShop equation-solving sets #5549 and #5604, Mathtutor #5360** | **Only seen in a search-result snippet; the dataset pages were not fetched.** The snippet describes multi-step equation solving, grades 6-8, over 10,000 step-level rows. | Not verified. | Algebra, not calculus. |
| **Open Learning Initiative (OLI)** | A search found only old (2011-2012) documentation of OLI data in DataShop and a statement that OLI contributes college online-course data (statistics is the example given). | Not verified for current terms. | Not found. |
| **OpenStax Tutor data on SafeInsights** (`kb.safeinsights.org`) | Appeared in search results only; page not fetched. | Not verified (access model unknown). | Not verified. |
| **EdNet, Junyi, other ASSISTments releases** | Appeared in search results only; not fetched. | Not verified. | Not verified as calculus. |

## The MATH benchmark is not student logs

Verified by fetching: the MATH dataset (Hendrycks et al., NeurIPS 2021) has 12,500 competition problems, each with a full step-by-step solution (`arxiv.org/abs/2103.03874`). The dataset card lists the fields `problem`, `solution`, `level` and `type`, with 7,500 training and 5,000 test examples, and no student attempts or logs (`huggingface.co/datasets/hendrycks/competition_math`). The code repository is MIT-licensed (`github.com/hendrycks/math`). The Hugging Face card also states that access to that copy "has been disabled" after a DMCA takedown notice. It can supply problems and reference solutions, not data on how students work through them, so it cannot calibrate a timing model.

## What would need to happen (not done)

1. Browse the DataShop catalog in a logged-in session for calculus or "precalculus/calculus" domains and read each dataset's access level. This needs a free account, which this note did not create.
2. Ask the DataShop contact (the FAQ says "Contact us" for dataset recommendations) whether any calculus-type multi-step step logs exist.
3. If none exist, the honest options are a small, reviewed classroom pilot (see `docs/aria_pace/PILOT_PROTOCOL.md`, which needs approval) or stating that calibration for calculus is untested (`docs/aria_pace/LIMITS.md`, section c).
