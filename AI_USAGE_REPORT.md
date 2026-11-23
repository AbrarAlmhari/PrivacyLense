# AI Usage Report - PrivacyLens

Course: ARTI 303 Programming for Artificial Intelligence - Final Project

This report is required by the project guidelines (Section 3, "AI tool transparency").
It states honestly how generative AI was used. **Sections marked `TO COMPLETE` must be filled in
by the team members themselves before submission.**

---

## 1. Tools used

| Tool | Version / model | Used for |
|---|---|---|
| Claude Code (Anthropic), desktop app | Claude Fable 5.2 | Designing, writing, running and testing the project code; choosing the dataset; training and evaluating the model; writing the documentation |

No other generative AI tool was used in this build session. `TO COMPLETE`: add any tool used by team members (e.g. for the plan, report wording or slides).

---

## 2. Extent of AI assistance (plain statement)

In one working session on 16 November 2026, the AI assistant produced the first complete version of
this repository working autonomously from the team's project plan, the course guidelines and the
lecture slides. This includes:

- all files in `src/`, `scripts/`, `tests/`, `app.py`, `config/`
- the choice of dataset (NVIDIA Nemotron-PII) and base model (distilbert-base-cased)
- running the data pipeline, the model training (on the team member's own computer and GPU) and the evaluation
- the documentation: `README.md`, `docs/PROJECT_LOG.md`, `docs/CODE_GUIDE.md`, `docs/MODEL_CARD.md`, `docs/REPORT_DRAFT.md`

The project idea, scope, module design, interface contract, risk-score rules and task distribution
come from the team's own document `PrivacyLens_Project_Plan_v2.docx`, which was given to the AI as input.

---

## 3. Prompt log

### Prompt 1 (start of the session)

Inputs attached: Lectures 1-14 (PDF), `PrivacyLens_Project_Plan_v2.docx`, `ARTI303_Final_Project_Guidelines_v2.pdf`.

> I have this project where we will build and train an AI system where it can detect sensitive
> information and redact them from a pdf file. I have attached my lectures content that we will use
> to build the project and also attached the project plan description and idea. I want you to start
> working on it and build it, make sure to keep it simple and aligning with the things we studied,
> yet make sure it is professional, works well, and can be actually used by everyone. [...] If you
> can find a great data source that will help us that would be great, make sure to document
> everything and keep everything recorded so I can follow up with you and actually understand the
> code and know how the project was built. [...] choose what is best for the project, I want you to
> act as a product designer, a project manager and a developer [...]

`TO COMPLETE`: add every later prompt (date, what was asked, what was received).

| # | Date | Prompt (summary) | What the AI produced | What we did with it |
|---|---|---|---|---|
| 1 | 2026-11-16 | Build the project from the plan (above) | Complete first version of the repository | `TO COMPLETE` |
| 2 | 2026-11-16 | "Ok how can i run the app?" | The start command and short usage steps | `TO COMPLETE` |
| 3 | 2026-11-16 | Give the record of the work, the explanation of everything done and the data sources as a PDF; create a PDF that explains how to run the app; list weak points and improvements in a separate file | `docs/DATA_SOURCES.md`, `docs/HOW_TO_RUN.md`, `docs/WEAK_POINTS_AND_IMPROVEMENTS.md`, `docs/PROJECT_RECORD.md`, the three PDFs in `docs/pdf/`, app screenshots in `reports/screenshots/`, `run_app.bat`, `scripts/build_pdf_docs.py`; a clean-installation test and hand tests of weak points | `TO COMPLETE` |
| 4 | 2026-11-23 | "how can i push the folder to a new github repo" | Step-by-step Git commands and a check of what would be uploaded | `TO COMPLETE` |
| 5 | 2026-11-24 | "ok make sure to push everything cuz my team will clone the repo and work after me" | Git repository with the model in Git LFS, cleaned copies of the run logs, team section in the README, updated run guide and documents, the Git commands and a check of the staged files; the commit and push were run by the team member | `TO COMPLETE` |
| 6 | | | | |

---

## 4. Decisions the AI made (so the team can review them)

All of these are explained with reasons in `docs/PROJECT_LOG.md`.

1. Dataset: NVIDIA Nemotron-PII (CC BY 4.0, synthetic), 20,000 / 2,000 / 3,000 documents for train / validation / test.
2. Model: fine-tune `distilbert-base-cased` with a plain PyTorch loop instead of only using a pretrained NER model.
3. Two additions beyond Plan v2: a redacted PDF output, and the categories `ADDRESS`, `CREDIT_CARD`, `IBAN`.
4. Gold labels for the synthetic documents are generated automatically by the template generator
   (the plan asked for cross-labelling by members).
5. Default confidence threshold 0.5; repeated names are propagated; everything is pre-selected for redaction.
6. Library version pin `PyMuPDF==1.26.7` because of a redaction bug in version 2.1.

---

## 5. Verification and corrections

### 5.1 Checks performed by the AI during the session (evidence in the repository)

| Check | Evidence | Result |
|---|---|---|
| Automated tests | `tests/` (109 tests), run with `pytest` | all pass |
| Model quality | `scripts/04_evaluate.py` -> `reports/metrics/evaluation_results.json` | F1 0.956 on 3,000 unseen documents |
| Comparison with a baseline | same file, `baseline_model` | baseline F1 0.687 |
| App behaviour | manual run in the browser: sample PDF, empty input, both pages | works; two small issues fixed |
| PDF redaction really removes text | `tests/test_pdf_redaction.py` and the in-app verification | passes |

### 5.2 Errors of the AI's own first attempts that were found and corrected

| Problem found | How it was found | Correction |
|---|---|---|
| Dataset validation flagged 9.5% of rows as invalid | Looked at the flagged rows | The check was too strict (number vs text, letter case); corrected, 0 invalid rows |
| Text after a redaction bar moved by one character | Looked at the rendered PDF | Bug in PyMuPDF 2.1; pinned 1.26.7 and added a regression test |
| Card-number rule found only 14% of the dataset's cards | Evaluation, then error analysis on the validation split | Added a layout-based card rule |
| Several national phone formats were missed | Same | Added `phone_national` and `phone_area_code` rules |
| A closing bracket was included in a phone match; a Saudi number was called a card | Error table of the synthetic test set | Fixed both rules, added tests, and documented that these two fixes used test errors |
| Old results stayed on screen after an input error | Manual test in the browser | Results are cleared at the start of every scan |

### 5.3 Verification by the team - `TO COMPLETE`

Each member should record what they personally checked, for example:

- read and can explain the module(s) they own (list files)
- ran `pytest` and the four scripts on their own machine (date, result)
- tried the app with their own non-confidential test inputs (what worked, what failed)
- reviewed a sample of the synthetic gold labels in `data/gold/`
- any code they changed, fixed or rewrote after review (commit links)

| Member | Files reviewed | What was verified | Corrections made |
|---|---|---|---|
| | | | |

---

## 6. Statement of independently implemented work - `TO COMPLETE`

State clearly which parts were written or changed by team members without AI assistance, and
which parts are AI-generated and reviewed. Do not claim AI-generated code as independently written.

- Project idea, problem definition, plan v2, risk-score design, task distribution: written by the team.
- First implementation of the repository: AI-generated (see Section 2), reviewed by: `TO COMPLETE`
- Changes made by members after the first version: `TO COMPLETE`
