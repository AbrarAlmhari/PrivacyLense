# Overview of the Work

This document is the complete record of how PrivacyLens was built: what was done, in which
order, why each decision was taken, where the data and models come from, and how the code works.
It has five parts.

| Part | Content | Read it when you want to... |
|---|---|---|
| 1. Overview of the Work (this part) | The whole project on a few pages | get the big picture |
| 2. Data and Model Sources | Dataset, models, licences, citations | cite sources or answer "where does the data come from?" |
| 3. Project Log | Every step and decision in the order it happened | understand why something was done |
| 4. Code Guide | Every code file explained, linked to the lectures | explain the code |
| 5. Model Card | Facts about the trained model and its results | present the model |

Two companion documents are separate files: **How to Run PrivacyLens** and
**Weak Points and Improvements**.

## 1. The project in short

PrivacyLens is the ARTI 303 final project. It scans pasted text, TXT files and text-based PDFs
for sensitive information, shows a privacy risk score, lets the user choose what to hide, and
exports a sanitized text file and a redacted PDF in which the selected text is deleted.

It finds nine categories with two methods:

| Method | Categories | Why this method |
|---|---|---|
| An AI model trained in this project | PERSON, ORG, LOCATION, ADDRESS | Names have no fixed format; a model reads the context |
| Rules (regular expressions) | EMAIL, PHONE, ID_LIKE, CREDIT_CARD, IBAN | These follow fixed formats; rules are exact and easy to explain |

## 2. What was produced

| Item | Where | Size |
|---|---|---|
| Application (Streamlit, two pages) | `app.py` | 619 lines |
| 14 code modules | `src/` | about 3,000 lines |
| 4 workflow scripts (data, training, test documents, evaluation) | `scripts/` | |
| 109 automated tests | `tests/` | all pass |
| Trained AI model | `models/privacylens-ner/` | 250 MB |
| Prepared dataset: 20,000 / 2,000 / 3,000 documents | `data/processed/` | |
| 60 fictional test documents with correct answers, 5 sample files | `data/synthetic/`, `data/gold/`, `data/samples/` | |
| 11 charts, 6 app screenshots | `reports/figures/`, `reports/screenshots/` | |
| Results and error tables | `reports/metrics/` | |
| Documentation | `README.md`, `docs/`, `AI_USAGE_REPORT.md` | |

## 3. The work in order

Steps 1 to 12 were done on 16 November 2026, step 13 on 24 November 2026. Details and reasons are in Part 3 (Project Log).

| # | Step | What was done | Result |
|---|---|---|---|
| 1 | Read the material | Course guidelines, Plan v2 and lectures 1 to 14 | Decision to fine-tune a small pretrained model with a plain PyTorch loop, using only taught techniques |
| 2 | Set up the environment | Python 3.14, virtual environment, PyTorch with GPU support | Training possible on the laptop's RTX 3050 |
| 3 | Choose the dataset | Compared four public datasets by licence, content and size | NVIDIA Nemotron-PII (CC BY 4.0, synthetic) |
| 4 | Design the architecture | One cleaned text with a position map; rules plus AI; one `scan()` function | PDF redaction becomes possible; app and tests use the same code |
| 5 | Build the data pipeline | Load, Validate, Clean, Explore, Visualize, Transform, Export, Summary | 199,958 clean documents; a too-strict check was found and corrected |
| 6 | Train the model | DistilBERT, 3 epochs, 16.9 minutes | Validation F1 0.966 |
| 7 | Build detection and redaction | Rules, model inference, merging, risk score, text and PDF redaction with verification | A bug in the newest PDF library was found; an older version is fixed in the requirements |
| 8 | Create test documents | 60 fictional documents from 14 templates with automatic correct answers | Independent style for testing the full pipeline |
| 9 | Evaluate (3 runs) | Model, baseline, rules and full pipeline; error analysis | Rules improved using validation data; two fixes after test errors are disclosed |
| 10 | Build the app | Upload / paste / sample, four result tabs, insights page | Checked in the browser; two display problems fixed |
| 11 | Write tests and documentation | 109 tests; README, log, code guide, model card, report draft, AI usage report | All tests pass |
| 12 | Follow-up session | Clean-installation test, hand tests of weak points, screenshots, these PDF documents, start file `run_app.bat` | Recorded in the Project Log, Section 11 |
| 13 | GitHub repository (24 November) | Code, documentation, prepared data and the trained model (through Git LFS) pushed to GitHub | Recorded in the Project Log, Section 12 |

## 4. Results

Scores are "strict": a detected item counts as correct only when its category and its exact
start and end match the correct answer. F1 combines precision (how many detections are right)
and recall (how many true items are found); 1.0 is perfect.

**The AI model on 3,000 public test documents it never saw**

| Category | Precision | Recall | F1 |
|---|---|---|---|
| PERSON | 0.981 | 0.983 | 0.982 |
| ORG | 0.905 | 0.926 | 0.915 |
| LOCATION | 0.940 | 0.960 | 0.950 |
| ADDRESS | 0.969 | 0.973 | 0.971 |
| **Overall** | **0.951** | **0.962** | **0.956** |

![Scores of the trained model per category on the public test set](../reports/figures/07_test_scores_by_category.png)

**Comparison on the same documents**

| System | F1 |
|---|---|
| Our fine-tuned model | 0.956 |
| Public model without fine-tuning (`dslim/distilbert-NER`) | 0.687 |
| Our rules (EMAIL, PHONE, CREDIT_CARD) | 0.974 |

Part of the gap to the public model comes from different labelling habits, not only from quality.

![Our model compared with the public baseline model](../reports/figures/08_model_comparison.png)

**The complete system on 40 fictional test documents**

Precision 0.956, recall 0.943, F1 0.949. The weak category is ORG (F1 0.70).
Two rule fixes were made after reading errors in these documents; the strictly held-out
result is F1 0.939.

![Scores of the complete system on the fictional test documents](../reports/figures/10_synthetic_test_scores.png)

**Training**

| Epoch | Training loss | Validation loss | Validation F1 |
|---|---|---|---|
| 1 | 0.0535 | 0.0081 | 0.9529 |
| 2 | 0.0039 | 0.0078 | 0.9580 |
| 3 | 0.0019 | 0.0078 | 0.9657 |

![Loss and validation score during training](../reports/figures/06_training_history.png)

## 5. How the work was checked

| Check | Result |
|---|---|
| 109 automated tests | All pass, on the build environment and on a newly installed environment without a graphics card |
| Model tested on documents never used for training | F1 0.956 |
| Full pipeline tested on documents of a different style | F1 0.949 |
| Redacted PDFs re-opened by the program to confirm that the text is gone | Passes for the three sample PDFs |
| App tried in the browser: sample files, empty input, both pages | Works |
| Installation from `requirements.txt` in a new environment | Works on Windows 11 (about 8 minutes) |

## 6. What is not finished

The project, including the trained model, is in the team's GitHub repository. The report PDF,
the slides and the team sections of the AI usage report are still to be done by the team.
The separate document **Weak Points and Improvements** lists these and all known limits.
