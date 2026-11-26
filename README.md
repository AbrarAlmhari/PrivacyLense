# PrivacyLens

**Find and redact sensitive information in text and PDF files before you share them.**

PrivacyLens is the ARTI 303 (Programming for AI) final project. It scans pasted text, TXT files
and text-based PDFs, detects personal and confidential details, explains a privacy risk score,
lets you choose what to hide, and exports a sanitized text file and a **truly redacted PDF**
(the text is deleted from the file, not just covered).

| | |
|---|---|
| **AI task** | Named Entity Recognition (token classification) with a transformer we fine-tuned ourselves |
| **Detects with AI** | `PERSON`, `ORG`, `LOCATION`, `ADDRESS` |
| **Detects with rules** | `EMAIL`, `PHONE` (incl. Saudi formats), `ID_LIKE`, `CREDIT_CARD`, `IBAN` |
| **Dataset** | [NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) (CC BY 4.0, fully synthetic) + 500 fictional ORG documents written by the team |
| **Base model** | [distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased) (Apache 2.0) |
| **Interface** | Streamlit |
| **Language** | English (Arabic is future work; Arabic-Indic digits are supported in the rules) |

> **Important:** PrivacyLens uses rules and an AI model. Both can miss items or raise false alarms.
> It does **not** guarantee that a document is safe to share. Always review the result yourself.

---

## Results at a glance

All numbers are produced by `scripts/04_evaluate.py` and stored in `reports/metrics/`.
Entity-level scores are **strict**: a prediction counts only if the category and both boundaries match exactly.

The model is **version 2** (26 November 2026): trained on Nemotron-PII plus 350 of the team's own
business documents with company names. What changed against version 1 is explained in
[`docs/DATASET_UPDATE.md`](docs/DATASET_UPDATE.md).

**AI model on 3,000 unseen public test documents**

| Category | Precision | Recall | F1 | Gold entities | F1 of v1 |
|---|---|---|---|---|---|
| PERSON | 0.983 | 0.981 | 0.982 | 2,936 | 0.982 |
| ORG | 0.910 | 0.942 | 0.925 | 1,667 | 0.915 |
| LOCATION | 0.949 | 0.959 | 0.954 | 1,868 | 0.950 |
| ADDRESS | 0.979 | 0.985 | 0.982 | 475 | 0.971 |
| **Overall** | **0.955** | **0.966** | **0.961** | 6,946 | 0.956 |

The same test with an off-the-shelf NER model (`dslim/distilbert-NER`, no fine-tuning) gives F1 **0.687**,
so fine-tuning on PII data clearly helps (part of the gap comes from different labelling conventions: the baseline was trained on news text). Regex layer on the same documents: F1 **0.974** (EMAIL 0.995, PHONE 0.943, CREDIT_CARD 0.962).

**Complete pipeline on 40 held-out synthetic documents** (fictional letters, CVs, forms, chats):
precision **0.965**, recall **0.936**, F1 **0.950**; 93.7% of all sensitive characters are covered.
The weakest category is `ORG` (F1 0.67; version 1: 0.70): see [Limitations](#limitations).

**The team's ORG documents** (100 documents from 4 templates not used in training): ORG F1 **1.00**
(version 1: 0.81). This number is optimistic; see the dataset update.

---

## Quick start

Requirements: Git with **Git LFS** (included in Git for Windows; otherwise see https://git-lfs.com),
a recent Python (developed and tested with 3.14; the pinned library versions need 3.12 or newer),
and about 2 GB of free disk space. A GPU is only needed for training, not for using the app.

```bash
# 1. Get the project (the 250 MB model is downloaded through Git LFS)
git lfs install
git clone <repository address>
cd <repository folder>

# 2. Create a virtual environment and install the libraries (about 8 minutes)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt     # Windows
# .venv/bin/python -m pip install -r requirements.txt            # macOS / Linux

# 3. Start the app
.venv\Scripts\python.exe -m streamlit run app.py                # or double-click run_app.bat on Windows
```

Open http://localhost:8501 and try the files in `data/samples/` (tab "Try a sample").
Full instructions with screenshots: [`docs/pdf/2_PrivacyLens_How_To_Run.pdf`](docs/pdf/2_PrivacyLens_How_To_Run.pdf).

**About the AI model.** The fine-tuned model is in the repository (`models/privacylens-ner/`);
its 250 MB weight file is stored with Git LFS. If `model.safetensors` is only about 1 KB after
cloning, Git LFS was not active: run `git lfs install` and then `git lfs pull`.
If the model folder is missing, the app falls back to the public model `dslim/distilbert-NER`
(lower quality, no `ADDRESS` category) and says so on screen.

For GPU training install the CUDA build of PyTorch first:
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`

---

## Working on this project as a team

1. **Start here:** read [`docs/pdf/1_PrivacyLens_Project_Record.pdf`](docs/pdf/1_PrivacyLens_Project_Record.pdf)
   (what was built and why) and [`docs/pdf/3_PrivacyLens_Weak_Points_and_Improvements.pdf`](docs/pdf/3_PrivacyLens_Weak_Points_and_Improvements.pdf)
   (what to improve, in a suggested order).
2. **Before and after every change:** run `python -m pytest`. All 122 tests must pass, and none
   should be skipped (a skip means the model is missing).
3. **Changing the model or the rules:** tune only on the validation split and `data/synthetic/dev`,
   never on the test data. Re-run `scripts/04_evaluate.py` and record what you changed and why in
   `docs/PROJECT_LOG.md`.
4. **Retraining** replaces `models/privacylens-ner/model.safetensors`. Commit the new model only
   when its scores are better, because every model version stays in the Git LFS storage.
5. **Keep `PyMuPDF==1.26.7`.** Newer versions move text next to a redaction bar; a test checks this.
6. **Documentation:** edit the Markdown files in `docs/`, then rebuild the PDFs with
   `python scripts/build_pdf_docs.py` (needs Edge or Chrome).
7. **AI use:** add every use of an AI tool to `AI_USAGE_REPORT.md`, as the course requires.
8. **Never commit real personal data**, and never commit documents scanned with the app.

What is in the repository and what is not:

| Folder | In the repository? |
|---|---|
| Code, tests, configuration, documentation, figures, results | yes |
| `models/privacylens-ner/` (trained model) | yes, weight file through Git LFS |
| `models/archive/`, `models/experiments/` (local copies of older and trial models) | no: older versions are in the Git history |
| `data/processed/` (the 20,000 / 2,000 / 3,000 prepared documents, 33 MB, plus the 350 / 50 / 100 ORG supplement documents) | yes |
| `data/extra/` (the team's 500 extra ORG documents, source file) | yes |
| `data/synthetic/`, `data/gold/`, `data/samples/` | yes |
| `reports/logs/` (logs of the training and evaluation runs) | yes |
| `data/raw/` (original dataset, 300 MB) | no: re-download with `scripts/01_prepare_data.py` |
| `outputs/` (local app logs) | no |

---

## Reproduce everything (data -> model -> evaluation)

Run the four scripts in order from the project folder. The prepared data and the model are already in
the repository, so steps 1 and 2 are only needed to rebuild them from scratch:

```bash
python scripts/01_prepare_data.py        # download + validate + clean + explore + visualize + export  (~1 min)
python scripts/02_train_model.py         # fine-tune the model (needs script 05 first, see below)       (~17 min on GPU)
python scripts/03_make_synthetic_docs.py # fictional evaluation documents + sample files               (seconds)
python scripts/04_evaluate.py            # metrics, figures and error-analysis tables                  (~3 min)
```

The model in the repository (version 2) was also trained on the team's extra ORG documents.
To rebuild that data, run `python scripts/05_prepare_org_supplement.py` after step 1 and before
step 2. `python scripts/06_compare_models.py` compares two model folders on the same documents
(see [`docs/DATASET_UPDATE.md`](docs/DATASET_UPDATE.md)).

Then run the tests:

```bash
pytest
```

122 tests cover every module. Tests that need the trained model are skipped automatically if it is missing.

---

## How it works

```
 Input (text / TXT / PDF)
        |
        v
 file_processor.py   Load -> Validate -> Clean  => "canonical text" (+ position of every character in the PDF)
        |
        +--> regex_detector.py   rules: EMAIL, PHONE, ID_LIKE, CREDIT_CARD, IBAN
        +--> ner_detector.py     AI model: PERSON, ORG, LOCATION, ADDRESS
        |
        v
 risk_analyzer.py    merge (rules win on overlaps) -> add repeated names -> risk score 0-100
        |
        v
 redactor.py         sanitized text  +  redacted PDF (text really removed, verified afterwards)
 report_generator.py scan summary (JSON / CSV, no sensitive strings by default)
        |
        v
 app.py (Streamlit)  Input -> Detections -> Risk summary -> Review & redact -> Export
```

**The canonical text rule.** All cleaning that changes the text length happens once, in the loader.
Every other module works on that cleaned text and its character offsets, so a detection at
`start..end` always means the same characters everywhere.

**Why two layers?** Emails, phone numbers and IDs follow fixed formats, so rules are exact and easy
to explain. Names, organisations and places have no fixed format, so they need a model that
understands context.

### The data workflow (course requirement)

| Step | Training data (`src/data_preparation.py`) | A user's document (app) |
|---|---|---|
| Load | Download Nemotron-PII parquet files, read with pandas | Paste text or upload TXT/PDF |
| Validate | Columns, missing values, every span offset checked against its text | File type, size, page count, password, scanned PDF |
| Clean | Drop duplicates, empty and over-long documents | Whitespace and invisible characters -> canonical text |
| Explore | Label counts, text lengths, domains, spans per document | Document statistics, detections per category |
| Visualize | 5 EDA figures in `reports/figures/` | Charts, highlighted text, PDF before/after |
| Transform | Map 55 labels to 4 categories, merge first+last name, split train/val/test | Regex + AI + merge + risk score |
| Export | `data/processed/*.jsonl` | Sanitized TXT, redacted PDF, summary JSON/CSV |
| Summary | `reports/metrics/data_summary.json` | Scan summary with counts, score and limitations |

### Risk score

Project-defined and transparent (not an official privacy standard). Weights are in `config/config.yaml`.

- Base points: ID_LIKE / CREDIT_CARD / IBAN 30, PHONE / PERSON / ADDRESS 20, EMAIL 15, ORG / LOCATION 10
- The same item repeated counts once; each *additional* unique item adds 25% of the base
- A category gives at most 2x its base; the total is capped at 100
- Levels: 0-20 Low, 21-50 Medium, 51-75 High, 76-100 Critical
- Example: one ID (30) + one phone (20) + two names (20 + 5) = 75 -> High

---

## Project structure

```
PrivacyLens/
├── app.py                      Streamlit application (entry point)
├── config/
│   ├── config.yaml             every tunable value (limits, model, risk weights, ...)
│   └── config.example.yaml     copy of the default configuration
├── src/
│   ├── schemas.py              shared data classes: Detection, LoadResult, RiskResult
│   ├── config.py               configuration + logging helpers
│   ├── file_processor.py       load / validate / clean, PDF text + character positions
│   ├── regex_detector.py       rule-based detection
│   ├── ner_detector.py         AI detection (inference with the fine-tuned model)
│   ├── risk_analyzer.py        merge, repeated-name propagation, risk score
│   ├── redactor.py             text redaction, true PDF redaction, verification
│   ├── report_generator.py     scan summary (JSON / CSV)
│   ├── pipeline.py             scan(): the whole workflow in one function
│   ├── visualization.py        all charts
│   ├── data_preparation.py     dataset pipeline (8 workflow steps)
│   ├── org_supplement.py       pipeline for the team's 500 extra ORG documents
│   ├── train_ner.py            PyTorch training loop
│   ├── evaluation.py           precision / recall / F1, character recall, error analysis
│   └── synthetic_data.py       generator for fictional evaluation documents
├── scripts/                    01_prepare_data, 02_train_model, 03_make_synthetic_docs, 04_evaluate,
│                               05_prepare_org_supplement, 06_compare_models, build_pdf_docs
├── tests/                      122 pytest tests
├── data/
│   ├── samples/                fictional sample files for the app
│   ├── synthetic/              60 fictional evaluation documents (dev / test)
│   ├── gold/                   gold labels for the synthetic documents
│   ├── extra/                  the team's 500 extra ORG documents (source file)
│   ├── processed/              prepared dataset (created by scripts 01 and 05)
│   ├── raw/                    original dataset files (not committed, downloaded by script 01)
├── models/                     fine-tuned model (weights stored with Git LFS, created by script 02)
├── reports/
│   ├── figures/                EDA + evaluation figures
│   ├── metrics/                data summary, training history, evaluation results, error tables
│   ├── logs/                   logs of the training and evaluation runs
│   └── screenshots/            screenshots of the app
├── docs/                       project log, code guide, model card, report draft
├── AI_USAGE_REPORT.md
├── requirements.txt
└── README.md
```

---

## Example inputs and expected behaviour

| Input | Expected behaviour |
|---|---|
| `data/samples/sample_letter.pdf` | Names, organisation, city, address, ID, email and phone are found; risk is Critical; redacted PDF passes verification |
| `data/samples/sample_email.txt` | Emails, phone, names and city found; sanitized text shows `[REDACTED-...]` placeholders |
| `data/samples/sample_multi_page.pdf` | Detections on several pages; page selector in the preview |
| `data/samples/sample_scanned.pdf` | Clear message: scanned PDFs are not supported |
| Empty text | "The input does not contain any readable text." |
| A `.png` or `.docx` file | "Unsupported file type" message |
| A text without personal data | No detections, score 0, level Low |

---

## Limitations

- **English only.** Arabic text is not analysed by the model (future work).
- **No OCR.** Scanned PDFs and text inside images are not analysed; the app says so clearly.
- **Organisations are the weakest category.** On our synthetic documents the model finds only about
  57% of organisation names exactly (it misses or splits names such as "Al Waha Medical Center").
  The training data labels *company* names; institutions like hospitals and academies look different.
  The team's extra ORG documents (version 2) did not change this, because they contain only
  company names in 20 sentence patterns.
- **Training data is synthetic and US-centric.** Real documents can look different. In our tests
  Saudi-style names were recognised well, but this was measured only on generated documents.
- **`ID_LIKE` is a pattern match.** It does not prove that a number is a real identifier.
- **PDF redaction** removes the page text, links on the redacted area and document metadata. It does
  not inspect bookmarks, attachments, form fields or hidden layers.
- **The risk score** is a simple project-defined indicator, not a legal or official assessment.

---

## Data, models and licenses

| Resource | Use in this project | License |
|---|---|---|
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | Training, validation and test data | CC BY 4.0 |
| The team's 500 ORG documents (`data/extra/`) | Extra training data for company names (350), validation (50), test (100) | Part of the project (fictional) |
| [distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased) | Pretrained model that we fine-tuned | Apache 2.0 |
| [dslim/distilbert-NER](https://huggingface.co/dslim/distilbert-NER) | Baseline for comparison and fallback model | Apache 2.0 |
| PyTorch, Transformers, PyMuPDF, Streamlit, pandas, matplotlib, seaborn | Libraries | see each project |

Dataset citation: Steier, A., Manoel, A., Haushalter, A., Van Segbroeck, M. (2025).
*Nemotron-PII: Synthesized Data for Privacy-Preserving AI.* NVIDIA.

No real personal data is used anywhere in this project. All sample and evaluation documents are fictional.

---

## More documentation

PDF versions (built with `python scripts/build_pdf_docs.py`):

- [`docs/pdf/1_PrivacyLens_Project_Record.pdf`](docs/pdf/1_PrivacyLens_Project_Record.pdf) - the complete record: overview, data sources, project log, code guide, model card
- [`docs/pdf/2_PrivacyLens_How_To_Run.pdf`](docs/pdf/2_PrivacyLens_How_To_Run.pdf) - installing, starting and using the app, with screenshots
- [`docs/pdf/3_PrivacyLens_Weak_Points_and_Improvements.pdf`](docs/pdf/3_PrivacyLens_Weak_Points_and_Improvements.pdf) - known limits and what to improve next
- [`docs/pdf/4_PrivacyLens_Dataset_Update.pdf`](docs/pdf/4_PrivacyLens_Dataset_Update.pdf) - the team's extra ORG dataset: assessment, processing, retraining and results

Markdown sources:

- [`docs/DATASET_UPDATE.md`](docs/DATASET_UPDATE.md) - the extra ORG dataset and the model update (version 2)
- [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) - every dataset, model and library used, with licences and citations
- [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md) - how to run the app (on Windows you can also double-click `run_app.bat`)
- [`docs/WEAK_POINTS_AND_IMPROVEMENTS.md`](docs/WEAK_POINTS_AND_IMPROVEMENTS.md) - weak points, tested examples and suggested improvements
- [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) - how the project was built, step by step, with every decision and its reason
- [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) - a tour of the code, file by file, linked to the course lectures
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) - model card of the fine-tuned model
- [`docs/REPORT_DRAFT.md`](docs/REPORT_DRAFT.md) - draft of the project report
- [`AI_USAGE_REPORT.md`](AI_USAGE_REPORT.md) - how generative AI was used
