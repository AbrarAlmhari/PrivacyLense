# PrivacyLens - Project Log

This file records **how the project was built, in order, and why each decision was made**.
Read it top to bottom to follow the whole build. Dates: 16 November 2026 (one working session),
follow-up work on 24 and 26 November 2026 (Sections 12 and 13). Sections 1 to 10 describe
version 1 of the model; the current model is version 2 (Section 13).

Sources used for every decision:

- `ARTI303_Final_Project_Guidelines_v2.pdf` (course requirements and rubric)
- `PrivacyLens_Project_Plan_v2.docx` (the team's plan)
- Lectures 1-14 of ARTI 303 (to keep the techniques inside what was taught)

---

## 0. Starting point: what was asked

Build and train an AI system that detects sensitive information and redacts it from a PDF,
simple, aligned with the course, professional, and usable by anyone. Document everything.

### Two places where this build goes beyond Plan v2 (on purpose)

| Plan v2 said | What was built | Why |
|---|---|---|
| "A redacted PDF is not produced in this version" (Section 9) | A truly redacted PDF **is** produced, next to the sanitized text | The project brief for this build was explicitly "redact from a PDF file". Text export is kept, so nothing from the plan is lost. |
| "Use a pretrained NER model" (Section 7.2) | A pretrained model is **fine-tuned (trained) by us** on a PII dataset | The brief said "build and train". Fine-tuning also gave much better results than the off-the-shelf model (F1 0.956 vs 0.687). |

Everything else follows the plan: canonical text rule, `Detection` structure, function names,
risk-score rules, bands, redaction from the end backward, summary without sensitive strings,
the app sections, the repository layout, no real personal data, no logging of sensitive text.

Two categories were **added** to the plan's six: `ADDRESS` (AI model) and `CREDIT_CARD` / `IBAN` (rules),
because a redaction tool that leaves street addresses and bank details visible is not very useful.
Their risk weights are in `config.yaml` (ADDRESS 20, CREDIT_CARD 30, IBAN 30).

---

## 1. Reading the material (what the lectures allow)

| Lecture | What the project uses from it |
|---|---|
| 1-2 Python essentials | functions, dataclasses, comprehensions, type hints |
| 3 Software engineering | project structure, venv, `requirements.txt`, `pathlib`, exceptions, logging, PEP 8, docstrings, pytest, reproducibility |
| 4 NumPy / 5 pandas | dataset loading, validation, cleaning and statistics with pandas |
| 6 Visualization & EDA | matplotlib / seaborn figures |
| 7 Preprocessing | train / validation / test split, avoiding data leakage |
| 8 PyTorch | tensors, custom `Dataset`, `DataLoader`, batching, shuffling |
| 9 PyTorch computing | CPU vs GPU device, autograd (`loss.backward()`), saving the model |
| 10 NLP | tokenization, Hugging Face tokenizers and Transformers, NER |
| 13-14 Modern AI dev & MLOps | Hugging Face Hub, pretrained models and model cards, dataset selection, seeds, experiment record, training vs inference, model lifecycle |

Decision: **fine-tune a small pretrained transformer with a plain PyTorch training loop.**
This uses exactly the taught pieces (Dataset, DataLoader, autograd, GPU, Hugging Face) and keeps
every training step visible, instead of hiding it inside a high-level trainer.

---

## 2. Environment

- Machine: Windows 11, Python 3.14.7, NVIDIA RTX 3050 (8 GB).
- A virtual environment was created (Lecture 3) and the libraries installed with pip.
- PyTorch with CUDA 12.6 was installed so that training can use the GPU.

---

## 3. Choosing the dataset

Requirement: public, documented, license known, no real personal data, language = English.

| Candidate | Result |
|---|---|
| CoNLL-2003 | Rejected: news text, licence restrictions (the plan warned about this) |
| ai4privacy PII-masking | Rejected: licence metadata is inconsistent between versions |
| Gretel PII masking v1 | Not chosen: smaller (60k documents) and its dataset licence could not be confirmed during the research |
| **NVIDIA Nemotron-PII** | **Chosen** |

Why Nemotron-PII (https://huggingface.co/datasets/nvidia/Nemotron-PII):

- License **CC BY 4.0**, verified on the official dataset card.
- **Fully synthetic**: no real person's data, which satisfies the plan's "no real personal data" rule.
- **Document-like text** (forms, invoices, emails, letters) from 58 domains. The plan worried that
  public NER corpora are news-style and do not look like personal documents; this dataset closes most of that gap.
- **Exact character spans** for 55 label types, the same format our `Detection` uses.
- 200,000 documents (100k train file + 100k test file), so there is far more than we need.

Limitations (also written in the README and the report): synthetic, US-centric personas, English only.

The course prefers Kaggle but accepts "a documented public data source"; Hugging Face Hub was covered in Lecture 13.

---

## 4. Architecture decisions

1. **Canonical text with a position map.** `clean_text_with_map` cleans the text once and returns,
   for each kept character, where it came from. For PDFs each character also has its box on the
   page. This is the piece that makes PDF redaction possible: detection offsets -> page rectangles.
2. **Rules for structured data, AI for names.** Kept from the plan.
3. **One `scan()` function** (`src/pipeline.py`) used by the app, the tests and the evaluation, so
   the reported numbers describe exactly what the app does.
4. **Repeated-name propagation.** If the model finds a name once, every exact repeat of it in the
   document is added too. One missed repeat would defeat a redaction.
5. **Safe defaults.** Everything detected is pre-selected for redaction; the summary leaves out the
   sensitive strings; logs contain counts only; uploads stay in memory.
6. **Fallback model.** If the fine-tuned model is missing, a public NER model is used so the app
   still runs on a new machine, and the app says so.

---

## 5. Data pipeline (`scripts/01_prepare_data.py`)

Load -> Validate -> Clean -> Explore -> Visualize -> Transform -> Export -> Summary.

Findings while building it:

- **The annotations are stored as text**, not as a list, so they are parsed with `ast.literal_eval`.
- **First validation was too strict.** 19,101 rows (9.5%) were flagged as "invalid". Investigation
  showed the offsets were correct; the dataset's helper field sometimes stores a number (`44` instead
  of `"44"`) or a different letter case (`"spanish"` for `"Spanish"`). The check was corrected to
  compare lower-case strings. Result: 0 invalid rows. *Lesson: investigate before deleting data.*
- **`uid` is not unique** (each id appears in the US and the international version), so duplicates
  are detected on the text instead. 1 duplicate text and 41 over-long documents were removed.
- 199,958 documents remain. 20,000 are used for training, 2,000 for validation (both from the
  dataset's train file) and 3,000 for testing (from its test file), with a fixed seed.
  Keeping the test documents in a separate file avoids data leakage (Lecture 7).

Label mapping: `first_name`, `last_name` -> PERSON (a first name directly followed by a last name
is merged into one entity); `company_name` -> ORG; `city`, `state`, `country`, `county` -> LOCATION;
`street_address` -> ADDRESS. The other labels are ignored by the model.

---

## 6. Training (`scripts/02_train_model.py`)

- Base model `distilbert-base-cased`: small (about 65 million parameters), fast, and **cased**, which matters
  because capital letters are a strong signal for names.
- Labels use the BIO scheme: `B-PERSON` starts a name, `I-PERSON` continues it, `O` is everything else.
- Long documents are cut into overlapping windows of 256 tokens.
- Hyperparameters (in `config.yaml`): batch 16, learning rate 5e-5, 3 epochs, warm-up 6%, seed 42.
- Mixed precision on the GPU for speed. The best epoch (highest validation F1) is kept.

A trial run on 2,500 documents (43 seconds) already gave F1 0.93 and recognised Saudi-style names
(for example "Abdulrahman Al-Qahtani", "Fatimah Al-Dossary"), so no extra data augmentation was added.

Full run: 31,467 training windows, **16.9 minutes** on the RTX 3050.

| Epoch | Train loss | Validation loss | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 0.0535 | 0.0081 | 0.9445 | 0.9614 | 0.9529 |
| 2 | 0.0039 | 0.0078 | 0.9549 | 0.9612 | 0.9580 |
| 3 | 0.0019 | 0.0078 | 0.9580 | 0.9735 | **0.9657** |

Validation loss stays flat while training loss keeps falling, so more epochs would start to overfit; 3 is enough.

---

## 7. PDF redaction and a library bug

How redaction works: every character's page position is saved while loading. A selected detection
becomes one rectangle per line; PyMuPDF "redaction annotations" delete the text under the rectangles
and paint them black; metadata is cleared; the file is saved with garbage collection so deleted
text is not left inside. Afterwards `verify_pdf_redaction` re-opens the new file and checks that no
text can be extracted from the redacted areas.

**Bug found by looking at the output.** In the first rendered test PDF, the text *after* a redaction
bar had moved left by one character and was partly hidden under the bar. The page content showed
that PyMuPDF 2.1 compensated for 9 removed characters instead of 10. Older versions (1.24, 1.25,
1.26) were tested on a generated PDF and on a PowerPoint-exported lecture PDF and are correct.
Fix: `requirements.txt` pins `PyMuPDF==1.26.7`, and a regression test
(`test_remaining_text_does_not_move`) fails if a broken version is installed.

---

## 8. Evaluation and what it changed

Three evaluation runs were made. They are all recorded here for honesty.

**Run 1 (first complete run).** AI model on the public test set: F1 0.956. Baseline: 0.687.
Regex on the public test set: EMAIL 0.995, PHONE 0.863, **CREDIT_CARD 0.233**. Synthetic test pipeline: F1 0.939.

To understand the weak regex numbers, the errors were analysed **on the validation split and the
synthetic development set only**, not on the test data:

- 84% of the card numbers in the validation split (143 of 170) do not pass the Luhn checksum (they are invented), and the rule
  required it. Added a second rule: digits written in card layout (4-4-4-4 or 4-6-5) are a card even
  with a wrong checksum.
- National phone formats with a leading zero (`0594-737737`, `07911 789 567`) and bracketed area codes
  were missed. Added the rules `phone_national` and `phone_area_code`.
- Numbers starting with `00` without separators (customer IDs) were wrongly called international
  phone numbers. The `00` form now needs separated groups.

**Run 2 (after tuning on dev/validation).** Regex on the public test set: PHONE 0.943, CREDIT_CARD 0.952,
overall 0.973. Synthetic test pipeline unchanged: F1 0.939 (PHONE 0.923, CREDIT_CARD 0.933).

**Reading the error table of run 2 showed two real rule bugs** in the synthetic *test* documents:

1. `(+44 20 7946 0526)` - the closing bracket was included in the phone match.
2. `00966573139096` - a Saudi number was called a credit card because 14 digits passed the checksum by chance.

Both were fixed (brackets must come in matching pairs; card numbers start with 2-6) and unit tests were added.

**Run 3 (final, reported numbers).** Synthetic test pipeline: F1 **0.949**, PHONE and CREDIT_CARD 1.000.
Public test regex: 0.974.

> Transparency note: because two fixes were made after looking at synthetic *test* errors, the
> synthetic PHONE and CREDIT_CARD scores of run 3 are slightly optimistic. The run 2 values
> (PHONE F1 0.923, CREDIT_CARD F1 0.933, overall 0.939) are the strictly held-out ones. The AI-model
> numbers were never tuned on any test data.

**Confidence threshold.** On the development set, thresholds 0.3-0.7 give the same recall (0.925)
and precision rises from 0.943 to 0.990. F1 is best at 0.7. The default stays at **0.5** anyway:
for redaction a missed item is worse than a false alarm, the user reviews every item, and the slider
in the app lets them change it.

**Known weakness: organisations.** On synthetic documents ORG reaches only F1 0.70. Names such as
"Al Waha Medical Center" or "Eastern Dunes Academy" are missed, split, or partly labelled as a place.
The dataset labels *company* names; hospital- or school-style names are a different pattern.
This is reported as a limitation and as future work (add institution names to the training data).

---

## 9. The app

Built with Streamlit, one file (`app.py`), two pages:

- **Scan a document**: Input (upload / paste / sample) -> Detections (highlighted text + table) ->
  Risk summary (score, gauge, two charts, breakdown table) -> Review & redact (tick boxes, select by
  category, text preview or PDF before/after) -> Export (TXT, PDF with verification, JSON, CSV).
- **Model & data insights**: dataset statistics and EDA figures, training history, evaluation tables,
  comparison with the baseline, error analysis.

Tested in the browser: sample CV (10 detections, PDF preview and verified download), empty input
(clear error message), page switch. Fixed after testing: old results stayed visible under a new
error message; a chart axis showed fractional counts.

---

## 10. Final state

- 109 automated tests pass (`pytest`).
- Final metrics: see README or `reports/metrics/evaluation_results.json`.
- Deviations from Plan v2: listed in Section 0.

### What is still open (for the team)

1. **Model distribution.** Done on 24 November 2026: the model is in the GitHub repository,
   stored with Git LFS (see Section 12).
2. **Report PDF and presentation.** `docs/REPORT_DRAFT.md` is a complete draft; add member names,
   screenshots and individual contributions, then export it as PDF. Build the slides from it.
3. **AI Usage Report.** `AI_USAGE_REPORT.md` describes the AI assistance honestly; each member must
   add what they personally did, checked and changed.
4. **Human check of the gold labels.** The plan asked for cross-labelling by members. The synthetic
   gold labels are created automatically by the generator; members should review a sample of them.
5. **Topic registration** in the course sheet, if not done yet.
6. Ideas for later: Arabic support, OCR for scanned PDFs, better organisation detection, DOCX input.

---

## 11. Follow-up session (16 November 2026, later the same day)

Requested: the record of the work and the data sources as a PDF, a PDF that explains how to run
the app, and a separate file with weak points and improvements.

What was done, in order:

1. **The project was moved** from a temporary folder to its permanent folder. All 109 tests were
   run again in the new place and pass.
2. **Clean-installation test.** A completely new virtual environment was created and
   `pip install -r requirements.txt` was run in it: 7.5 minutes, about 1.2 GB, CPU build of PyTorch.
   All 109 tests pass in it. This proves that the requirements file is complete on Windows 11.
   (macOS and Linux were not tested.) The test environment was deleted afterwards.
3. **Speed without a graphics card** was measured in that environment: 0.2 s for the 4-page sample,
   3.7 s for 26,840 characters, 29 s for 201,299 characters; loading the model takes about 3 s.
4. **Hand tests of suspected weak points** with the real pipeline: lower-case names, upper-case
   names, Arabic text, institution names, order numbers, and data kinds outside the nine categories.
   The results are in `docs/WEAK_POINTS_AND_IMPROVEMENTS.md`. Nothing in the model or the rules was
   changed because of them, so all reported scores stay valid.
5. **Screenshots of the app** were taken automatically with a headless browser on the fictional
   sample CV and saved in `reports/screenshots/`.
6. **New documents:** `docs/DATA_SOURCES.md`, `docs/HOW_TO_RUN.md`,
   `docs/WEAK_POINTS_AND_IMPROVEMENTS.md`, `docs/PROJECT_RECORD.md` (overview).
7. **PDF versions** are built by `scripts/build_pdf_docs.py` into `docs/pdf/`. The script turns the
   Markdown files into HTML and prints them with the Edge or Chrome browser that is already on the
   computer. The Project Record PDF is assembled from the overview, the data sources, this log, the
   code guide and the model card, so the PDF and the Markdown files cannot drift apart.
8. **`run_app.bat`** was added so the app can be started with a double-click on Windows. It was
   tested: the app answers on http://localhost:8501. The file also opens the browser, because
   the plain `streamlit run app.py` command does not (`.streamlit/config.toml` sets
   `headless = true`, which avoids Streamlit's first-run e-mail question).
9. A wrong comment in `config/config.yaml` was corrected (the two dataset files hold
   100,000 rows each, not 50,000).

---

## 12. GitHub repository (24 November 2026)

Requested: push the whole project to a new GitHub repository, because the team will clone it and
continue the work.

Decisions:

| What | Decision | Why |
|---|---|---|
| Trained model (`models/privacylens-ner/`, 250 MB) | **Included.** The weight file `model.safetensors` is stored with **Git LFS** (`.gitattributes`). | GitHub refuses normal files over 100 MB. With Git LFS a teammate gets a working model with `git clone` and does not need to retrain. GitHub Free includes 10 GB of LFS storage and 10 GB of LFS downloads per month, which is about 40 clones of the model. |
| Prepared data (`data/processed/`, 33 MB) | **Included** | Teammates can run the evaluation and retrain without downloading the 300 MB original dataset. |
| Original dataset (`data/raw/`, 300 MB) | **Not included** | Both files are over GitHub's 100 MB limit, and `scripts/01_prepare_data.py` downloads them again. |
| Training and evaluation logs | **Included as `reports/logs/`** | They are evidence for the numbers in this log. The copies have the local folder paths replaced by `<project folder>` so the repository contains no personal paths. |
| `outputs/` (local app log) | **Not included** | It belongs to one computer and grows with every scan. |
| Line endings | `.gitattributes` sets `* text=auto`, and CRLF for `run_app.bat` | Teammates on Windows and macOS see the same files. |

Other changes in the same step:

- The README has a new section, "Working on this project as a team", and its quick start now
  begins with `git clone`.
- The run guide explains cloning with Git LFS and how to recognise a missing model file
  (1 KB instead of 250 MB). Its start command no longer contains the Windows user name.
- The weak-points document and the project record were updated: the model is now shared.
- The three PDF documents were rebuilt.
- Before the first commit, the staged files were checked: `git lfs ls-files` lists
  `model.safetensors`, no other staged file is over 50 MB, and no staged file or PDF contains
  the Windows user name or the e-mail address.
- The AI prepared the files and the commands; the team member ran `git init`, the commit and
  the push themselves.

---

## 13. The team's extra ORG dataset and model version 2 (26 November 2026)

Requested: check the extra dataset the team gathered (`train_with_org_augmented.jsonl`), assess
whether it is valid and useful, process it, use it to train and test the model, and report what
changed. The full write-up is `docs/DATASET_UPDATE.md`; this section keeps the decisions in order.

### 13.1 What the file contains

| Finding | How it was found | Decision |
|---|---|---|
| 20,500 rows = our 20,000 prepared training documents + 500 new ones | Compared the `uid` of every row with `data/processed/train.jsonl` | Only the 500 new rows are new data |
| 1,957 of the 20,000 old rows have a different text than ours under the same `uid` | Row-by-row comparison | Nemotron-PII stores every `uid` twice (a US and an international version of the same document). The file took the other version for these rows. All of them come from the dataset's training file, so there is **no leak into our test set**. We keep our own `train.jsonl`, because script 01 can rebuild it and because then the 500 new documents are the only difference between the old and the new model. |
| The 500 new rows: valid offsets, no duplicates, no overlap with any Nemotron split or with our synthetic test documents | `scripts/05_prepare_org_supplement.py` (validation step) and a separate string check | Usable |
| Only `ORG` is labelled; 475 person names and 325 city names in the same texts are not | Read one document of every template | Must be fixed: unlabelled names would teach the model that "Sarah Mitchell" is not a person |
| 20 templates x 25 documents; inside a template only the company names change | Replaced the ORG names by a placeholder and counted different texts per template (1 each) | A random split would test on sentences seen in training, so the split is made by template |

### 13.2 Rule for choosing the model, fixed before any result was seen

Two candidates are trained with the normal settings (same seed, 3 epochs) on the Nemotron
training documents plus the 350 supplement training documents: each supplement document used once
(`v2_repeat1`) and three times (`v2_repeat3`). Version 1 is the current model.

1. Only validation data is used for the choice: Nemotron validation (2,000), supplement validation
   (50, two unseen templates) and synthetic dev (20).
2. A candidate is allowed only if its overall F1 on Nemotron validation is at most 0.005 lower
   than version 1's.
3. Among allowed candidates, the one with the highest average ORG F1 on supplement validation and
   synthetic dev is chosen.
4. It replaces version 1 only if that average is higher than version 1's.
5. The test sets are run once, after the choice, for version 1 and the chosen model, and all test
   numbers are reported, whatever they show.

### 13.3 Training and choice

| Model | Training documents | Minutes | Nemotron validation F1 | ORG F1, supplement validation | ORG F1, synthetic dev | Average of the two ORG scores |
|---|---|---|---|---|---|---|
| v1 (current) | 20,000 | 16.9 | 0.9657 | 0.864 | 0.844 | 0.854 |
| v2_repeat1 | 20,350 | 16.1 | 0.9679 | 1.000 | 0.864 | 0.932 |
| v2_repeat3 | 21,050 | 18.2 | 0.9658 | 1.000 | 0.864 | 0.932 |

Both candidates pass rule 2. They have exactly the same ORG average, which rule 3 did not foresee.
The tie was broken, before any test set was run, with another validation number: the higher overall
F1 on Nemotron validation, which chooses **v2_repeat1** (each supplement document used once; also
the simpler setting). Its ORG average (0.932) is higher than version 1's (0.854), so it replaces
version 1 (rule 4).

### 13.4 Test results (run once, after the choice)

| Test set | v1 | v2 |
|---|---|---|
| Nemotron test (3,000), AI model, overall F1 | 0.956 | 0.961 |
| Nemotron test, ORG F1 | 0.915 | 0.925 |
| Team's documents, 4 held-out templates (100), ORG F1 | 0.813 | 1.000 |
| Synthetic test (40), complete pipeline, overall F1 | 0.949 | 0.950 |
| Synthetic test, ORG F1 | 0.703 | 0.667 |
| Synthetic test, share of sensitive characters covered | 94.5% | 93.7% |

On the synthetic test documents version 2 misses "Harbor Point Bank" (two documents) and "Blue
Falcon Logistics", which version 1 found, and finds one "Red Sea Analytics" that version 1 missed.
This is reported, not fixed: changing anything because of test errors would make the test numbers
meaningless. Institutions (university, hospital, medical center) are still missed by both versions.

### 13.5 Other decisions in this step

- Version 2 became the app's model (`models/privacylens-ner/`). Version 1 is kept locally in
  `models/archive/` and in the Git history; its result files are in `reports/metrics/v1/`.
- `scripts/04_evaluate.py` was run again, so `evaluation_results.json`, the error tables and
  figures 06 to 11 now describe version 2.
- Found on the way: 420 validation documents have their "sister" version in the training split
  (Nemotron-PII has two versions of every `uid`, and script 01 splits by row). The test split is not
  affected. Not changed now, so that version 1 and version 2 use the same data; recorded as a weak
  point with its fix (split by `uid`).
- 13 new tests; all 122 pass with version 2 and none is skipped.

