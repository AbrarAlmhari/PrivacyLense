# PrivacyLens: Detecting and Redacting Sensitive Information in Documents

**ARTI 303 - Programming for Artificial Intelligence - Final Project Report (draft)**

> This is a draft that follows the section list in the course guidelines. Before submitting:
> fill in every `TO COMPLETE`, add screenshots of the app, and export to PDF.

**Team members:** `TO COMPLETE` (names and student IDs)
**Repository:** `TO COMPLETE` (GitHub link)

---

## 1. Introduction

People share documents every day: CVs, letters, forms, screenshots of chats. These documents often
contain personal details that the sender did not mean to share, such as a national ID number, a
phone number or a home address. PrivacyLens is a Python application that scans text and PDF files,
finds such details, explains how risky the document is, and produces a redacted copy.

It combines two techniques: **regular-expression rules** for structured identifiers and a
**fine-tuned transformer model** (Named Entity Recognition) for names, organisations, places and addresses.

## 2. Target users and requirements

**Users:** students, employees and individuals who review a document before sharing it.

| Need | How PrivacyLens answers it |
|---|---|
| "I do not notice every personal detail myself." | Automatic detection of 9 categories |
| "I want to decide what is hidden." | Review step with a tick box per item and selection by category |
| "The hidden text must really be gone." | True PDF redaction with an automatic verification step |
| "I do not want to upload private files to a website." | Runs locally; files stay in memory; nothing is logged |
| "I need to understand the result." | Highlighted text, risk score with a visible calculation, stated limitations |

## 3. Objectives

1. Load pasted text, TXT files and text-based PDFs, with clear handling of invalid input.
2. Produce one clean "canonical text" that every module works on.
3. Detect emails, phone numbers (including Saudi formats), ID-like numbers, bank cards and IBANs with rules.
4. Train a model to detect persons, organisations, locations and addresses.
5. Calculate and explain a privacy risk score.
6. Let the user review detections and export sanitized text, a redacted PDF and a scan summary.
7. Evaluate both layers with precision, recall and F1 on data that was not used for training.

## 4. Existing solutions

| Solution | Type | Strength | Gap that PrivacyLens addresses |
|---|---|---|---|
| Manual black boxes in a PDF editor | Manual | Simple | The text usually stays in the file and can be copied; items are easy to overlook |
| Adobe Acrobat Pro "Redact" | Commercial tool | Real redaction | Paid; the user must find most items manually |
| Microsoft Presidio | Open-source library | Many recognisers, extensible | A developer toolkit, not a ready app with review and PDF output |
| Cloud PII services (Google Sensitive Data Protection, Amazon Comprehend) | Cloud API | Accurate, many languages | The document must be sent to a cloud service; needs an account and payment |

PrivacyLens is a small local application that joins the three steps (detect, review, really redact)
and shows how it reached its result.

## 5. Dataset

| | |
|---|---|
| Name | NVIDIA Nemotron-PII |
| Source | https://huggingface.co/datasets/nvidia/Nemotron-PII |
| License | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Content | 200,000 synthetic English documents (forms, invoices, emails, notes) from 58 domains with character-level labels for 55 PII types |
| Why chosen | Open licence, no real personal data, document-like text, exact span labels |
| Used | 20,000 training, 2,000 validation, 3,000 test documents (test from the dataset's separate test file) |

**Limitations:** synthetic text; personas based on U.S. Census data (US-centric names and addresses);
English only; only inserted values are labelled.

**Second evaluation set (ours):** 60 fictional documents generated from 14 templates
(letters, emails, CVs, forms, chats, invoices...) with fictional Saudi-style and international
names, phone numbers and IDs. 20 are a development set, 40 a held-out test set. Gold labels are
created by the generator. *(See Figure: `reports/figures/01_dataset_label_distribution.png`, `02_category_distribution.png`.)*

**Ethics:** no real personal or confidential document was used at any point. The app processes
files locally in memory and never logs their content.

## 6. Design and architecture

*(Insert the diagram from README "How it works".)*

- **Shared data structure:** every detector returns `Detection(id, category, text, start, end, source, confidence, raw_label)`.
- **Canonical text rule:** cleaning happens once in the loader; all offsets refer to that text.
- **Modules:** `file_processor`, `regex_detector`, `ner_detector`, `risk_analyzer`, `redactor`,
  `report_generator`, `pipeline`, `visualization`; training side: `data_preparation`, `train_ner`, `evaluation`.
- **Configuration:** `config/config.yaml` holds limits, label mapping, hyperparameters, risk weights.
- **Engineering practices:** custom exception classes, logging without sensitive text, type hints,
  docstrings, 109 pytest tests, pinned requirements, fixed random seeds.

## 7. Data processing workflow

| Step | Implementation | Result |
|---|---|---|
| Load | Download two parquet files, read with pandas | 200,000 documents |
| Validate | Required columns, missing values, parse span lists, check every offset against its text | 0 missing values, 0 invalid spans, non-unique `uid` discovered |
| Clean | Remove duplicate texts, empty texts, documents over 6,000 characters | 199,958 documents |
| Explore | Counts per label, split, format, locale, domain; text length; spans per document | Median length 759 characters, 8.4 spans per document |
| Visualize | Five figures (label distribution, categories, length, domains, spans per document) | `reports/figures/01-05` |
| Transform | Map 55 labels to 4 categories, merge first + last name, seeded train/validation/test split | 31,467 training windows after tokenization |
| Export | JSON Lines files | `data/processed/*.jsonl` |
| Summary | One JSON with all reports | `reports/metrics/data_summary.json` |

A finding worth reporting: the first validation flagged 9.5% of the rows. Inspection showed the
labels were correct and our check was too strict, so the check was fixed instead of deleting data.

For a user's document the same steps apply at run time: load, validate (type, size, scanned PDF),
clean (canonical text), explore and visualize (statistics and charts in the app), transform
(detection and scoring), export (TXT, PDF, JSON, CSV), summary.

## 8. AI method

**Task:** token classification. Each token receives one of 9 labels (`O`, and `B-`/`I-` for PERSON, ORG, LOCATION, ADDRESS).

**Model:** `distilbert-base-cased`, a pretrained transformer from the Hugging Face Hub, fine-tuned by
us. Pretraining gives it general knowledge of English; fine-tuning teaches it our four categories.

**Training:** plain PyTorch loop (custom Dataset, DataLoader, AdamW, linear warm-up/decay, mixed
precision), 3 epochs, 16.9 minutes on an RTX 3050. The best epoch by validation F1 is saved.
*(Figure: `06_training_history.png`.)*

**Rule layer:** 12 regular-expression rules, each documented and tested separately, with extra
validators (Luhn checksum, digit counts) and priority ordering.

**Merging:** rules win on overlaps; exact repeats of detected names are added.

## 9. Evaluation

Metric: strict entity-level precision, recall and F1 (category and both boundaries must match),
plus character-level recall.

**9.1 AI model, 3,000 unseen public documents** *(Figure: `07_test_scores_by_category.png`)*

| Category | Precision | Recall | F1 |
|---|---|---|---|
| PERSON | 0.981 | 0.983 | 0.982 |
| ORG | 0.905 | 0.926 | 0.915 |
| LOCATION | 0.940 | 0.960 | 0.950 |
| ADDRESS | 0.969 | 0.973 | 0.971 |
| Overall | 0.951 | 0.962 | 0.956 |

**9.2 Does training help?** The off-the-shelf model `dslim/distilbert-NER` reaches F1 0.687 on the
same documents (PERSON 0.782, ORG 0.529, LOCATION 0.733) and has no address category. Part of this gap is a difference in labelling conventions: the baseline was trained on news text (CoNLL-2003 style labels) and marks many names that the PII dataset does not label, which lowers its precision here. *(Figure: `08_model_comparison.png`.)*

**9.3 Rule layer, same public documents:** EMAIL 0.995, PHONE 0.943, CREDIT_CARD 0.962, overall F1 0.974.
*(Figure: `11_regex_scores.png`.)*

**9.4 Complete pipeline, 40 held-out synthetic documents** *(Figure: `10_synthetic_test_scores.png`)*

| Category | Precision | Recall | F1 | Entities |
|---|---|---|---|---|
| EMAIL | 1.000 | 1.000 | 1.000 | 39 |
| PHONE | 1.000 | 1.000 | 1.000 | 33 |
| ID_LIKE | 1.000 | 1.000 | 1.000 | 14 |
| CREDIT_CARD | 1.000 | 1.000 | 1.000 | 7 |
| IBAN | 1.000 | 1.000 | 1.000 | 9 |
| PERSON | 1.000 | 0.990 | 0.995 | 98 |
| ORG | 0.812 | 0.619 | 0.703 | 42 |
| LOCATION | 0.844 | 1.000 | 0.916 | 38 |
| ADDRESS | 1.000 | 1.000 | 1.000 | 17 |
| Overall | 0.956 | 0.943 | 0.949 | 297 |

Character-level recall: 94.5%.

Note on fairness: the rules were tuned on the validation split and the development documents.
Two rule bugs were later found in the error table of the synthetic test set and fixed; before these
two fixes the synthetic test scores were PHONE 0.923, CREDIT_CARD 0.933, overall 0.939.

**9.5 Error analysis** (full table: `reports/metrics/error_analysis_synthetic_test.csv`)

| Example | Expected | Predicted | Reason |
|---|---|---|---|
| "...records at Al Waha Medical Center" | ORG | LOCATION "Al" / "Al Waha" | Looks like a place name; training data has company names, few institutions |
| "Intern at Red Sea Analytics (2022)" | ORG | nothing | Organisation built from a geographic name |
| "Software Engineer at Oasis Travel Agency" | ORG (3 words) | ORG "Oasis" | Boundary too short |
| "Billed to: Abdulrahman Mansour / Company: ..." | PERSON | ORG | Invoice layout: the "billed to" field is usually a company in the training data |
| "EXPERIENCE" (CV heading) | nothing | LOCATION | Upper-case heading after a city name |

**9.6 Confidence threshold.** On the development set, F1 is highest at 0.7, but recall is the same
from 0.3 to 0.7. The default is 0.5 because for redaction a missed item is worse than a false alarm
and the user reviews every detection. *(Figure: `09_threshold_study.png`.)*

## 10. Application and results

`TO COMPLETE`: add screenshots of (1) the input step, (2) highlighted detections, (3) risk summary
with charts, (4) review and PDF before/after preview, (5) export with the verification message,
(6) an error message for a scanned PDF, (7) the Model & data insights page.

## 11. Limitations

- English only; no OCR for scanned PDFs or images.
- Organisation names are the weakest category (F1 0.70 on our synthetic documents).
- Training and evaluation data are synthetic; performance on real documents is not measured.
- The synthetic test set is small (40 documents from 14 templates).
- `ID_LIKE` is a pattern match and does not prove a number is a real identifier.
- PDF redaction does not inspect bookmarks, attachments or form fields.
- The risk score is project-defined, not an official standard.

## 12. Conclusion

PrivacyLens shows a complete AI workflow in Python: a documented public dataset is validated,
cleaned, explored and transformed; a transformer is fine-tuned and evaluated against a baseline;
and the result is delivered in an interactive application that redacts PDFs for real. On unseen documents the
fine-tuned model reaches F1 0.96, where an off-the-shelf NER model reaches 0.69. Future work: Arabic support, OCR, more
institution-style organisation names in training, and DOCX input.

## 13. References

1. Steier, A., Manoel, A., Haushalter, A., Van Segbroeck, M. (2025). *Nemotron-PII: Synthesized Data for Privacy-Preserving AI.* NVIDIA. https://huggingface.co/datasets/nvidia/Nemotron-PII
2. Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
3. Hugging Face Transformers documentation. https://huggingface.co/docs/transformers
4. PyTorch documentation. https://pytorch.org/docs
5. PyMuPDF documentation. https://pymupdf.readthedocs.io
6. Streamlit documentation. https://docs.streamlit.io
7. `dslim/distilbert-NER` model card. https://huggingface.co/dslim/distilbert-NER
8. Microsoft Presidio. https://microsoft.github.io/presidio/
9. ARTI 303 lecture slides 1-14, Imam Abdulrahman Bin Faisal University, 2026.

## 14. Individual contributions

`TO COMPLETE` honestly, per member (see also `AI_USAGE_REPORT.md`).

| Member | Package (plan Section 11) | Contribution |
|---|---|---|
| 1 | Data preparation & file processing | |
| 2 | Regex detection | |
| 3 | NER detection | |
| 4 | Risk analysis & redaction | |
| 5 | Streamlit app & integration | |

---

## Appendix: presentation outline (about 10 slides)

1. Title, team
2. Problem and target users
3. Existing solutions and our idea
4. Dataset and data workflow (one EDA figure)
5. Architecture: two layers and the canonical text rule
6. Training the model (training-history figure)
7. Results: test scores and baseline comparison
8. Live demo: upload sample PDF -> review -> download redacted PDF
9. Error analysis and limitations
10. Contributions, AI usage statement, questions
