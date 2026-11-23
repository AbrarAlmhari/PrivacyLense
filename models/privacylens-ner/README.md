# Model card: privacylens-ner

## Summary

| | |
|---|---|
| **Task** | Token classification (Named Entity Recognition) for PII detection |
| **Labels** | `PERSON`, `ORG`, `LOCATION`, `ADDRESS` (BIO scheme, 9 classes including `O`) |
| **Base model** | `distilbert-base-cased` (Apache 2.0), about 65 million parameters |
| **Fine-tuning data** | 20,000 documents from NVIDIA Nemotron-PII (CC BY 4.0, synthetic) |
| **Language** | English |
| **Size** | about 250 MB |
| **Trained** | 16 November 2026, 16.9 minutes on an NVIDIA RTX 3050 (8 GB) |
| **Code** | `src/train_ner.py`, run with `python scripts/02_train_model.py` |

## Intended use

Finding names of people, organisations, places and street addresses in English documents so that a
person can review and redact them with the PrivacyLens app. It is an assistant for a human reviewer.

**Not intended for:** automatic redaction without review, legal or compliance guarantees,
non-English text, scanned documents, or identifying real people.

## Training

| Setting | Value |
|---|---|
| Window length / overlap | 256 tokens / 32 tokens |
| Batch size | 16 |
| Optimizer | AdamW, learning rate 5e-5, weight decay 0.01 |
| Schedule | linear warm-up (6% of steps), then linear decay |
| Epochs | 3 (best epoch by validation F1 is kept) |
| Precision | mixed precision (autocast) on GPU |
| Seed | 42 |
| Libraries | Python 3.14.7, torch 2.14.1+cu126, transformers 5.23.0 |

Label mapping from the dataset: `first_name` + `last_name` -> PERSON (adjacent first and last name
merged), `company_name` -> ORG, `city` / `state` / `country` / `county` -> LOCATION,
`street_address` -> ADDRESS.

| Epoch | Train loss | Validation loss | Validation F1 |
|---|---|---|---|
| 1 | 0.0535 | 0.0081 | 0.9529 |
| 2 | 0.0039 | 0.0078 | 0.9580 |
| 3 | 0.0019 | 0.0078 | 0.9657 |

## Evaluation

Strict entity-level scores (category and both boundaries must match), confidence threshold 0.5.

**Public test set: 3,000 Nemotron-PII documents from the dataset's separate test file**

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PERSON | 0.981 | 0.983 | 0.982 | 2,936 |
| ORG | 0.905 | 0.926 | 0.915 | 1,667 |
| LOCATION | 0.940 | 0.960 | 0.950 | 1,868 |
| ADDRESS | 0.969 | 0.973 | 0.971 | 475 |
| Overall (micro) | 0.951 | 0.962 | 0.956 | 6,946 |

Character-level recall (share of sensitive characters covered): 96.8%.
Baseline `dslim/distilbert-NER` on the same documents: overall F1 0.687 (PERSON 0.782, ORG 0.529, LOCATION 0.733).
Part of this gap is a difference in labelling conventions: the baseline was trained on news text (CoNLL-2003 style labels) and marks many names that the PII dataset does not label, which lowers its precision here.

**Held-out synthetic documents (40 fictional letters, CVs, forms, chats; different style from the training data)**

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PERSON | 1.000 | 0.990 | 0.995 | 98 |
| ORG | 0.812 | 0.619 | 0.703 | 42 |
| LOCATION | 0.844 | 1.000 | 0.916 | 38 |
| ADDRESS | 1.000 | 1.000 | 1.000 | 17 |

## Limitations and risks

- **Synthetic, US-centric training data.** Real documents may differ in style and names.
- **Organisations outside the "company" pattern** (hospitals, academies, agencies) are often missed,
  split, or partly labelled as a location. This is the main known weakness.
- **Only inserted values are labelled in the dataset**, so the model may have learned to ignore
  some names that a human would mark.
- **False negatives are possible.** A missed name stays visible in the output. Always review.
- **The synthetic evaluation set is small** (40 documents) and built from 14 templates, so its
  numbers have wide uncertainty.
- English only; no dates of birth, usernames or other categories.

## How to load

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/privacylens-ner")
model = AutoModelForTokenClassification.from_pretrained("models/privacylens-ner")
```

In the project, use `src.ner_detector.detect_ner(text)` instead; it handles long texts and offsets.
