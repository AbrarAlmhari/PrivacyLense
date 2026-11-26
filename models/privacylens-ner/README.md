# Model card: privacylens-ner (version 2)

## Summary

| | |
|---|---|
| **Task** | Token classification (Named Entity Recognition) for PII detection |
| **Labels** | `PERSON`, `ORG`, `LOCATION`, `ADDRESS` (BIO scheme, 9 classes including `O`) |
| **Base model** | `distilbert-base-cased` (Apache 2.0), about 65 million parameters |
| **Fine-tuning data** | 20,000 documents from NVIDIA Nemotron-PII (CC BY 4.0, synthetic) + 350 of the team's fictional business documents with company names |
| **Language** | English |
| **Size** | about 250 MB |
| **Trained** | 26 November 2026, 16.1 minutes on an NVIDIA RTX 3050 (8 GB) |
| **Code** | `src/train_ner.py`, run with `python scripts/02_train_model.py` |
| **Version history** | v1 (16 November 2026): Nemotron-PII only. v2 (26 November 2026): + the team's ORG documents. Details: `docs/DATASET_UPDATE.md` |

## Intended use

Finding names of people, organisations, places and street addresses in English documents so that a
person can review and redact them with the PrivacyLens app. It is an assistant for a human reviewer.

**Not intended for:** automatic redaction without review, legal or compliance guarantees,
non-English text, scanned documents, or identifying real people.

## Training

| Setting | Value |
|---|---|
| Training documents | 20,350 (31,817 windows): 20,000 Nemotron-PII + 350 ORG supplement documents, each used once per epoch |
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
`street_address` -> ADDRESS. The team's documents label company names (ORG); the person names and
cities in them were labelled by `scripts/05_prepare_org_supplement.py`.

| Epoch | Train loss | Validation loss | Validation F1 (2,000 Nemotron documents) |
|---|---|---|---|
| 1 | 0.0531 | 0.0084 | 0.9543 |
| 2 | 0.0038 | 0.0072 | 0.9648 |
| 3 | 0.0018 | 0.0069 | 0.9679 |

Version 2 was chosen over a second candidate (team documents used three times per epoch) and over
version 1 using validation data only (see `docs/DATASET_UPDATE.md`, Section 6).

## Evaluation

Strict entity-level scores (category and both boundaries must match), confidence threshold 0.5.

**Public test set: 3,000 Nemotron-PII documents from the dataset's separate test file**

| Category | Precision | Recall | F1 | Support | F1 of v1 |
|---|---|---|---|---|---|
| PERSON | 0.983 | 0.981 | 0.982 | 2,936 | 0.982 |
| ORG | 0.910 | 0.942 | 0.925 | 1,667 | 0.915 |
| LOCATION | 0.949 | 0.959 | 0.954 | 1,868 | 0.950 |
| ADDRESS | 0.979 | 0.985 | 0.982 | 475 | 0.971 |
| Overall (micro) | 0.955 | 0.966 | 0.961 | 6,946 | 0.956 |

Character-level recall (share of sensitive characters covered): 97.3% (v1: 96.8%).
Baseline `dslim/distilbert-NER` on the same documents: overall F1 0.687 (PERSON 0.782, ORG 0.529, LOCATION 0.733).
Part of this gap is a difference in labelling conventions: the baseline was trained on news text (CoNLL-2003 style labels) and marks many names that the PII dataset does not label, which lowers its precision here.

**Held-out synthetic documents (40 fictional letters, CVs, forms, chats; different style from the training data)**

| Category | Precision | Recall | F1 | Support | F1 of v1 |
|---|---|---|---|---|---|
| PERSON | 1.000 | 0.990 | 0.995 | 98 | 0.995 |
| ORG | 0.800 | 0.571 | 0.667 | 42 | 0.703 |
| LOCATION | 0.905 | 1.000 | 0.950 | 38 | 0.916 |
| ADDRESS | 1.000 | 1.000 | 1.000 | 17 | 1.000 |

**The team's ORG documents, 4 templates never used in training (100 documents)**

ORG F1 1.000 (v1: 0.813), overall 0.999 (v1: 0.835). Optimistic: short, regular business forms,
and the first words of the company names were seen in training.

## Limitations and risks

- **Synthetic, US-centric training data.** Real documents may differ in style and names.
- **Organisations outside the "company" pattern** (universities, hospitals, medical centers,
  agencies) are often missed, split, or partly labelled as a location or address. This is the main
  known weakness, and version 2 did not fix it (ORG F1 0.667 on the synthetic test documents).
- **Only inserted values are labelled in the dataset**, so the model may have learned to ignore
  some names that a human would mark.
- **False negatives are possible.** A missed name stays visible in the output. Always review.
- **The synthetic evaluation set is small** (40 documents) and built from 14 templates, so its
  numbers have wide uncertainty.
- **One training run.** Differences of a few thousandths between versions can come from chance.
- English only; no dates of birth, usernames or other categories.

## How to load

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/privacylens-ner")
model = AutoModelForTokenClassification.from_pretrained("models/privacylens-ner")
```

In the project, use `src.ner_detector.detect_ner(text)` instead; it handles long texts and offsets.
