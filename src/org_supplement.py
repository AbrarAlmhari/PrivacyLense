"""Pipeline for the team's extra ORG documents ("ORG supplement").

The team wrote 500 short business documents to help the model recognise
company names, which is its weakest category. This module turns them into
training, validation and test files with the same workflow as the main data:

    Load -> Validate -> Complete labels -> Split by template -> Export -> Summary

Run it with:  ``python scripts/05_prepare_org_supplement.py``

Two problems of the source file are fixed here (details in docs/DATASET_UPDATE.md):

1. Only company names were labelled. The person names and cities in the same
   documents were not, which would teach the model that "Sarah Mitchell" is
   *not* a person. ``complete_labels`` adds those labels.
2. The documents come from 20 templates; inside one template only the company
   names change. A random split would put the same sentences in training and
   testing, so ``split_by_template`` keeps whole templates apart.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

from src.config import resolve_path
from src.data_preparation import DataError, load_processed
from src.schemas import NER_CATEGORIES

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("uid", "domain", "document_type", "document_format", "locale", "text", "entities")
SPLITS = ("train", "validation", "test")


# ------------------------------------------------------------------ 1. LOAD
def load_supplement(config: dict) -> list[dict]:
    """Read the source file (one JSON document per line)."""
    path = resolve_path(config["org_supplement"]["source_file"])
    if not path.exists():
        raise DataError(f"ORG supplement file not found: {path}")
    with open(path, encoding="utf-8") as file:
        documents = [json.loads(line) for line in file if line.strip()]
    logger.info("Loaded %d ORG supplement documents", len(documents))
    return documents


# -------------------------------------------------------------- 2. VALIDATE
def entity_problems(document: dict) -> list[str]:
    """Return a list of problems with one document (empty list = valid)."""
    problems = [f"missing key '{key}'" for key in REQUIRED_KEYS if key not in document]
    if problems:
        return problems
    text = document["text"]
    if not isinstance(text, str) or not text.strip():
        return ["empty text"]
    previous_end = -1
    for entity in sorted(document["entities"], key=lambda item: item["start"]):
        start, end, label = entity.get("start"), entity.get("end"), entity.get("label")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            problems.append(f"offsets outside the text: {entity}")
        elif label not in NER_CATEGORIES:
            problems.append(f"unknown label {label}")
        elif text[start:end] != text[start:end].strip():
            problems.append(f"span has spaces at its edge: {text[start:end]!r}")
        elif start < previous_end:
            problems.append("overlapping entities")
        else:
            previous_end = end
    return problems


def validate_supplement(documents: list[dict], config: dict) -> dict:
    """Check every document and its overlap with the data we already use.

    Raises ``DataError`` if a document is broken. Overlaps are reported, and
    documents that also appear in a Nemotron split are removed in ``clean``.
    """
    problems = {document.get("uid", f"row {index}"): entity_problems(document) for index, document in enumerate(documents)}
    problems = {uid: found for uid, found in problems.items() if found}
    if problems:
        raise DataError(f"{len(problems)} ORG supplement document(s) are invalid, e.g. {next(iter(problems.items()))}")

    existing_texts, existing_uids = set(), set()
    for split in SPLITS:
        for document in load_processed(split, config):
            existing_texts.add(document["text"])
            existing_uids.add(document["uid"])

    texts = [document["text"] for document in documents]
    uids = [document["uid"] for document in documents]
    report = {
        "documents": len(documents),
        "invalid_documents": 0,
        "duplicate_texts": len(texts) - len(set(texts)),
        "duplicate_uids": len(uids) - len(set(uids)),
        "same_text_as_a_nemotron_document": sum(text in existing_texts for text in texts),
        "same_uid_as_a_nemotron_document": sum(uid in existing_uids for uid in uids),
        "labels_in_source": dict(Counter(entity["label"] for document in documents for entity in document["entities"])),
        "longest_text_characters": max(len(text) for text in texts),
    }
    logger.info("ORG supplement validation: %s", report)
    return report


def clean(documents: list[dict], config: dict) -> tuple[list[dict], dict]:
    """Drop exact duplicates, documents already in the Nemotron data, and over-long texts."""
    existing_texts = {document["text"] for split in SPLITS for document in load_processed(split, config)}
    max_characters = config["data"]["max_text_characters"]
    kept, seen = [], set()
    removed = Counter()
    for document in documents:
        if document["text"] in seen:
            removed["duplicate_text"] += 1
        elif document["text"] in existing_texts:
            removed["already_in_nemotron_data"] += 1
        elif len(document["text"]) > max_characters:
            removed["too_long"] += 1
        else:
            seen.add(document["text"])
            kept.append(document)
    report = {"rows_before": len(documents), **{f"removed_{key}": value for key, value in removed.items()}, "rows_after": len(kept)}
    logger.info("ORG supplement cleaning: %s", report)
    return kept, report


# ------------------------------------------------------- 3. COMPLETE LABELS
def complete_labels(document: dict, person_names: list[str], location_names: list[str]) -> dict:
    """Add PERSON and LOCATION labels for the listed names.

    A name is labelled wherever it appears as a whole word and does not
    overlap an existing entity (so "Riyadh" inside the company name
    "Riyadh Smart Services" stays part of the ORG).
    """
    text = document["text"]
    entities = [dict(entity) for entity in document["entities"]]
    for label, names in (("PERSON", person_names), ("LOCATION", location_names)):
        for name in names:
            for match in re.finditer(rf"\b{re.escape(name)}\b", text):
                start, end = match.span()
                if any(start < entity["end"] and entity["start"] < end for entity in entities):
                    continue
                entities.append({"start": start, "end": end, "label": label})
    entities.sort(key=lambda entity: entity["start"])
    return {**document, "entities": entities, "regex_entities": document.get("regex_entities", [])}


# ------------------------------------------------------ 4. SPLIT BY TEMPLATE
def split_by_template(documents: list[dict], config: dict) -> dict[str, list[dict]]:
    """Put whole templates (``document_type``) into train, validation or test.

    The templates are shuffled with a fixed seed, separately for structured and
    unstructured documents, so each split gets both formats.
    """
    settings = config["org_supplement"]
    rng = random.Random(settings["seed"])
    validation_per_format = settings["validation_templates"] // 2
    test_per_format = settings["test_templates"] // 2

    assignment: dict[str, str] = {}
    for document_format in sorted({document["document_format"] for document in documents}):
        templates = sorted({d["document_type"] for d in documents if d["document_format"] == document_format})
        if len(templates) <= validation_per_format + test_per_format:
            raise DataError(f"Not enough '{document_format}' templates to hold some out.")
        rng.shuffle(templates)
        for index, template in enumerate(templates):
            if index < test_per_format:
                assignment[template] = "test"
            elif index < test_per_format + validation_per_format:
                assignment[template] = "validation"
            else:
                assignment[template] = "train"

    splits: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for document in documents:
        splits[assignment[document["document_type"]]].append(document)
    return splits


def organisation_family(name: str) -> str:
    """First word of a company name ("Al Noor" counts as one word)."""
    words = name.split()
    return " ".join(words[:2]) if words[0] == "Al" and len(words) > 1 else words[0]


def family_overlap(splits: dict[str, list[dict]]) -> dict:
    """How many company-name families of validation/test were also seen in training."""

    def families(documents: list[dict]) -> set[str]:
        return {
            organisation_family(document["text"][entity["start"] : entity["end"]])
            for document in documents
            for entity in document["entities"]
            if entity["label"] == "ORG"
        }

    train_families = families(splits["train"])
    return {
        split: {"families": len(families(splits[split])), "also_in_train": len(families(splits[split]) & train_families)}
        for split in ("validation", "test")
    }


# --------------------------------------------------------- 5. EXPORT + SUMMARY
def export(splits: dict[str, list[dict]], config: dict) -> dict[str, Path]:
    """Write ``data/processed/org_supplement_<split>.jsonl``."""
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    columns = ("uid", "domain", "document_type", "document_format", "locale", "text", "entities", "regex_entities")
    paths = {}
    for split, documents in splits.items():
        path = processed_dir / f"org_supplement_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as file:
            for document in documents:
                file.write(json.dumps({key: document[key] for key in columns}, ensure_ascii=False) + "\n")
        paths[split] = path
        logger.info("Exported %d documents to %s", len(documents), path)
    return paths


def run_pipeline(config: dict) -> dict:
    """Run every step and save ``reports/metrics/org_supplement_summary.json``."""
    settings = config["org_supplement"]
    documents = load_supplement(config)                                         # 1. Load
    validation_report = validate_supplement(documents, config)                  # 2. Validate
    documents, cleaning_report = clean(documents, config)
    labels_before = Counter(e["label"] for d in documents for e in d["entities"])
    documents = [complete_labels(d, settings["person_names"], settings["location_names"]) for d in documents]  # 3.
    labels_after = Counter(e["label"] for d in documents for e in d["entities"])
    splits = split_by_template(documents, config)                               # 4. Split
    export(splits, config)                                                      # 5. Export

    summary = {
        "source_file": settings["source_file"],
        "validation": validation_report,
        "cleaning": cleaning_report,
        "labels_before_completion": dict(labels_before),
        "labels_after_completion": dict(labels_after),
        "splits": {
            split: {
                "documents": len(items),
                "templates": sorted({d["document_type"] for d in items}),
                "entities": dict(Counter(e["label"] for d in items for e in d["entities"])),
            }
            for split, items in splits.items()
        },
        "company_name_families": family_overlap(splits),
    }
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "org_supplement_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
