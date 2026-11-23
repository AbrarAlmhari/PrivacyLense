"""Dataset pipeline for the training data (Nemotron-PII).

This module implements the full data-processing workflow required by the
course guidelines, adapted to *text with span annotations*:

    Load -> Validate -> Clean -> Explore -> Visualize -> Transform -> Export -> Summary

Run it with:  ``python scripts/01_prepare_data.py``

Dataset: NVIDIA Nemotron-PII (https://huggingface.co/datasets/nvidia/Nemotron-PII),
license CC BY 4.0. It is fully synthetic, so it contains no real personal data.
"""

from __future__ import annotations

import ast
import json
import logging
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import resolve_path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["uid", "domain", "document_type", "document_format", "locale", "text", "spans"]


class DataError(Exception):
    """Raised when the dataset is missing or does not look as expected."""


# ------------------------------------------------------------------ 1. LOAD
def download_dataset(config: dict) -> None:
    """Download the dataset files from Hugging Face if they are not on disk."""
    raw_dir = resolve_path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, url in config["data"]["download_urls"].items():
        target = raw_dir / f"{split}.parquet"
        if target.exists():
            logger.info("Dataset file already present: %s", target.name)
            continue
        logger.info("Downloading %s split from %s", split, url)
        try:
            urllib.request.urlretrieve(url, target)
        except OSError as error:
            raise DataError(f"Could not download the dataset ({url}). Check your internet connection.") from error


def load_raw(config: dict) -> pd.DataFrame:
    """Read both dataset files into one DataFrame with a ``split`` column."""
    raw_dir = resolve_path(config["data"]["raw_dir"])
    frames = []
    for split in ("train", "test"):
        path = raw_dir / f"{split}.parquet"
        if not path.exists():
            raise DataError(f"Dataset file not found: {path}. Run scripts/01_prepare_data.py with internet access.")
        frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
        frame["split"] = split
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d raw documents", len(data))
    return data


# -------------------------------------------------------------- 2. VALIDATE
def _parse_spans(value: str) -> list[dict] | None:
    """The dataset stores spans as text (a Python list). Parse it safely."""
    try:
        spans = ast.literal_eval(value) if isinstance(value, str) else list(value)
    except (ValueError, SyntaxError):
        return None
    return spans if isinstance(spans, list) else None


def _spans_are_valid(text: str, spans: list[dict] | None) -> bool:
    """Every span must have a label and offsets that really point into the text."""
    if spans is None or not isinstance(text, str):
        return False
    for span in spans:
        start, end, label = span.get("start"), span.get("end"), span.get("label")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            return False
        if not 0 <= start < end <= len(text):
            return False
        # The dataset sometimes stores the helper "text" field as a number
        # (age 44) or in another letter case ("spanish" for "Spanish"), so we
        # compare as lower-case strings. The offsets are what really matter.
        if "text" in span and text[start:end].lower() != str(span["text"]).lower():
            return False
    return True


def validate(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Check the schema and every annotation; flag invalid rows.

    Returns the DataFrame with two new columns (``span_list``, ``is_valid``)
    and a small validation report.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise DataError(f"The dataset is missing required columns: {missing}")

    data = data.copy()
    data["span_list"] = data["spans"].map(_parse_spans)
    data["is_valid"] = [_spans_are_valid(text, spans) for text, spans in zip(data["text"], data["span_list"])]

    report = {
        "rows": int(len(data)),
        "missing_values": {column: int(count) for column, count in data[REQUIRED_COLUMNS].isna().sum().items()},
        "unparseable_spans": int(data["span_list"].isna().sum()),
        "rows_with_invalid_spans": int((~data["is_valid"]).sum()),
        "duplicate_uids": int(data["uid"].duplicated().sum()),
        "duplicate_texts": int(data["text"].duplicated().sum()),
    }
    logger.info("Validation report: %s", report)
    return data, report


# ----------------------------------------------------------------- 3. CLEAN
def clean(data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Remove rows that should not be used for training or evaluation.

    Note: the *text itself is not edited*, because the annotations are
    character offsets and any edit would shift them.
    """
    max_characters = config["data"]["max_text_characters"]
    rows_before = len(data)

    invalid = ~data["is_valid"]
    empty = data["text"].fillna("").str.strip().eq("")
    too_long = data["text"].str.len() > max_characters
    duplicate = data["text"].duplicated(keep="first")

    cleaned = data[~(invalid | empty | too_long | duplicate)].drop(columns=["spans", "is_valid"]).reset_index(drop=True)
    report = {
        "rows_before": int(rows_before),
        "removed_invalid_spans": int(invalid.sum()),
        "removed_empty_text": int(empty.sum()),
        "removed_too_long": int(too_long.sum()),
        "removed_duplicate_text": int(duplicate.sum()),
        "rows_after": int(len(cleaned)),
    }
    logger.info("Cleaning report: %s", report)
    return cleaned, report


# --------------------------------------------------------------- 4. EXPLORE
def explore(data: pd.DataFrame, config: dict) -> dict:
    """Compute the statistics used in the report and in the figures."""
    label_map = config["data"]["label_map"]
    raw_labels: Counter = Counter()
    categories: Counter = Counter()
    for spans in data["span_list"]:
        for span in spans:
            raw_labels[span["label"]] += 1
            if span["label"] in label_map:
                categories[label_map[span["label"]]] += 1

    text_length = data["text"].str.len()
    spans_per_document = data["span_list"].map(len)
    return {
        "documents": int(len(data)),
        "documents_per_split": data["split"].value_counts().to_dict(),
        "documents_per_format": data["document_format"].value_counts().to_dict(),
        "documents_per_locale": data["locale"].value_counts().to_dict(),
        "domains": int(data["domain"].nunique()),
        "document_types": int(data["document_type"].nunique()),
        "top_domains": data["domain"].value_counts().head(15).to_dict(),
        "text_length": {key: round(float(value), 1) for key, value in text_length.describe().items()},
        "spans_per_document": {key: round(float(value), 2) for key, value in spans_per_document.describe().items()},
        "raw_label_types": len(raw_labels),
        "raw_label_counts": dict(raw_labels.most_common()),
        "category_counts": dict(categories.most_common()),
        "share_of_spans_used_by_model": round(sum(categories.values()) / max(sum(raw_labels.values()), 1), 4),
    }


# ------------------------------------------------------------- 5. VISUALIZE
def visualize(data: pd.DataFrame, stats: dict, config: dict) -> list[Path]:
    """Save the EDA figures as PNG files and return their paths."""
    figures_dir = resolve_path(config["data"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    label_map = config["data"]["label_map"]
    saved: list[Path] = []

    def save(figure: plt.Figure, name: str) -> None:
        path = figures_dir / name
        figure.tight_layout()
        figure.savefig(path, dpi=130)
        plt.close(figure)
        saved.append(path)

    # Figure 1: all dataset labels; the ones our model learns are highlighted.
    labels = pd.Series(stats["raw_label_counts"]).head(30)
    colors = ["#1f6feb" if label in label_map else "#b6bfcc" for label in labels.index]
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.barh(labels.index[::-1], labels.values[::-1], color=colors[::-1])
    axis.set_title("Top 30 dataset labels (blue = used by the PrivacyLens model)")
    axis.set_xlabel("Number of annotated spans")
    save(figure, "01_dataset_label_distribution.png")

    # Figure 2: our categories after label mapping.
    categories = pd.Series(stats["category_counts"])
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(x=categories.index, y=categories.values, hue=categories.index, palette="Blues_d", legend=False, ax=axis)
    axis.set_title("Entities per PrivacyLens category (after label mapping)")
    axis.set_ylabel("Number of annotated spans")
    axis.set_xlabel("")
    save(figure, "02_category_distribution.png")

    # Figure 3: document length by document format.
    figure, axis = plt.subplots(figsize=(8, 4))
    plot_data = pd.DataFrame({"characters": data["text"].str.len(), "format": data["document_format"]})
    sns.histplot(data=plot_data, x="characters", hue="format", bins=60, element="step", ax=axis)
    axis.set_title("Document length (characters) by document format")
    axis.set_xlabel("Characters per document")
    save(figure, "03_text_length_distribution.png")

    # Figure 4: documents per domain.
    domains = pd.Series(stats["top_domains"])
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(domains.index[::-1], domains.values[::-1], color="#2a9d8f")
    axis.set_title("Documents per domain (top 15)")
    axis.set_xlabel("Number of documents")
    save(figure, "04_documents_per_domain.png")

    # Figure 5: annotations per document, US vs international locale.
    figure, axis = plt.subplots(figsize=(7, 4))
    plot_data = pd.DataFrame({"spans": data["span_list"].map(len), "locale": data["locale"]})
    sns.boxplot(data=plot_data, x="locale", y="spans", hue="locale", palette="Set2", legend=False, ax=axis)
    axis.set_title("Annotated spans per document by locale")
    axis.set_ylabel("Spans per document")
    save(figure, "05_spans_per_document.png")

    logger.info("Saved %d figures to %s", len(saved), figures_dir)
    return saved


# ------------------------------------------------------------- 6. TRANSFORM
def map_spans_to_entities(text: str, spans: list[dict], label_map: dict) -> list[dict]:
    """Convert dataset spans into PrivacyLens entities.

    * Labels are renamed with ``label_map``; unlisted labels are dropped.
    * A first name directly followed by a last name becomes ONE ``PERSON``
      entity ("Sara" + "Ali" -> "Sara Ali"), because that is how a name
      should be redacted.
    """
    mapped = sorted(
        (span["start"], span["end"], label_map[span["label"]]) for span in spans if span["label"] in label_map
    )
    entities: list[dict] = []
    for start, end, category in mapped:
        if entities:
            previous = entities[-1]
            if start < previous["end"]:
                continue  # overlapping annotation: keep the first one
            gap = text[previous["end"] : start]
            if category == "PERSON" and previous["label"] == "PERSON" and gap in (" ", ""):
                previous["end"] = end
                continue
        entities.append({"start": start, "end": end, "label": category})
    return entities


def transform(data: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Map labels and create the train / validation / test sets.

    To avoid data leakage, the test set comes only from the dataset's own
    ``test`` file, while train and validation come from its ``train`` file.
    Sampling uses a fixed seed, so the result is reproducible.
    """
    settings = config["data"]
    label_map, seed = settings["label_map"], settings["seed"]

    data = data.copy()
    data["entities"] = [map_spans_to_entities(text, spans, label_map) for text, spans in zip(data["text"], data["span_list"])]
    # Independent gold labels for evaluating the regex layer (not used in training).
    regex_label_map = settings.get("regex_label_map", {})
    data["regex_entities"] = [
        map_spans_to_entities(text, spans, regex_label_map) for text, spans in zip(data["text"], data["span_list"])
    ]
    columns = ["uid", "domain", "document_type", "document_format", "locale", "text", "entities", "regex_entities"]

    train_pool = data[data["split"] == "train"].sample(frac=1.0, random_state=seed)
    test_pool = data[data["split"] == "test"].sample(frac=1.0, random_state=seed)

    train_size, validation_size, test_size = settings["train_size"], settings["validation_size"], settings["test_size"]
    if len(train_pool) < train_size + validation_size or len(test_pool) < test_size:
        raise DataError("The configured train/validation/test sizes are larger than the dataset.")

    return {
        "train": train_pool.iloc[:train_size][columns].reset_index(drop=True),
        "validation": train_pool.iloc[train_size : train_size + validation_size][columns].reset_index(drop=True),
        "test": test_pool.iloc[:test_size][columns].reset_index(drop=True),
    }


# ---------------------------------------------------------------- 7. EXPORT
def export(splits: dict[str, pd.DataFrame], config: dict) -> dict[str, Path]:
    """Write each split as a JSON Lines file (one document per line)."""
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in splits.items():
        path = processed_dir / f"{name}.jsonl"
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
        paths[name] = path
        logger.info("Exported %d documents to %s", len(frame), path)
    return paths


def load_processed(split: str, config: dict) -> list[dict]:
    """Read one exported split back as a list of dictionaries."""
    path = resolve_path(config["data"]["processed_dir"]) / f"{split}.jsonl"
    if not path.exists():
        raise DataError(f"Processed file not found: {path}. Run scripts/01_prepare_data.py first.")
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# --------------------------------------------------------------- 8. SUMMARY
def summarize(
    validation_report: dict,
    cleaning_report: dict,
    stats: dict,
    splits: dict[str, pd.DataFrame],
    figures: list[Path],
    config: dict,
) -> dict:
    """Collect everything into one summary and save it as JSON."""
    split_summary = {}
    for name, frame in splits.items():
        counts: Counter = Counter(entity["label"] for entities in frame["entities"] for entity in entities)
        split_summary[name] = {"documents": int(len(frame)), "entities": dict(counts.most_common())}

    summary = {
        "dataset": {key: config["data"][key] for key in ("name", "url", "license")},
        "validation": validation_report,
        "cleaning": cleaning_report,
        "exploration": stats,
        "splits": split_summary,
        "figures": [path.name for path in figures],
        "limitations": [
            "The dataset is synthetic (generated by NVIDIA NeMo Data Designer), not real documents.",
            "Personas are grounded in U.S. Census data, so names and addresses are US-centric.",
            "English only.",
            "Only inserted PII values are annotated; other names in a text may be unlabelled.",
        ],
    }
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "data_summary.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    logger.info("Data summary written to %s", path)
    return summary


def run_pipeline(config: dict) -> dict:
    """Run all eight workflow steps in order and return the summary."""
    download_dataset(config)                                        # 1. Load
    raw = load_raw(config)
    validated, validation_report = validate(raw)                    # 2. Validate
    cleaned, cleaning_report = clean(validated, config)             # 3. Clean
    stats = explore(cleaned, config)                                # 4. Explore
    figures = visualize(cleaned, stats, config)                     # 5. Visualize
    splits = transform(cleaned, config)                             # 6. Transform
    export(splits, config)                                          # 7. Export
    return summarize(validation_report, cleaning_report, stats, splits, figures, config)  # 8. Summary
