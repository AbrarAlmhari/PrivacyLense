"""Evaluation metrics for the detectors.

Two views of quality are reported:

* **Entity-level (strict)** precision / recall / F1: a predicted entity is
  correct only if its category *and* both boundaries match the gold label
  exactly (plan Section 7.3).
* **Character-level recall**: the share of sensitive characters that the
  system would hide. For a redaction tool this is the number that matters
  most: a name found with a slightly wrong boundary is still mostly hidden.

A "span" is a tuple ``(category, start, end)``.
"""

from __future__ import annotations

import json

from src.config import resolve_path

Span = tuple[str, int, int]


def load_synthetic(split: str, config: dict) -> list[dict]:
    """Read the synthetic documents of one split ("dev" or "test") with their gold labels."""
    gold_dir = resolve_path(config["evaluation"]["gold_dir"])
    synthetic_dir = resolve_path(config["evaluation"]["synthetic_dir"])
    with open(gold_dir / "ner_gold.json", encoding="utf-8") as file:
        ner_gold = json.load(file)
    with open(gold_dir / "regex_gold.json", encoding="utf-8") as file:
        regex_gold = json.load(file)

    documents = []
    for document_id, entry in ner_gold.items():
        if entry["split"] != split:
            continue
        text = (synthetic_dir / split / f"{document_id}.txt").read_text(encoding="utf-8")
        documents.append(
            {
                "id": document_id,
                "text": text,
                "entities": entry["entities"] + regex_gold[document_id]["entities"],
            }
        )
    return documents


def precision_recall_f1(true_positives: int, false_positives: int, false_negatives: int) -> dict:
    """Compute precision, recall and F1 from the three basic counts."""
    predicted = true_positives + false_positives
    actual = true_positives + false_negatives
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "support": actual,  # number of gold entities
    }


def evaluate_entities(gold_documents: list[set[Span]], predicted_documents: list[set[Span]], categories: tuple[str, ...]) -> dict:
    """Strict entity-level evaluation over many documents.

    Args:
        gold_documents: one set of gold spans per document.
        predicted_documents: one set of predicted spans per document.
        categories: the categories to evaluate (others are ignored).

    Returns:
        ``{"per_category": {...}, "overall": {...}, "macro_f1": float}`` where
        "overall" is the micro-average over all listed categories.
    """
    counts = {category: {"tp": 0, "fp": 0, "fn": 0} for category in categories}
    for gold, predicted in zip(gold_documents, predicted_documents):
        gold = {span for span in gold if span[0] in counts}
        predicted = {span for span in predicted if span[0] in counts}
        for span in predicted:
            counts[span[0]]["tp" if span in gold else "fp"] += 1
        for span in gold - predicted:
            counts[span[0]]["fn"] += 1

    per_category = {
        category: precision_recall_f1(count["tp"], count["fp"], count["fn"]) for category, count in counts.items()
    }
    overall = precision_recall_f1(
        sum(count["tp"] for count in counts.values()),
        sum(count["fp"] for count in counts.values()),
        sum(count["fn"] for count in counts.values()),
    )
    with_support = [scores["f1"] for scores in per_category.values() if scores["support"] > 0]
    macro_f1 = round(sum(with_support) / len(with_support), 4) if with_support else 0.0
    return {"per_category": per_category, "overall": overall, "macro_f1": macro_f1}


def character_recall(gold_documents: list[set[Span]], predicted_documents: list[set[Span]], categories: tuple[str, ...]) -> dict:
    """Share of gold sensitive characters covered by *any* prediction.

    The category of the prediction does not matter here: if a city was
    labelled as ADDRESS it is still hidden by the redaction.
    """
    covered = {category: 0 for category in categories}
    total = {category: 0 for category in categories}

    for gold, predicted in zip(gold_documents, predicted_documents):
        hidden: set[int] = set()
        for _, start, end in predicted:
            hidden.update(range(start, end))
        for category, start, end in gold:
            if category not in total:
                continue
            total[category] += end - start
            covered[category] += sum(1 for position in range(start, end) if position in hidden)

    result = {
        category: round(covered[category] / total[category], 4) if total[category] else None for category in categories
    }
    all_total = sum(total.values())
    result["overall"] = round(sum(covered.values()) / all_total, 4) if all_total else None
    return result


def collect_errors(text: str, gold: set[Span], predicted: set[Span], document_id: str, context: int = 30) -> list[dict]:
    """Build error-analysis rows for one document.

    Error types:
      * ``missed``          - gold entity with no overlapping prediction
      * ``false alarm``     - prediction with no overlapping gold entity
      * ``wrong boundary``  - same category, overlapping, different boundaries
      * ``wrong category``  - overlapping but a different category

    Only call this on synthetic / public data: the rows contain text snippets.
    """

    def overlaps(a: Span, b: Span) -> bool:
        return a[1] < b[2] and b[1] < a[2]

    def snippet(start: int, end: int) -> str:
        left = text[max(0, start - context) : start]
        right = text[end : end + context]
        return f"...{left}[{text[start:end]}]{right}...".replace("\n", " ")

    rows: list[dict] = []
    for gold_span in sorted(gold - predicted, key=lambda span: span[1]):
        partners = [span for span in predicted if overlaps(gold_span, span)]
        if not partners:
            error, predicted_text = "missed", "(nothing)"
        else:
            partner = partners[0]
            error = "wrong boundary" if partner[0] == gold_span[0] else "wrong category"
            predicted_text = f"{partner[0]}: {text[partner[1]:partner[2]]}"
        rows.append(
            {
                "document": document_id,
                "error": error,
                "expected": f"{gold_span[0]}: {text[gold_span[1]:gold_span[2]]}",
                "predicted": predicted_text,
                "context": snippet(gold_span[1], gold_span[2]),
            }
        )
    for predicted_span in sorted(predicted - gold, key=lambda span: span[1]):
        if any(overlaps(predicted_span, span) for span in gold):
            continue  # already reported above from the gold side
        rows.append(
            {
                "document": document_id,
                "error": "false alarm",
                "expected": "(nothing)",
                "predicted": f"{predicted_span[0]}: {text[predicted_span[1]:predicted_span[2]]}",
                "context": snippet(predicted_span[1], predicted_span[2]),
            }
        )
    return rows


def spans_from_detections(detections) -> set[Span]:
    """Convert ``Detection`` objects into comparable spans."""
    return {(detection.category, detection.start, detection.end) for detection in detections}


def spans_from_entities(entities: list[dict]) -> set[Span]:
    """Convert gold entities (``{"start", "end", "label"}``) into spans."""
    return {(entity["label"], entity["start"], entity["end"]) for entity in entities}
