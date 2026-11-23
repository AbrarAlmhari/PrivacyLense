"""Scan summary export (the Export and Summary workflow steps).

Privacy rule: the summary contains **counts, categories, positions and
scores only**. By default it does not contain the sensitive strings
themselves, so the summary file is safe to share.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import datetime

from src import __version__
from src.schemas import Detection, LoadResult, RiskResult

LIMITATIONS = [
    "Results are pattern-based and model-based and may contain errors (missed items and false alarms).",
    "PrivacyLens does not guarantee that a document is safe to share. Always review the result yourself.",
    "English text only. Text inside images and scanned PDFs is not analysed.",
    "An ID_LIKE match is only a pattern match; it does not prove that a number is a real identifier.",
    "The risk score is a project-defined indicator, not an official privacy standard.",
]


def build_summary(
    load_result: LoadResult,
    detections: list[Detection],
    risk: RiskResult,
    selected_ids: set[str] | list[str],
    notes: list[str] | None = None,
    model_name: str = "",
    include_sensitive_text: bool = False,
) -> dict:
    """Build the scan summary as a dictionary (ready for JSON export).

    Args:
        include_sensitive_text: If True, the matched strings are added to the
            detection list. Off by default.
    """
    selected = set(selected_ids)
    by_category = Counter(detection.category for detection in detections)
    by_source = Counter(detection.source for detection in detections)
    redacted_by_category = Counter(detection.category for detection in detections if detection.id in selected)

    detection_rows = []
    for detection in detections:
        row = {
            "id": detection.id,
            "category": detection.category,
            "start": detection.start,
            "end": detection.end,
            "length": detection.end - detection.start,
            "source": detection.source,
            "confidence": detection.confidence,
            "rule_or_label": detection.raw_label,
            "redacted": detection.id in selected,
        }
        if include_sensitive_text:
            row["text"] = detection.text
        detection_rows.append(row)

    return {
        "tool": {"name": "PrivacyLens", "version": __version__, "ner_model": model_name},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "source_type": load_result.source_type,
            "file_name": load_result.file_name,
            "statistics": load_result.stats,
            "warnings": load_result.warnings,
        },
        "detections": {
            "total": len(detections),
            "by_category": dict(by_category),
            "by_source": dict(by_source),
            "items": detection_rows,
        },
        "risk": risk.to_dict(),
        "redaction": {
            "selected": len(selected & {detection.id for detection in detections}),
            "by_category": dict(redacted_by_category),
        },
        "processing_notes": notes or [],
        "limitations": LIMITATIONS,
    }


def summary_to_json(summary: dict) -> str:
    """Serialise the summary as pretty-printed JSON text."""
    return json.dumps(summary, indent=2, ensure_ascii=False)


def detections_to_csv(summary: dict) -> str:
    """Serialise the detection list of a summary as CSV text."""
    items = summary["detections"]["items"]
    buffer = io.StringIO()
    if items:
        writer = csv.DictWriter(buffer, fieldnames=list(items[0].keys()))
        writer.writeheader()
        writer.writerows(items)
    return buffer.getvalue()
