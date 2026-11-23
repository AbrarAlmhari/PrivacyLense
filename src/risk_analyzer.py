"""Merge detections from both layers and compute the privacy risk score.

The risk score is **project-defined and transparent** (plan Section 8). It is
*not* an official privacy standard; it is a simple, explainable way to tell
the user "how much sensitive information is in this document".
"""

from __future__ import annotations

import logging
import re

from src.schemas import NER_CATEGORIES, REGEX_CATEGORIES, Detection, RiskResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------
def merge_detections(
    regex_detections: list[Detection],
    ner_detections: list[Detection],
    notes: list[str] | None = None,
) -> list[Detection]:
    """Combine regex and NER detections into one clean, sorted list.

    Overlap rule (plan Section 8): when a regex detection and an NER detection
    cover overlapping text, the regex detection wins, because structured
    identifiers (emails, phones, IDs) are matched more precisely by rules.
    Exact duplicates are removed.

    Args:
        regex_detections: Output of ``detect_regex``.
        ner_detections: Output of ``detect_ner``.
        notes: Optional list; a short message is appended for every decision
            (no sensitive text is written, only categories and counts).
    """
    merged: list[Detection] = []
    dropped = 0

    # Regex first (higher priority), then NER; longer matches before shorter.
    candidates = sorted(regex_detections, key=lambda d: (d.start, -(d.end - d.start)))
    candidates += sorted(ner_detections, key=lambda d: (d.start, -(d.end - d.start)))

    for candidate in candidates:
        overlaps_existing = any(
            candidate.start < kept.end and kept.start < candidate.end for kept in merged
        )
        if overlaps_existing:
            dropped += 1
            continue
        merged.append(candidate)

    merged.sort(key=lambda d: d.start)
    if dropped and notes is not None:
        notes.append(
            f"{dropped} overlapping detection(s) were removed "
            "(regex results take priority over NER results on the same text)."
        )
    logger.info("Merged detections: %d kept, %d dropped as overlaps", len(merged), dropped)
    return merged


def propagate_detections(
    text: str,
    detections: list[Detection],
    categories: tuple[str, ...] = ("PERSON", "ORG"),
    min_length: int = 4,
    notes: list[str] | None = None,
) -> list[Detection]:
    """Find *other* occurrences of names that were already detected.

    A neural model can catch "Sara Al-Qahtani" in one sentence and miss the
    very same name three lines later. For a redaction tool, one missed
    occurrence defeats the purpose, so we search the text for every exact
    repetition of each detected name and add it (source = "propagated").
    """
    known_texts = {
        (d.category, d.text) for d in detections if d.category in categories and len(d.text) >= min_length
    }
    added: list[Detection] = []

    for category, name in sorted(known_texts, key=lambda item: -len(item[1])):
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
        for match in pattern.finditer(text):
            start, end = match.span()
            overlaps = any(start < d.end and d.start < end for d in detections + added)
            if overlaps:
                continue
            added.append(
                Detection(
                    id=f"p-{len(added) + 1:04d}",
                    category=category,
                    text=name,
                    start=start,
                    end=end,
                    source="propagated",
                    confidence=None,
                    raw_label="repeat_of_detected_text",
                )
            )

    if added and notes is not None:
        notes.append(f"{len(added)} repeated occurrence(s) of already-detected names were added.")
    return sorted(detections + added, key=lambda d: d.start)


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------
def normalize_for_counting(category: str, text: str) -> str:
    """Normalise a detection so that repeats are counted only once.

    "Sara@Mail.com" and "sara@mail.com" are the same email; "055 123 4567"
    and "0551234567" are the same phone number.
    """
    if category in ("PHONE", "ID_LIKE", "CREDIT_CARD", "IBAN"):
        return re.sub(r"[\s.\-()]", "", text).upper()
    return " ".join(text.lower().split())


def score(detections: list[Detection], config: dict) -> RiskResult:
    """Compute the 0-100 privacy risk score.

    Rules (all numbers come from ``config.yaml`` -> ``risk``):

    1. **Unique items**: the same item repeated many times counts once.
    2. **Diminishing returns**: the first unique item of a category gives the
       full base points; every additional unique item gives 25% of the base.
    3. **Per-category cap**: a category gives at most 2 x its base points.
    4. **Total cap**: the total never exceeds 100.

    Worked example: one ID (30) + one phone (20) + two names (20 + 5) = 75 -> High.
    """
    risk_config = config["risk"]
    weights: dict = risk_config["weights"]
    extra_factor: float = risk_config["extra_item_factor"]
    cap_multiplier: float = risk_config["category_cap_multiplier"]
    total_cap: int = risk_config["total_cap"]

    unique_items: dict[str, set[str]] = {}
    for detection in detections:
        key = normalize_for_counting(detection.category, detection.text)
        unique_items.setdefault(detection.category, set()).add(key)

    breakdown: dict = {}
    raw_total = 0.0
    for category in REGEX_CATEGORIES + NER_CATEGORIES:
        if category not in unique_items:
            continue
        base = weights.get(category, 0)
        unique_count = len(unique_items[category])
        uncapped = base + base * extra_factor * (unique_count - 1)
        points = min(uncapped, base * cap_multiplier)
        raw_total += points
        breakdown[category] = {
            "detections": sum(1 for d in detections if d.category == category),
            "unique_items": unique_count,
            "base_points": base,
            "points": round(points, 2),
            "capped": uncapped > points,
        }

    final_score = int(round(min(raw_total, total_cap)))
    notes = []
    if raw_total > total_cap:
        notes.append(f"Raw total {raw_total:.0f} was capped at {total_cap}.")

    return RiskResult(
        score=final_score,
        level=_level_for(final_score, risk_config["bands"]),
        breakdown=breakdown,
        notes=notes,
    )


def _level_for(score_value: int, bands: list[dict]) -> str:
    """Translate a score into a level name using the configured bands."""
    for band in bands:
        if score_value <= band["max"]:
            return band["level"]
    return bands[-1]["level"]
