"""Shared data structures used by every PrivacyLens module.

These classes are the "interface contract" from the project plan (Section 4).
Every detector returns ``Detection`` objects, the loader returns a
``LoadResult`` and the risk analyzer returns a ``RiskResult``. Because all
modules agree on these shapes, they can be developed and tested separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Categories found by the rule-based (regex) layer.
REGEX_CATEGORIES = ("EMAIL", "PHONE", "ID_LIKE", "CREDIT_CARD", "IBAN")

# Categories found by the AI (NER) layer.
NER_CATEGORIES = ("PERSON", "ORG", "LOCATION", "ADDRESS")

ALL_CATEGORIES = REGEX_CATEGORIES + NER_CATEGORIES


@dataclass
class Detection:
    """One piece of sensitive information found in the canonical text.

    ``start`` and ``end`` are character offsets in the canonical text
    (start is inclusive, end is exclusive), so ``text[start:end]`` always
    gives back the matched string.
    """

    id: str  # unique, e.g. "r-0012" (regex) or "n-0003" (NER)
    category: str  # one of ALL_CATEGORIES
    text: str  # matched string
    start: int  # character offset (inclusive)
    end: int  # character offset (exclusive)
    source: str  # "regex" | "ner" | "propagated"
    confidence: float | None = None  # model probability (NER only)
    raw_label: str | None = None  # original rule name or model label

    def to_dict(self) -> dict:
        """Return the detection as a plain dictionary (for tables / JSON)."""
        return asdict(self)


@dataclass
class CharBox:
    """Where one character of the canonical text sits on a PDF page."""

    page: int  # 0-based page number
    line: int  # line counter (unique per page), used to group characters
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class LoadResult:
    """Everything the loader knows about one input."""

    canonical_text: str
    source_type: str  # "text" | "txt" | "pdf"
    file_name: str | None = None
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # PDF only: one entry per character of canonical_text (None for characters
    # that were inserted by us, such as line breaks). Used to draw redactions.
    char_boxes: list[CharBox | None] | None = None
    # PDF only: the original file bytes (never modified).
    pdf_bytes: bytes | None = None


@dataclass
class RiskResult:
    """Result of the project-defined privacy risk score."""

    score: int  # 0-100
    level: str  # Low | Medium | High | Critical
    breakdown: dict = field(default_factory=dict)  # category -> details
    notes: list[str] = field(default_factory=list)  # decisions worth logging

    def to_dict(self) -> dict:
        return asdict(self)
