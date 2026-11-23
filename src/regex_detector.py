"""Rule-based detection with regular expressions.

Structured identifiers (emails, phone numbers, ID numbers, bank cards, IBANs)
follow fixed formats, so explicit rules are more reliable and more explainable
for them than a neural network. The AI model (``ner_detector.py``) handles
the things rules cannot describe: names, organisations and places.

Every rule is listed separately in ``RULES`` with its own name, so each one
can be documented and tested on its own.

Important honesty note: an ``ID_LIKE`` match is only a *pattern* match. It
does not prove that a number is a real or valid identifier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from src.schemas import Detection

logger = logging.getLogger(__name__)

# Arabic-Indic and Eastern Arabic-Indic digits -> Western digits.
# The mapping is 1 character -> 1 character, so offsets never shift
# (canonical text rule, plan Section 4.1).
_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Reusable pieces. "SEP" is an optional single space, dash or dot between
# groups of digits. Look-arounds stop a match from starting or ending in the
# middle of a longer number or word.
SEP = r"[\s.\-]?"
NOT_AFTER_DIGIT_OR_WORD = r"(?<![\w.,/\-])"
NOT_BEFORE_DIGIT = r"(?![\-/]?\d)(?![.,]\d)"
# A group of 1-4 digits, optionally inside a *matching* pair of brackets.
GROUP = r"(?:\(\d{1,4}\)|\d{1,4})"


def _luhn_is_valid(match_text: str) -> bool:
    """Luhn checksum used by real bank-card numbers (removes most false hits)."""
    digits = [int(char) for char in match_text if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _digit_count_between(minimum: int, maximum: int) -> Callable[[str], bool]:
    """Build a validator that checks how many digits a match contains."""

    def validator(match_text: str) -> bool:
        return minimum <= sum(char.isdigit() for char in match_text) <= maximum

    return validator


@dataclass(frozen=True)
class Rule:
    """One regular-expression rule."""

    name: str  # short rule name, stored in Detection.raw_label
    category: str  # application category
    pattern: re.Pattern
    description: str
    validator: Callable[[str], bool] | None = None  # optional extra check


# Order matters: when two rules match overlapping text, the rule that comes
# first in this list wins (e.g. an IBAN is never also reported as an ID).
RULES: tuple[Rule, ...] = (
    Rule(
        name="email",
        category="EMAIL",
        pattern=re.compile(r"(?<![\w.+\-])[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}(?![\w\-])"),
        description="Standard email address: name@domain.tld",
    ),
    Rule(
        name="iban",
        category="IBAN",
        pattern=re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?(?![A-Za-z0-9])"),
        description="International bank account number (e.g. Saudi IBAN: SA + 22 characters)",
        validator=lambda text: 15 <= len(text.replace(" ", "")) <= 34 and sum(c.isdigit() for c in text) >= 10,
    ),
    Rule(
        name="credit_card",
        category="CREDIT_CARD",
        pattern=re.compile(NOT_AFTER_DIGIT_OR_WORD + r"[2-6](?:[ \-]?\d){12,18}" + NOT_BEFORE_DIGIT),
        description="13-19 digit payment card number (first digit 2-6) that passes the Luhn checksum",
        validator=_luhn_is_valid,
    ),
    Rule(
        name="credit_card_grouped",
        category="CREDIT_CARD",
        pattern=re.compile(
            NOT_AFTER_DIGIT_OR_WORD + r"(?:\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{4}|\d{4}[ \-]\d{6}[ \-]\d{5})" + NOT_BEFORE_DIGIT
        ),
        description="Digits written in card layout (4-4-4-4 or 4-6-5), even if the checksum is wrong",
    ),
    Rule(
        name="id_ssn_style",
        category="ID_LIKE",
        pattern=re.compile(NOT_AFTER_DIGIT_OR_WORD + r"\d{3}-\d{2}-\d{4}" + NOT_BEFORE_DIGIT),
        description="US social-security style: 123-45-6789",
    ),
    Rule(
        name="phone_saudi_mobile",
        category="PHONE",
        pattern=re.compile(
            NOT_AFTER_DIGIT_OR_WORD
            + r"(?:(?:\+|00)966" + SEP + r"(?:\(0\)|0)?" + SEP + r"5|05)\d" + SEP + r"\d{3}" + SEP + r"\d{4}"
            + NOT_BEFORE_DIGIT
        ),
        description="Saudi mobile: 05xxxxxxxx, +9665xxxxxxxx, 009665xxxxxxxx (spaces/dashes allowed)",
    ),
    Rule(
        name="phone_international_plus",
        category="PHONE",
        pattern=re.compile(r"(?<![\w.,/\-+])\+[1-9]\d{0,2}" + r"(?:[\s.\-]?" + GROUP + r"){2,5}" + NOT_BEFORE_DIGIT),
        description="International number that starts with + and a country code",
        validator=_digit_count_between(8, 15),
    ),
    Rule(
        name="phone_international_00",
        category="PHONE",
        pattern=re.compile(NOT_AFTER_DIGIT_OR_WORD + r"00[1-9]\d{0,2}" + r"(?:[\s.\-]" + GROUP + r"){2,5}" + NOT_BEFORE_DIGIT),
        description="International number that starts with 00; groups must be separated (0044 20 7946 0958)",
        validator=_digit_count_between(9, 15),
    ),
    Rule(
        name="phone_area_code",
        category="PHONE",
        pattern=re.compile(
            NOT_AFTER_DIGIT_OR_WORD + r"(?:\(\d{2,4}\)\s?\d{3,4}|\d{3}[\s.\-]\d{3})[\s.\-]\d{4}" + NOT_BEFORE_DIGIT
        ),
        description="Area-code style: (555) 123-4567, 555-123-4567, (31) 3452-6789",
    ),
    Rule(
        name="phone_national",
        category="PHONE",
        pattern=re.compile(
            NOT_AFTER_DIGIT_OR_WORD + r"0(?:[1-9]\d{8,9}|\d{1,4}(?:[\s.\-]\d{2,8}){1,4})" + NOT_BEFORE_DIGIT
        ),
        description="National number with a leading 0: 0594-737737, 07911 789 567, 0112345678",
        validator=_digit_count_between(9, 11),
    ),
    Rule(
        name="id_saudi_national",
        category="ID_LIKE",
        pattern=re.compile(NOT_AFTER_DIGIT_OR_WORD + r"[12]\d{9}" + NOT_BEFORE_DIGIT),
        description="Saudi-style national ID / Iqama: exactly 10 digits starting with 1 or 2",
    ),
    Rule(
        name="id_long_number",
        category="ID_LIKE",
        pattern=re.compile(NOT_AFTER_DIGIT_OR_WORD + r"\d{9,20}" + NOT_BEFORE_DIGIT),
        description="Generic long digit sequence (9-20 digits) such as account or ID numbers",
    ),
)


def normalize_digits(text: str) -> str:
    """Map Arabic-Indic digits to Western digits (length-preserving)."""
    return text.translate(_DIGIT_TABLE)


def detect_regex(text: str) -> list[Detection]:
    """Find emails, phone numbers, ID-like numbers, bank cards and IBANs.

    Args:
        text: The canonical text.

    Returns:
        Detections sorted by position. Offsets refer to ``text``.
    """
    if not text:
        return []

    search_text = normalize_digits(text)
    accepted: list[tuple[int, int, Rule]] = []

    for rule in RULES:
        for match in rule.pattern.finditer(search_text):
            start, end = match.span()
            if rule.validator and not rule.validator(match.group()):
                continue
            # Earlier rules have priority: skip matches that overlap them.
            if any(start < other_end and other_start < end for other_start, other_end, _ in accepted):
                continue
            accepted.append((start, end, rule))

    accepted.sort(key=lambda item: item[0])
    detections = [
        Detection(
            id=f"r-{number:04d}",
            category=rule.category,
            text=text[start:end],  # original characters, not the normalised ones
            start=start,
            end=end,
            source="regex",
            confidence=None,
            raw_label=rule.name,
        )
        for number, (start, end, rule) in enumerate(accepted, start=1)
    ]
    # Privacy rule: log counts only.
    logger.info("Regex layer found %d item(s)", len(detections))
    return detections
