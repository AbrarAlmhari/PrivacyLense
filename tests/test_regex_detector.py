"""Tests for the rule-based layer. All numbers and addresses are fictional."""

import pytest

from src.regex_detector import RULES, detect_regex, normalize_digits


def found(text: str) -> list[tuple[str, str]]:
    """Helper: return (category, matched text) pairs."""
    return [(d.category, d.text) for d in detect_regex(text)]


# ------------------------------------------------------------ positive cases
@pytest.mark.parametrize(
    "text, category, expected",
    [
        ("Write to sara.ali@example.com today", "EMAIL", "sara.ali@example.com"),
        ("Contact: first_last+tag@mail.example.org.", "EMAIL", "first_last+tag@mail.example.org"),
        ("Mobile 0551234567", "PHONE", "0551234567"),
        ("Mobile 055 123 4567", "PHONE", "055 123 4567"),
        ("Call +966551234567 now", "PHONE", "+966551234567"),
        ("Call +966 55 123 4567 now", "PHONE", "+966 55 123 4567"),
        ("Call 00966551234567 now", "PHONE", "00966551234567"),
        ("UK office +44 20 7946 0958", "PHONE", "+44 20 7946 0958"),
        ("US office (555) 010-4477", "PHONE", "(555) 010-4477"),
        ("US office 555-010-4477", "PHONE", "555-010-4477"),
        ("Landline 011 465 7788", "PHONE", "011 465 7788"),
        ("Tel. 0594-737737", "PHONE", "0594-737737"),
        ("Tel. 07911 789 567", "PHONE", "07911 789 567"),
        ("Tel. (31) 3452-6789", "PHONE", "(31) 3452-6789"),
        ("Dial 0044 20 7946 0958", "PHONE", "0044 20 7946 0958"),
        ("Card 5342 3456 7890 1234 (wrong checksum)", "CREDIT_CARD", "5342 3456 7890 1234"),
        ("Customer id 0028347916", "ID_LIKE", "0028347916"),
        ("National ID 1098765432", "ID_LIKE", "1098765432"),
        ("Iqama: 2345678901.", "ID_LIKE", "2345678901"),
        ("SSN 123-45-6789", "ID_LIKE", "123-45-6789"),
        ("Account 400123456789", "ID_LIKE", "400123456789"),
        ("Card 4111 1111 1111 1111", "CREDIT_CARD", "4111 1111 1111 1111"),
        ("Card 4111-1111-1111-1111", "CREDIT_CARD", "4111-1111-1111-1111"),
        ("IBAN SA03 8000 0000 6080 1016 7519", "IBAN", "SA03 8000 0000 6080 1016 7519"),
        ("IBAN SA0380000000608010167519", "IBAN", "SA0380000000608010167519"),
    ],
)
def test_positive_matches(text, category, expected):
    assert (category, expected) in found(text)


# ------------------------------------------------------------ negative cases
@pytest.mark.parametrize(
    "text",
    [
        "The meeting is on 2026-11-16 at 10:30.",
        "Deadline: 16/11/2026.",
        "The total is 1,250.00 SAR.",
        "Price: $12345.67",
        "Order #48213 was shipped.",
        "Room 204, Building 7",
        "It weighs 3.14159 kg and costs 99.99.",
        "See page 123 of the 2026 report.",
        "Version 10.0.22631 of the software.",
        "not-an-email@ and also @nothing",
        "",
    ],
)
def test_negative_cases(text):
    assert found(text) == []


def test_unformatted_card_must_pass_luhn_check():
    # 16 digits in a row with a wrong checksum are not called a card;
    # they are still reported as a long ID-like number.
    assert found("Number 4111111111111112") == [("ID_LIKE", "4111111111111112")]
    assert found("Number 4111111111111111") == [("CREDIT_CARD", "4111111111111111")]


# ------------------------------------------------------------------- offsets
def test_offsets_point_to_the_matched_text():
    text = "Email a@b.co, phone 0551234567, ID 1098765432."
    detections = detect_regex(text)
    assert len(detections) == 3
    for detection in detections:
        assert text[detection.start : detection.end] == detection.text
        assert detection.source == "regex"
        assert detection.id.startswith("r-")


def test_detections_are_sorted_and_do_not_overlap():
    text = "IBAN SA0380000000608010167519 and card 4111111111111111 and 0551234567"
    detections = detect_regex(text)
    assert [d.category for d in detections] == ["IBAN", "CREDIT_CARD", "PHONE"]
    for first, second in zip(detections, detections[1:]):
        assert first.end <= second.start


def test_arabic_indic_digits_are_supported_without_shifting_offsets():
    text = "رقم الجوال ٠٥٥١٢٣٤٥٦٧"
    assert len(normalize_digits(text)) == len(text)
    detections = detect_regex(text)
    assert len(detections) == 1
    assert detections[0].category == "PHONE"
    assert detections[0].text == "٠٥٥١٢٣٤٥٦٧"  # original characters are kept


def test_every_rule_is_documented():
    names = [rule.name for rule in RULES]
    assert len(names) == len(set(names))
    assert all(rule.description for rule in RULES)


def test_closing_bracket_after_a_phone_number_is_not_included():
    assert found("Emergency contact: Hassan (+44 20 7946 0526) today") == [("PHONE", "+44 20 7946 0526")]
    assert found("Call +1 (555) 010-9349.") == [("PHONE", "+1 (555) 010-9349")]


def test_saudi_number_with_00_prefix_is_a_phone_not_a_card():
    # 14 digits can pass the card checksum by chance; cards never start with 0.
    assert found("Reach me on 00966573139096 please") == [("PHONE", "00966573139096")]
