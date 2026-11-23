"""Tests for merging, risk scoring and text redaction (fictional data only)."""

import pytest

from src.config import load_config
from src.redactor import redact
from src.risk_analyzer import merge_detections, propagate_detections, score
from src.schemas import Detection


@pytest.fixture(scope="module")
def config():
    return load_config()


def make(category: str, text: str, start: int = 0, source: str = "regex", number: int = 1) -> Detection:
    prefix = "r" if source == "regex" else "n"
    return Detection(f"{prefix}-{number:04d}", category, text, start, start + len(text), source)


# ------------------------------------------------------------------- scoring
def test_empty_input_scores_zero(config):
    result = score([], config)
    assert result.score == 0
    assert result.level == "Low"
    assert result.breakdown == {}


def test_single_item_gets_base_points(config):
    result = score([make("EMAIL", "a@b.co")], config)
    assert result.score == 15
    assert result.level == "Low"


def test_worked_example_from_the_plan(config):
    # one ID (30) + one phone (20) + two different names (20 + 5) = 75 -> High
    detections = [
        make("ID_LIKE", "1098765432", 0, number=1),
        make("PHONE", "0551234567", 20, number=2),
        make("PERSON", "Sara Ali", 40, "ner", 1),
        make("PERSON", "Omar Zaid", 60, "ner", 2),
    ]
    result = score(detections, config)
    assert result.score == 75
    assert result.level == "High"


def test_repeated_items_count_once(config):
    repeated = [make("EMAIL", "a@b.co", i * 10, number=i) for i in range(5)]
    assert score(repeated, config).score == 15
    # Different formatting of the same phone number is still one item.
    phones = [make("PHONE", "055 123 4567", 0, number=1), make("PHONE", "0551234567", 30, number=2)]
    assert score(phones, config).breakdown["PHONE"]["unique_items"] == 1


def test_category_cap_is_two_times_base(config):
    many = [make("EMAIL", f"user{i}@b.co", i * 20, number=i) for i in range(20)]
    result = score(many, config)
    assert result.breakdown["EMAIL"]["points"] == 30  # 2 x 15
    assert result.breakdown["EMAIL"]["capped"] is True


def test_total_is_capped_at_100(config):
    detections = []
    offset = 0
    for category in ("ID_LIKE", "CREDIT_CARD", "IBAN", "PHONE", "PERSON", "ADDRESS", "EMAIL"):
        for i in range(6):
            detections.append(make(category, f"{category}-{i}", offset, number=len(detections)))
            offset += 20
    result = score(detections, config)
    assert result.score == 100
    assert result.level == "Critical"


@pytest.mark.parametrize("value, level", [(0, "Low"), (20, "Low"), (21, "Medium"), (50, "Medium"), (51, "High"), (75, "High"), (76, "Critical"), (100, "Critical")])
def test_band_boundaries(config, value, level):
    from src.risk_analyzer import _level_for

    assert _level_for(value, config["risk"]["bands"]) == level


# ------------------------------------------------------------------- merging
def test_regex_wins_when_regex_and_ner_overlap():
    text = "Mail sara.ali@example.com"
    regex = [make("EMAIL", "sara.ali@example.com", 5)]
    ner = [make("PERSON", "sara.ali", 5, "ner")]
    notes: list[str] = []
    merged = merge_detections(regex, ner, notes)
    assert [d.category for d in merged] == ["EMAIL"]
    assert notes and "overlapping" in notes[0]
    assert text[merged[0].start : merged[0].end] == merged[0].text


def test_merge_keeps_non_overlapping_items_sorted():
    regex = [make("PHONE", "0551234567", 30)]
    ner = [make("PERSON", "Sara Ali", 0, "ner")]
    merged = merge_detections(regex, ner)
    assert [d.category for d in merged] == ["PERSON", "PHONE"]


def test_propagation_finds_repeated_names():
    text = "Sara Ali wrote the report. Later, Sara Ali signed it. Sarah did not."
    first = make("PERSON", "Sara Ali", 0, "ner")
    result = propagate_detections(text, [first])
    assert len(result) == 2
    assert result[1].source == "propagated"
    assert text[result[1].start : result[1].end] == "Sara Ali"


# ----------------------------------------------------------------- redaction
def test_redaction_replaces_only_selected_items():
    text = "Call Sara Ali on 0551234567 or mail sara@example.com."
    detections = [
        make("PERSON", "Sara Ali", text.index("Sara Ali"), "ner", 1),
        make("PHONE", "0551234567", text.index("0551234567"), number=1),
        make("EMAIL", "sara@example.com", text.index("sara@example.com"), number=2),
    ]
    sanitized = redact(text, detections, {"n-0001", "r-0001"})
    assert sanitized == "Call [REDACTED-PERSON] on [REDACTED-PHONE] or mail sara@example.com."


def test_selected_strings_are_absent_from_output():
    text = "ID 1098765432 belongs to Omar Zaid. Omar Zaid lives in Dammam."
    detections = [
        make("ID_LIKE", "1098765432", 3, number=1),
        make("PERSON", "Omar Zaid", text.index("Omar Zaid"), "ner", 1),
        make("PERSON", "Omar Zaid", text.rindex("Omar Zaid"), "ner", 2),
    ]
    sanitized = redact(text, detections, [d.id for d in detections])
    assert "1098765432" not in sanitized
    assert "Omar Zaid" not in sanitized
    assert "Dammam" in sanitized  # not selected, so it stays


def test_redaction_with_nothing_selected_returns_original_text():
    text = "Nothing to hide here."
    assert redact(text, [], set()) == text
