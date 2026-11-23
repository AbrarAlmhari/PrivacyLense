"""End-to-end tests: load -> scan -> score -> redact -> export.

The regex-only tests always run. Tests marked ``needs_model`` use the
fine-tuned AI model and are skipped when it has not been trained yet.
"""

import json

import pymupdf
import pytest

from src.config import load_config, resolve_path
from src.evaluation import character_recall, evaluate_entities, precision_recall_f1
from src.file_processor import load_input
from src.pipeline import scan
from src.redactor import redact, redact_pdf, verify_pdf_redaction
from src.report_generator import build_summary, detections_to_csv, summary_to_json
from src.synthetic_data import TEMPLATES, fill_template, generate_documents, text_to_pdf

CONFIG = load_config()
needs_model = pytest.mark.skipif(
    not (resolve_path(CONFIG["model"]["output_dir"]) / "config.json").exists(),
    reason="Fine-tuned model not found. Train it with scripts/02_train_model.py",
)

DOCUMENT = (
    "Dear Sara Al-Qahtani,\n\n"
    "Your appointment at Desert Bloom Technologies in Riyadh is confirmed. "
    "We registered your national ID 1098765432 and your mobile 0551234567. "
    "Please confirm by email to hr.team@example.com.\n\n"
    "Regards,\nOmar Haddad"
)


# --------------------------------------------------------- regex-only workflow
def test_text_workflow_without_model():
    loaded = load_input(text=DOCUMENT)
    result = scan(loaded, CONFIG, use_ner=False)

    assert {d.category for d in result.detections} == {"ID_LIKE", "PHONE", "EMAIL"}
    assert result.risk.score == 65  # 30 + 20 + 15
    assert result.risk.level == "High"

    selected = {d.id for d in result.detections}
    sanitized = redact(loaded.canonical_text, result.detections, selected)
    for detection in result.detections:
        assert detection.text not in sanitized
    assert "[REDACTED-PHONE]" in sanitized


def test_summary_does_not_leak_sensitive_text_by_default():
    loaded = load_input(text=DOCUMENT)
    result = scan(loaded, CONFIG, use_ner=False)
    selected = {d.id for d in result.detections}

    summary = build_summary(loaded, result.detections, result.risk, selected, result.notes)
    summary_json = summary_to_json(summary)
    for detection in result.detections:
        assert detection.text not in summary_json
        assert detection.text not in detections_to_csv(summary)

    parsed = json.loads(summary_json)
    assert parsed["detections"]["total"] == 3
    assert parsed["risk"]["score"] == 65
    assert parsed["redaction"]["selected"] == 3
    assert parsed["limitations"]

    with_text = build_summary(loaded, result.detections, result.risk, selected, include_sensitive_text=True)
    assert "0551234567" in summary_to_json(with_text)


def test_document_without_sensitive_data_scores_zero():
    result = scan(load_input(text="The library is closed on Friday. Books are extended by 7 days."), CONFIG, use_ner=False)
    assert result.detections == []
    assert result.risk.score == 0
    assert result.risk.level == "Low"


# ----------------------------------------------------------- synthetic data
def test_synthetic_gold_labels_point_to_the_right_text():
    for document in generate_documents(count=len(TEMPLATES) * 2, seed=7):
        for entity in document["entities"]:
            value = document["text"][entity["start"] : entity["end"]]
            assert value.strip() == value and value
        starts = [entity["start"] for entity in document["entities"]]
        assert starts == sorted(starts)


def test_synthetic_generation_is_reproducible():
    assert generate_documents(5, seed=1) == generate_documents(5, seed=1)
    assert generate_documents(5, seed=1) != generate_documents(5, seed=2)


def test_same_slot_gives_same_value():
    import random

    text, entities = fill_template("{PERSON_1} met {PERSON_2}. Later {PERSON_1} left.", random.Random(3))
    names = [text[e["start"] : e["end"]] for e in entities]
    assert names[0] == names[2]


# ---------------------------------------------------------------- evaluation
def test_metric_helpers():
    assert precision_recall_f1(8, 2, 2)["f1"] == 0.8
    assert precision_recall_f1(0, 0, 0)["f1"] == 0.0

    gold = [{("PERSON", 0, 8), ("ORG", 20, 30)}]
    predicted = [{("PERSON", 0, 8), ("ORG", 20, 28), ("LOCATION", 40, 45)}]
    scores = evaluate_entities(gold, predicted, ("PERSON", "ORG", "LOCATION"))
    assert scores["per_category"]["PERSON"]["f1"] == 1.0
    assert scores["per_category"]["ORG"]["f1"] == 0.0  # boundary differs -> strict miss
    assert scores["per_category"]["LOCATION"]["false_positives"] == 1
    assert scores["overall"]["true_positives"] == 1
    # ...but 8 of the 10 ORG characters are still hidden.
    assert character_recall(gold, predicted, ("PERSON", "ORG"))["ORG"] == 0.8


# ---------------------------------------------------- full workflow with model
@needs_model
def test_full_pdf_workflow_with_ai_model(tmp_path):
    path = tmp_path / "letter.pdf"
    text_to_pdf(DOCUMENT, path)
    loaded = load_input(file=path)
    result = scan(loaded, CONFIG)

    found = {(d.category, d.text) for d in result.detections}
    for expected in [
        ("PERSON", "Sara Al-Qahtani"),
        ("PERSON", "Omar Haddad"),
        ("LOCATION", "Riyadh"),
        ("ID_LIKE", "1098765432"),
        ("PHONE", "0551234567"),
        ("EMAIL", "hr.team@example.com"),
    ]:
        assert expected in found
    assert result.risk.level in ("High", "Critical")
    for detection in result.detections:
        assert loaded.canonical_text[detection.start : detection.end] == detection.text

    selected = {d.id for d in result.detections}
    redacted = redact_pdf(loaded, result.detections, selected)
    assert verify_pdf_redaction(redacted, loaded, result.detections, selected)["passed"]

    with pymupdf.open(stream=redacted, filetype="pdf") as document:
        remaining = "\n".join(page.get_text() for page in document)
    for sensitive in ("Sara Al-Qahtani", "Omar Haddad", "1098765432", "0551234567", "hr.team@example.com"):
        assert sensitive not in remaining
    assert "Your appointment at" in remaining


@needs_model
def test_repeated_names_are_all_detected():
    text = "Khalid Alghamdi approved the budget.\n\nItem 4: budget follow-up - Khalid Alghamdi."
    result = scan(load_input(text=text), CONFIG)
    people = [d for d in result.detections if d.category == "PERSON" and d.text == "Khalid Alghamdi"]
    assert len(people) == 2
    sanitized = redact(text, result.detections, {d.id for d in result.detections})
    assert "Khalid Alghamdi" not in sanitized
