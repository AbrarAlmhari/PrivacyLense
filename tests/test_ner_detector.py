"""Tests for the AI layer.

The first group tests the decoding logic with hand-made token labels, so it
runs without any model. The second group needs the fine-tuned model in
``models/privacylens-ner`` and is skipped automatically when it is missing.
"""

import pytest

from src.config import load_config, resolve_path
from src.ner_detector import _expand_to_word_boundaries, detect_ner, group_tokens_into_entities

MODEL_DIR = resolve_path(load_config()["model"]["output_dir"])
needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "config.json").exists(),
    reason="Fine-tuned model not found. Train it with scripts/02_train_model.py",
)


# ------------------------------------------------- decoding logic (no model)
def test_bio_tokens_are_glued_into_one_entity():
    text = "Dear Sara Ali, welcome"
    tokens = [(0, 4, "O", 0.99), (5, 9, "B-PERSON", 0.9), (10, 13, "I-PERSON", 0.8), (13, 14, "O", 0.99)]
    entities = group_tokens_into_entities(text, tokens)
    assert len(entities) == 1
    assert (entities[0]["type"], entities[0]["start"], entities[0]["end"]) == ("PERSON", 5, 13)
    assert entities[0]["probabilities"] == [0.9, 0.8]


def test_new_b_label_starts_a_new_entity():
    text = "Sara Ali Omar Zaid"
    tokens = [(0, 4, "B-PERSON", 0.9), (5, 8, "I-PERSON", 0.9), (9, 13, "B-PERSON", 0.9), (14, 18, "I-PERSON", 0.9)]
    entities = group_tokens_into_entities(text, tokens)
    assert [(e["start"], e["end"]) for e in entities] == [(0, 8), (9, 18)]


def test_different_types_are_not_merged():
    text = "Sara Aramco"
    tokens = [(0, 4, "B-PERSON", 0.9), (5, 11, "I-ORG", 0.9)]
    entities = group_tokens_into_entities(text, tokens)
    assert [e["type"] for e in entities] == ["PERSON", "ORG"]


def test_entities_are_expanded_to_whole_words():
    text = "Meet Abdulrahman today"
    # The model only tagged the middle of the word: "dulrah"
    assert _expand_to_word_boundaries(text, 7, 13) == (5, 16)


def test_empty_text_returns_no_detections():
    assert detect_ner("") == []
    assert detect_ner("   \n ") == []


# ------------------------------------------------------- real model required
@needs_model
def test_model_finds_people_organisations_and_places():
    text = (
        "Dear Noura Alharbi,\n"
        "Thank you for joining Acme Corporation. Your manager John Smith will meet you "
        "at our office in Riyadh on Sunday."
    )
    detections = detect_ner(text)
    found = {(d.category, d.text) for d in detections}

    assert ("PERSON", "Noura Alharbi") in found
    assert ("PERSON", "John Smith") in found
    assert ("ORG", "Acme Corporation") in found
    assert ("LOCATION", "Riyadh") in found


@needs_model
def test_offsets_and_fields_follow_the_contract():
    text = "Please forward the contract to Fatimah Al-Dossary in Dammam."
    detections = detect_ner(text)
    assert detections, "expected at least one detection"
    for detection in detections:
        assert text[detection.start : detection.end] == detection.text
        assert detection.source == "ner"
        assert detection.id.startswith("n-")
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.category in ("PERSON", "ORG", "LOCATION", "ADDRESS")
    assert detections == sorted(detections, key=lambda d: d.start)


@needs_model
def test_text_without_entities_stays_clean():
    text = "The library will be closed for maintenance. All borrowed books are extended by seven days."
    assert detect_ner(text) == []


@needs_model
def test_long_text_is_processed_in_windows():
    # ~1,200 words: far more than the 256 tokens that fit into one window.
    filler = "This sentence is ordinary filler text without any private details. " * 120
    text = filler + "The final approval was signed by Khalid Alghamdi."
    detections = detect_ner(text)
    assert any(d.text == "Khalid Alghamdi" and d.category == "PERSON" for d in detections)
    last = detections[-1]
    assert text[last.start : last.end] == last.text
