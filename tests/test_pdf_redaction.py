"""Tests for true PDF redaction (fictional data only).

These tests use the regex layer only, so they run without the AI model.
"""

import pymupdf
import pytest

from src.file_processor import load_input
from src.redactor import RedactionError, detection_rectangles, redact_pdf, verify_pdf_redaction
from src.regex_detector import detect_regex
from src.synthetic_data import text_to_pdf

TEXT = (
    "Customer record\n"
    "Name: Sara Ali\n"
    "Mobile: 0551234567 (preferred)\n"
    "Email: sara.ali@example.com\n"
    "National ID: 1098765432\n"
    "Notes: The customer asked for a call back on Sunday."
)


@pytest.fixture()
def loaded_pdf(tmp_path):
    path = tmp_path / "record.pdf"
    text_to_pdf(TEXT, path, title="Customer record")
    return load_input(file=path)


def extract_text(pdf_bytes: bytes) -> str:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def test_selected_text_is_removed_from_the_pdf(loaded_pdf):
    detections = detect_regex(loaded_pdf.canonical_text)
    assert {d.category for d in detections} == {"PHONE", "EMAIL", "ID_LIKE"}

    redacted = redact_pdf(loaded_pdf, detections, [d.id for d in detections])
    remaining_text = extract_text(redacted)

    # The sensitive strings are really gone from the file (not just covered).
    assert "0551234567" not in remaining_text
    assert "sara.ali@example.com" not in remaining_text
    assert "1098765432" not in remaining_text
    assert "0551234567".encode() not in redacted


def test_neighbouring_text_is_kept(loaded_pdf):
    detections = detect_regex(loaded_pdf.canonical_text)
    remaining_text = extract_text(redact_pdf(loaded_pdf, detections, [d.id for d in detections], show_labels=False))

    for kept in ("Mobile:", "(preferred)", "Email:", "National ID:", "Name: Sara Ali", "call back on Sunday"):
        assert kept in remaining_text


def test_remaining_text_does_not_move(loaded_pdf):
    """Regression test: PyMuPDF 2.1 shifts the text that follows a redaction
    by one character width (a library bug). requirements.txt pins a version
    without that bug; this test fails if a broken version is installed."""

    def word_positions(pdf_bytes: bytes) -> dict[str, float]:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            return {word[4]: round(word[0], 1) for word in document[0].get_text("words")}

    detections = detect_regex(loaded_pdf.canonical_text)
    before = word_positions(loaded_pdf.pdf_bytes)
    after = word_positions(redact_pdf(loaded_pdf, detections, [d.id for d in detections], show_labels=False))

    assert after["(preferred)"] == before["(preferred)"]
    for word, x_position in after.items():
        assert before[word] == x_position


def test_only_selected_detections_are_redacted(loaded_pdf):
    detections = detect_regex(loaded_pdf.canonical_text)
    phone = next(d for d in detections if d.category == "PHONE")
    remaining_text = extract_text(redact_pdf(loaded_pdf, detections, [phone.id]))

    assert "0551234567" not in remaining_text
    assert "sara.ali@example.com" in remaining_text
    assert "1098765432" in remaining_text


def test_original_pdf_is_not_modified(loaded_pdf):
    original = bytes(loaded_pdf.pdf_bytes)
    detections = detect_regex(loaded_pdf.canonical_text)
    redact_pdf(loaded_pdf, detections, [d.id for d in detections])
    assert loaded_pdf.pdf_bytes == original
    assert "0551234567" in extract_text(loaded_pdf.pdf_bytes)


def test_metadata_is_cleared(loaded_pdf):
    detections = detect_regex(loaded_pdf.canonical_text)
    redacted = redact_pdf(loaded_pdf, detections, [d.id for d in detections])
    with pymupdf.open(stream=redacted, filetype="pdf") as document:
        assert not document.metadata.get("author")
        assert not document.metadata.get("title")


def test_verification_passes_after_redaction_and_fails_without_it(loaded_pdf):
    detections = detect_regex(loaded_pdf.canonical_text)
    selected = [d.id for d in detections]

    redacted = redact_pdf(loaded_pdf, detections, selected)
    report = verify_pdf_redaction(redacted, loaded_pdf, detections, selected)
    assert report == {"passed": True, "checked": 3, "failed_ids": []}

    # Checking the ORIGINAL file must fail: the text is still there.
    report = verify_pdf_redaction(loaded_pdf.pdf_bytes, loaded_pdf, detections, selected)
    assert report["passed"] is False
    assert len(report["failed_ids"]) == 3


def test_detection_spanning_two_lines_gives_two_rectangles(tmp_path):
    path = tmp_path / "wrapped.pdf"
    text_to_pdf("First line ends with Sara\nAli starts the second line.", path)
    loaded = load_input(file=path)
    start = loaded.canonical_text.index("Sara")
    end = loaded.canonical_text.index("Ali") + 3
    rectangles = detection_rectangles(loaded.char_boxes, start, end)
    assert len(rectangles) == 2
    assert rectangles[0][1].y0 < rectangles[1][1].y0


def test_redact_pdf_requires_a_pdf_input():
    text_input = load_input(text="Mobile 0551234567")
    with pytest.raises(RedactionError):
        redact_pdf(text_input, detect_regex(text_input.canonical_text), [])
