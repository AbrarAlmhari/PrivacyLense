"""Tests for loading, validation and cleaning (valid, empty, malformed, unsupported)."""

import pymupdf
import pytest

from src.file_processor import (
    CorruptFileError,
    EmptyInputError,
    FileTooLargeError,
    ScannedPdfError,
    UnsupportedFileError,
    clean_text,
    clean_text_with_map,
    load_input,
)
from src.synthetic_data import make_scanned_pdf, text_to_pdf


def make_pdf_bytes(text: str, tmp_path) -> bytes:
    path = tmp_path / "document.pdf"
    text_to_pdf(text, path)
    return path.read_bytes()


# ------------------------------------------------------------------ cleaning
def test_cleaning_collapses_whitespace_and_blank_lines():
    raw = "  Hello \t  world \r\n\r\n\r\n\r\nSecond line​  \n"
    assert clean_text(raw) == "Hello world\n\nSecond line"


def test_cleaning_keeps_normal_text_unchanged():
    text = "Dear Sara,\n\nYour ID is 1098765432.\nRegards"
    assert clean_text(text) == text


def test_index_map_points_back_to_the_raw_text():
    raw = "A\t\tB \n\n\n C​D"
    cleaned, index_map = clean_text_with_map(raw)
    assert len(cleaned) == len(index_map)
    for clean_index, raw_index in enumerate(index_map):
        # Tabs became spaces; every other character is identical.
        assert cleaned[clean_index] == raw[raw_index].replace("\t", " ")


def test_cleaning_is_idempotent():
    raw = "Line one  \n\n\n\n  Line   two\t!"
    once = clean_text(raw)
    assert clean_text(once) == once


# ---------------------------------------------------------------- text input
def test_pasted_text_is_loaded_with_statistics():
    result = load_input(text="Hello   world.\nSecond line.")
    assert result.source_type == "text"
    assert result.canonical_text == "Hello world.\nSecond line."
    assert result.stats["words"] == 4
    assert result.stats["lines"] == 2


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_text_is_rejected(text):
    with pytest.raises(EmptyInputError):
        load_input(text=text)


def test_no_input_at_all_is_rejected():
    with pytest.raises(EmptyInputError):
        load_input()


# ---------------------------------------------------------------- file input
def test_txt_file_is_loaded():
    result = load_input(file="Name: Sara\r\nCity: Dammam".encode("utf-8"), file_name="note.txt")
    assert result.source_type == "txt"
    assert result.canonical_text == "Name: Sara\nCity: Dammam"


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedFileError):
        load_input(file=b"data", file_name="photo.png")


def test_empty_file_is_rejected():
    with pytest.raises(EmptyInputError):
        load_input(file=b"", file_name="empty.txt")


def test_binary_data_in_txt_is_rejected():
    with pytest.raises(CorruptFileError):
        load_input(file=b"\x00\x01\x02\x00binary\x00", file_name="strange.txt")


def test_fake_pdf_is_rejected():
    with pytest.raises(CorruptFileError):
        load_input(file=b"this is not a pdf", file_name="fake.pdf")


def test_file_size_limit_is_enforced():
    config = {"input": {"max_file_mb": 0.0001, "max_pdf_pages": 10, "max_characters": 1000}}
    with pytest.raises(FileTooLargeError):
        load_input(file=b"x" * 5000, file_name="big.txt", config=config)


def test_missing_file_path_is_reported(tmp_path):
    with pytest.raises(CorruptFileError):
        load_input(file=tmp_path / "does_not_exist.txt")


# ----------------------------------------------------------------------- PDF
def test_pdf_text_and_character_positions(tmp_path):
    text = "Name: Sara Ali\nMobile: 0551234567\n\nSecond paragraph here."
    result = load_input(file=make_pdf_bytes(text, tmp_path), file_name="document.pdf")

    assert result.source_type == "pdf"
    assert "Sara Ali" in result.canonical_text
    assert "0551234567" in result.canonical_text
    assert result.stats["pages"] == 1
    # One position entry per character of the canonical text.
    assert len(result.char_boxes) == len(result.canonical_text)
    start = result.canonical_text.index("Sara")
    assert all(result.char_boxes[i] is not None for i in range(start, start + 4))
    assert result.char_boxes[start].page == 0


def test_multi_page_pdf(tmp_path):
    text = "\n".join(f"Line number {i} of the long document." for i in range(120))
    result = load_input(file=make_pdf_bytes(text, tmp_path), file_name="long.pdf")
    assert result.stats["pages"] >= 2
    last_line_start = result.canonical_text.index("Line number 119")
    assert result.char_boxes[last_line_start].page >= 1


def test_scanned_pdf_is_reported_clearly(tmp_path):
    path = tmp_path / "scan.pdf"
    make_scanned_pdf(path)
    with pytest.raises(ScannedPdfError) as error:
        load_input(file=path)
    assert "scanned" in str(error.value).lower()


def test_password_protected_pdf_is_rejected(tmp_path):
    path = tmp_path / "locked.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "secret")
    document.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    document.close()
    with pytest.raises(CorruptFileError) as error:
        load_input(file=path)
    assert "password" in str(error.value).lower()
