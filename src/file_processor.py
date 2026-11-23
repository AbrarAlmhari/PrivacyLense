"""Load, validate and clean user input (pasted text, TXT files, PDFs).

Workflow steps covered here: **Load -> Validate -> Clean**.

The most important idea in this module is the *canonical text* rule
(project plan, Section 4.1):

* All cleaning that changes the length of the text happens **once**, here.
* Every other module (regex, NER, scoring, redaction, export) works on the
  cleaned "canonical text" and its character offsets.

For PDFs we additionally remember *where every character sits on the page*
(``char_boxes``). That is what later lets us draw a redaction box exactly on
top of a detected name or phone number.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from src.config import load_config
from src.schemas import CharBox, LoadResult

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".pdf")

# Invisible characters that PDF/text extraction often leaves behind.
_INVISIBLE_CHARS = {
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "⁠",  # word joiner
    "﻿",  # byte-order mark
    "­",  # soft hyphen
}
# Characters that should simply become a normal space.
_SPACE_LIKE_CHARS = {"\t", " ", " ", " ", " ", " ", " ", "\x0b", "\x0c"}


# --------------------------------------------------------------------------
# Exceptions: one class per problem, so the app can show a clear message.
# --------------------------------------------------------------------------
class InputError(Exception):
    """Base class for every problem with the user's input."""


class EmptyInputError(InputError):
    """No text was provided, or the file contains no text."""


class UnsupportedFileError(InputError):
    """The file type is not supported."""


class FileTooLargeError(InputError):
    """The file or text is larger than the configured limit."""


class CorruptFileError(InputError):
    """The file could not be opened or decoded."""


class ScannedPdfError(InputError):
    """The PDF has no extractable text (it is probably a scanned image)."""


# --------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------
def clean_text_with_map(raw_text: str) -> tuple[str, list[int]]:
    """Clean ``raw_text`` and remember where each kept character came from.

    Cleaning rules (they never change the meaning of the text):

    1. Windows/Mac line endings become ``\\n``.
    2. Invisible characters (zero-width spaces, soft hyphens, ...) are removed.
    3. Tabs and special spaces become a normal space.
    4. Repeated spaces collapse into one; spaces around line breaks are removed.
    5. Three or more line breaks collapse into two (one blank line).
    6. Leading and trailing whitespace is removed.

    Returns:
        ``(clean_text, index_map)`` where ``index_map[i]`` is the position in
        ``raw_text`` of the character ``clean_text[i]``.
    """
    clean_chars: list[str] = []
    index_map: list[int] = []

    for raw_index, char in enumerate(raw_text):
        if char == "\r":
            # "\r\n" -> "\n" (skip the "\r"); a lonely "\r" becomes "\n".
            next_char = raw_text[raw_index + 1] if raw_index + 1 < len(raw_text) else ""
            if next_char == "\n":
                continue
            char = "\n"
        if char in _INVISIBLE_CHARS:
            continue
        if char in _SPACE_LIKE_CHARS:
            char = " "
        elif char != "\n" and not char.isprintable():
            continue  # other control characters

        previous = clean_chars[-1] if clean_chars else ""
        if char == " ":
            # No space at the start, after another space, or after a newline.
            if previous in ("", " ", "\n"):
                continue
        elif char == "\n":
            # Remove a space that sits right before the line break.
            if previous == " ":
                clean_chars.pop()
                index_map.pop()
            # No newline at the start and never more than two in a row.
            if not clean_chars:
                continue
            if len(clean_chars) >= 2 and clean_chars[-1] == "\n" and clean_chars[-2] == "\n":
                continue

        clean_chars.append(char)
        index_map.append(raw_index)

    # Remove whitespace left at the very end.
    while clean_chars and clean_chars[-1] in (" ", "\n"):
        clean_chars.pop()
        index_map.pop()

    return "".join(clean_chars), index_map


def clean_text(raw_text: str) -> str:
    """Return only the cleaned (canonical) text."""
    return clean_text_with_map(raw_text)[0]


# --------------------------------------------------------------------------
# Load + Validate
# --------------------------------------------------------------------------
def load_input(
    text: str | None = None,
    file: bytes | str | Path | None = None,
    file_name: str | None = None,
    config: dict | None = None,
) -> LoadResult:
    """Load pasted text **or** a TXT/PDF file and return the canonical text.

    Args:
        text: Text pasted by the user.
        file: File content as bytes, or a path to a file on disk.
        file_name: Original file name (needed when ``file`` is bytes, because
            the extension tells us the file type).
        config: Optional configuration dictionary (defaults to config.yaml).

    Raises:
        InputError: (or a subclass) with a message that is safe to show to
            the user.
    """
    config = config or load_config()
    limits = config["input"]

    if file is not None:
        file_bytes, file_name = _read_file_bytes(file, file_name)
        extension = Path(file_name).suffix.lower()
        _validate_file(file_bytes, extension, limits)
        if extension == ".pdf":
            result = _load_pdf(file_bytes, file_name, limits)
        else:
            result = _load_txt(file_bytes, file_name)
    elif text is not None:
        result = LoadResult(canonical_text=clean_text(text), source_type="text")
    else:
        raise EmptyInputError("Please paste some text or upload a TXT/PDF file.")

    if not result.canonical_text:
        raise EmptyInputError("The input does not contain any readable text.")
    if len(result.canonical_text) > limits["max_characters"]:
        raise FileTooLargeError(
            f"The text has {len(result.canonical_text):,} characters. "
            f"The limit is {limits['max_characters']:,}."
        )

    result.stats.update(_text_statistics(result.canonical_text))
    # Privacy rule: log counts only, never the text itself.
    logger.info(
        "Loaded %s input: %d characters, %d words",
        result.source_type,
        result.stats["characters"],
        result.stats["words"],
    )
    return result


def _read_file_bytes(file: bytes | str | Path, file_name: str | None) -> tuple[bytes, str]:
    """Return ``(bytes, name)`` for a path or for already-loaded bytes."""
    if isinstance(file, (bytes, bytearray)):
        if not file_name:
            raise UnsupportedFileError("A file name is required to detect the file type.")
        return bytes(file), file_name

    path = Path(file)
    try:
        return path.read_bytes(), file_name or path.name
    except FileNotFoundError as error:
        raise CorruptFileError(f"File not found: {path.name}") from error
    except OSError as error:
        raise CorruptFileError(f"The file could not be read: {path.name}") from error


def _validate_file(file_bytes: bytes, extension: str, limits: dict) -> None:
    """Check file type, size and emptiness before doing any real work."""
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            "Please upload a .txt file or a text-based .pdf file."
        )
    if len(file_bytes) == 0:
        raise EmptyInputError("The uploaded file is empty.")

    max_bytes = limits["max_file_mb"] * 1024 * 1024
    if len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise FileTooLargeError(
            f"The file is {size_mb:.1f} MB. The limit is {limits['max_file_mb']} MB."
        )
    if extension == ".pdf" and not file_bytes.lstrip().startswith(b"%PDF"):
        raise CorruptFileError("This file has a .pdf extension but is not a valid PDF.")


def _load_txt(file_bytes: bytes, file_name: str) -> LoadResult:
    """Decode a TXT file. UTF-8 is tried first, then common fallbacks."""
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            raw_text = file_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise CorruptFileError("The text file could not be decoded. Please save it as UTF-8.")

    if "\x00" in raw_text:
        raise CorruptFileError("This does not look like a plain text file.")

    result = LoadResult(canonical_text=clean_text(raw_text), source_type="txt", file_name=file_name)
    result.stats["encoding"] = encoding
    return result


def _load_pdf(file_bytes: bytes, file_name: str, limits: dict) -> LoadResult:
    """Extract text *and character positions* from a text-based PDF."""
    try:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as error:  # PyMuPDF raises several different error types
        raise CorruptFileError("The PDF could not be opened. It may be damaged.") from error

    with document:
        if document.needs_pass:
            raise CorruptFileError("The PDF is password-protected. Please remove the password first.")
        if document.page_count == 0:
            raise EmptyInputError("The PDF has no pages.")
        if document.page_count > limits["max_pdf_pages"]:
            raise FileTooLargeError(
                f"The PDF has {document.page_count} pages. The limit is {limits['max_pdf_pages']}."
            )

        raw_chars: list[str] = []
        raw_boxes: list[CharBox | None] = []
        pages_without_text = 0
        image_count = 0

        for page_number, page in enumerate(document):
            chars_before = len(raw_chars)
            image_count += len(page.get_images())
            _extract_page_chars(page, page_number, raw_chars, raw_boxes)
            if len(raw_chars) == chars_before:
                pages_without_text += 1
            # Blank line between pages.
            raw_chars.extend("\n\n")
            raw_boxes.extend([None, None])

        page_count = document.page_count

    raw_text = "".join(raw_chars)
    canonical_text, index_map = clean_text_with_map(raw_text)

    if not canonical_text:
        raise ScannedPdfError(
            "No text could be extracted from this PDF. It is probably a scanned "
            "document (images of pages). PrivacyLens supports text-based PDFs only."
        )

    result = LoadResult(
        canonical_text=canonical_text,
        source_type="pdf",
        file_name=file_name,
        char_boxes=[raw_boxes[raw_index] for raw_index in index_map],
        pdf_bytes=file_bytes,
    )
    result.stats.update({"pages": page_count, "pages_without_text": pages_without_text, "images": image_count})

    if pages_without_text:
        result.warnings.append(
            f"{pages_without_text} of {page_count} page(s) have no extractable text "
            "(possibly scanned). Those pages were NOT analysed."
        )
    if image_count:
        result.warnings.append(
            f"The PDF contains {image_count} image(s). Text inside images is NOT analysed."
        )
    return result


def _extract_page_chars(
    page: pymupdf.Page,
    page_number: int,
    raw_chars: list[str],
    raw_boxes: list[CharBox | None],
) -> None:
    """Append every character of one PDF page (with its position) to the lists.

    PyMuPDF's "rawdict" output is organised as blocks -> lines -> spans ->
    characters, and each character comes with its bounding box.
    """
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    blocks = page.get_text("rawdict", flags=flags, sort=True)["blocks"]
    line_id = 0

    for block in blocks:
        if block.get("type") != 0:  # 0 = text block, 1 = image block
            continue
        previous_line_bbox = None
        for line in block["lines"]:
            line_chars = [char for span in line["spans"] for char in span["chars"]]
            if not line_chars:
                continue

            if previous_line_bbox is not None:
                # Two pieces on the same visual row (e.g. "Name:      Sara") are
                # joined with a space; real new rows get a line break.
                same_row = _same_visual_row(previous_line_bbox, line["bbox"])
                raw_chars.append(" " if same_row else "\n")
                raw_boxes.append(None)
                if not same_row:
                    line_id += 1

            for char in line_chars:
                x0, y0, x1, y1 = char["bbox"]
                raw_chars.append(char["c"])
                raw_boxes.append(CharBox(page_number, line_id, x0, y0, x1, y1))
            previous_line_bbox = line["bbox"]

        if previous_line_bbox is not None:
            # Blank line after every block (paragraph).
            raw_chars.extend("\n\n")
            raw_boxes.extend([None, None])
            line_id += 1


def _same_visual_row(bbox_a: tuple, bbox_b: tuple) -> bool:
    """True when two line boxes share (most of) the same vertical position."""
    top = max(bbox_a[1], bbox_b[1])
    bottom = min(bbox_a[3], bbox_b[3])
    overlap = bottom - top
    smaller_height = min(bbox_a[3] - bbox_a[1], bbox_b[3] - bbox_b[1])
    return smaller_height > 0 and overlap / smaller_height > 0.6 and bbox_b[0] >= bbox_a[0]


def _text_statistics(text: str) -> dict:
    """Simple document statistics shown in the app and the scan summary."""
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": text.count("\n") + 1,
    }
