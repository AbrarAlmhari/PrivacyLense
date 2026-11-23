"""Redaction: remove selected sensitive information from text and PDFs.

Two outputs are supported:

* ``redact``      -> sanitized plain text with placeholders such as
  ``[REDACTED-PHONE]``.
* ``redact_pdf``  -> a new PDF where the selected text is **really removed**
  from the file (not just covered by a black box) and a black bar is drawn
  in its place.

The original input is never modified.
"""

from __future__ import annotations

import logging

import pymupdf

from src.schemas import CharBox, Detection, LoadResult

logger = logging.getLogger(__name__)


class RedactionError(Exception):
    """Raised when a redacted document cannot be produced."""


# --------------------------------------------------------------------------
# Plain text
# --------------------------------------------------------------------------
def redact(
    text: str,
    detections: list[Detection],
    selected_ids: set[str] | list[str],
    placeholder: str = "[REDACTED-{category}]",
) -> str:
    """Replace the selected detections in ``text`` with placeholders.

    The replacements are applied from the **end of the text backward**, so the
    offsets of the detections that are still waiting stay valid.
    """
    selected = [d for d in detections if d.id in set(selected_ids)]
    sanitized = text
    for detection in sorted(selected, key=lambda d: d.start, reverse=True):
        label = placeholder.format(category=detection.category)
        sanitized = sanitized[: detection.start] + label + sanitized[detection.end :]
    logger.info("Text redaction applied to %d detection(s)", len(selected))
    return sanitized


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def detection_rectangles(
    char_boxes: list[CharBox | None], start: int, end: int
) -> list[tuple[int, pymupdf.Rect]]:
    """Return the page rectangles that cover characters ``start`` .. ``end``.

    A detection that continues on a second line produces one rectangle per
    line. Each result is ``(page_number, rectangle)``.
    """
    rectangles: list[tuple[int, pymupdf.Rect]] = []
    current_key = None

    for box in char_boxes[start:end]:
        if box is None:  # a line break or space inserted by the loader
            continue
        key = (box.page, box.line)
        char_rect = pymupdf.Rect(box.x0, box.y0, box.x1, box.y1)
        if key == current_key:
            rectangles[-1][1].include_rect(char_rect)
        else:
            rectangles.append((box.page, char_rect))
            current_key = key
    return rectangles


def redact_pdf(
    load_result: LoadResult,
    detections: list[Detection],
    selected_ids: set[str] | list[str],
    show_labels: bool = True,
    fill_color: tuple[float, float, float] = (0, 0, 0),
) -> bytes:
    """Create a redacted copy of the uploaded PDF and return it as bytes.

    How it works:

    1. Every selected detection is converted to rectangles on the page using
       the character positions saved by the loader.
    2. A PDF "redaction annotation" is added for each rectangle.
    3. ``apply_redactions`` deletes the text under the rectangles from the
       page content and paints the rectangles black.
    4. Document metadata (author, title, ...) is cleared, because it often
       contains names too.
    """
    if load_result.source_type != "pdf" or not load_result.pdf_bytes or load_result.char_boxes is None:
        raise RedactionError("A redacted PDF can only be created when the input was a PDF.")

    selected = [d for d in detections if d.id in set(selected_ids)]
    try:
        document = pymupdf.open(stream=load_result.pdf_bytes, filetype="pdf")
    except Exception as error:
        raise RedactionError("The original PDF could not be reopened.") from error

    with document:
        touched_pages: set[int] = set()
        for detection in selected:
            for page_number, rect in detection_rectangles(load_result.char_boxes, detection.start, detection.end):
                page = document[page_number]
                # Shrink the box a little so that neighbouring characters and
                # the lines above/below are not removed by accident.
                shrink_y = rect.height * 0.12
                target = pymupdf.Rect(rect.x0 + 0.3, rect.y0 + shrink_y, rect.x1 - 0.3, rect.y1 - shrink_y)
                if target.is_empty:
                    target = rect
                label = detection.category if show_labels else None
                page.add_redact_annot(
                    target,
                    text=label,
                    fontsize=max(4, min(7, rect.height * 0.55)),
                    text_color=(1, 1, 1),
                    fill=fill_color,
                    align=pymupdf.TEXT_ALIGN_CENTER,
                )
                # Links (for example mailto: links) could leak the hidden text.
                for link in page.get_links():
                    if pymupdf.Rect(link["from"]).intersects(rect):
                        page.delete_link(link)
                touched_pages.add(page_number)

        for page_number in touched_pages:
            document[page_number].apply_redactions()

        document.set_metadata({})
        document.del_xml_metadata()
        # garbage=4 removes unused objects, so deleted text is not left behind
        # inside the file.
        output = document.tobytes(garbage=4, deflate=True)

    logger.info("PDF redaction applied to %d detection(s) on %d page(s)", len(selected), len(touched_pages))
    return output


def verify_pdf_redaction(
    redacted_pdf: bytes,
    load_result: LoadResult,
    detections: list[Detection],
    selected_ids: set[str] | list[str],
) -> dict:
    """Check that the redacted PDF no longer contains the selected text.

    For each redacted rectangle we extract the text that is still inside it.
    If anything readable is left, the detection is reported as a failure.

    Returns:
        ``{"passed": bool, "checked": int, "failed_ids": [...]}``
    """
    selected = [d for d in detections if d.id in set(selected_ids)]
    failed_ids: list[str] = []

    with pymupdf.open(stream=redacted_pdf, filetype="pdf") as document:
        for detection in selected:
            for page_number, rect in detection_rectangles(load_result.char_boxes, detection.start, detection.end):
                shrink_y = rect.height * 0.25
                inner = pymupdf.Rect(rect.x0 + 1, rect.y0 + shrink_y, rect.x1 - 1, rect.y1 - shrink_y)
                remaining = document[page_number].get_text("text", clip=inner).strip()
                # The white category label printed on the black bar (or a
                # clipped part of it) is fine; anything else is a failure.
                if remaining and remaining not in detection.category:
                    failed_ids.append(detection.id)
                    break

    return {"passed": not failed_ids, "checked": len(selected), "failed_ids": failed_ids}
