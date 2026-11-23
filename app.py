"""PrivacyLens - Streamlit application.

Run it from the project folder with:

    streamlit run app.py

The app follows the workflow of the project:
Input -> Detections -> Risk summary -> Review & redact -> Export.
All real work is done by the modules in ``src/``; this file only builds the
user interface and handles errors in a friendly way.
"""

from __future__ import annotations

import copy
import html
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import streamlit as st

from src.config import ConfigError, load_config, resolve_path, setup_logging
from src.file_processor import InputError, load_input
from src.ner_detector import ModelNotAvailableError, load_ner_model
from src.pipeline import ScanResult, scan
from src.redactor import RedactionError, detection_rectangles, redact, redact_pdf, verify_pdf_redaction
from src.report_generator import LIMITATIONS, build_summary, detections_to_csv, summary_to_json
from src.schemas import ALL_CATEGORIES, LoadResult
from src.visualization import (
    CATEGORY_COLORS,
    LEVEL_COLORS,
    plot_category_counts,
    plot_risk_breakdown,
    plot_risk_gauge,
)

logger = logging.getLogger("privacylens.app")

CATEGORY_HELP = {
    "PERSON": "Name of a person (found by the AI model)",
    "ORG": "Company or organisation (AI model)",
    "LOCATION": "City, state or country (AI model)",
    "ADDRESS": "Street address (AI model)",
    "EMAIL": "Email address (rule)",
    "PHONE": "Phone number, incl. Saudi formats (rule)",
    "ID_LIKE": "Number that looks like an ID (rule)",
    "CREDIT_CARD": "Payment card number (rule + checksum)",
    "IBAN": "Bank account number (rule)",
}
MAX_PREVIEW_CHARACTERS = 30_000


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading the AI model (first time only)...")
def get_model():
    """Load the NER model once per server process."""
    return load_ner_model()


def show_figure(figure) -> None:
    """Display a matplotlib figure and free its memory."""
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def highlight_html(text: str, detections: list, hidden_ids: set[str] | None = None) -> str:
    """Return the text as HTML with every detection highlighted.

    If ``hidden_ids`` is given, those detections are shown as black
    ``[REDACTED]`` bars instead (redaction preview).
    """
    hidden_ids = hidden_ids or set()
    parts: list[str] = []
    cursor = 0
    for detection in sorted(detections, key=lambda d: d.start):
        if detection.start >= len(text):
            break
        parts.append(html.escape(text[cursor : detection.start]))
        if detection.id in hidden_ids:
            parts.append(
                '<span style="background:#111;color:#fff;border-radius:3px;padding:0 6px;'
                f'font-size:0.8em">REDACTED-{detection.category}</span>'
            )
        else:
            color = CATEGORY_COLORS.get(detection.category, "#dddddd")
            parts.append(
                f'<mark style="background:{color};color:#111;border-radius:3px;padding:0 3px" '
                f'title="{detection.category}">{html.escape(detection.text)}'
                f'<sup style="font-size:0.62em;margin-left:3px;opacity:0.75">{detection.category}</sup></mark>'
            )
        cursor = detection.end
    parts.append(html.escape(text[cursor:]))
    return (
        '<div style="white-space:pre-wrap;line-height:1.9;max-height:460px;overflow-y:auto;'
        'border:1px solid rgba(128,128,128,0.35);border-radius:8px;padding:14px 16px;font-size:0.92rem">'
        + "".join(parts)
        + "</div>"
    )


def legend_html(categories: list[str]) -> str:
    """Small coloured chips that explain the highlight colours."""
    chips = [
        f'<span title="{CATEGORY_HELP[c]}" style="background:{CATEGORY_COLORS[c]};color:#111;border-radius:10px;'
        f'padding:2px 10px;margin-right:6px;font-size:0.78rem">{c}</span>'
        for c in categories
    ]
    return '<div style="margin:4px 0 10px 0">' + "".join(chips) + "</div>"


def render_pdf_page(pdf_bytes: bytes, page_number: int, outline: list | None = None) -> bytes:
    """Render one PDF page as a PNG image, optionally outlining rectangles."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document[page_number]
        for rect, color in outline or []:
            page.draw_rect(rect, color=color, width=1.2)
        return page.get_pixmap(dpi=110).tobytes("png")


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))


# -------------------------------------------------------------------- pages
def sidebar(config: dict) -> tuple[str, dict]:
    """Draw the sidebar and return ``(page_name, settings)``."""
    with st.sidebar:
        st.title("🔍 PrivacyLens")
        st.caption("Find and redact sensitive information before you share a document.")
        page = st.radio("Page", ["Scan a document", "Model & data insights"], label_visibility="collapsed")

        st.divider()
        st.subheader("Settings")
        min_confidence = st.slider(
            "AI confidence threshold",
            min_value=0.30,
            max_value=0.95,
            value=float(config["ner"]["min_confidence"]),
            step=0.05,
            help="Detections of the AI model below this confidence are ignored. "
            "Lower = catches more (but more false alarms). Applies to the next scan.",
        )
        show_labels = st.toggle(
            "Print category on PDF redaction bars",
            value=bool(config["redaction"]["pdf_show_labels"]),
            help="Writes e.g. PERSON in white on each black bar of the redacted PDF.",
        )

        st.divider()
        st.caption(
            "**Privacy:** your document is processed on this computer, in memory only. "
            "Nothing is uploaded to a cloud service or saved to disk."
        )
        st.caption(f"Version {config['app']['version']} - ARTI 303 final project")
    return page, {"min_confidence": min_confidence, "show_labels": show_labels}


def run_scan(config: dict, settings: dict, text: str | None = None, file: bytes | None = None, file_name: str | None = None) -> None:
    """Load the input, run the scan and store the result in the session."""
    scan_config = copy.deepcopy(config)
    scan_config["ner"]["min_confidence"] = settings["min_confidence"]
    # Forget the previous document, so old results never sit under a new error.
    for key in ("load_result", "scan_result", "selected_ids", "pdf_cache"):
        st.session_state.pop(key, None)
    try:
        with st.spinner("Reading and checking the input..."):
            load_result = load_input(text=text, file=file, file_name=file_name, config=scan_config)
        get_model()  # shows its own spinner the first time
        with st.spinner("Looking for sensitive information..."):
            scan_result = scan(load_result, scan_config)
    except InputError as error:
        st.error(f"**Input problem:** {error}")
        return
    except ModelNotAvailableError as error:
        st.error(f"**AI model problem:** {error}")
        return
    except Exception:  # anything unexpected: tell the user, log the details
        logger.exception("Unexpected failure during scan")
        st.error("Something went wrong while processing this input. Please try another file.")
        return

    st.session_state["load_result"] = load_result
    st.session_state["scan_result"] = scan_result
    # Safe default: everything that was found is selected for redaction.
    st.session_state["selected_ids"] = {detection.id for detection in scan_result.detections}
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def input_section(config: dict, settings: dict) -> None:
    """Step 1: the three ways to provide a document."""
    st.subheader("1. Choose what to scan")
    upload_tab, paste_tab, sample_tab = st.tabs(["📄 Upload a file", "✏️ Paste text", "🧪 Try a sample"])

    with upload_tab:
        limits = config["input"]
        uploaded = st.file_uploader(
            f"PDF (text-based) or TXT, up to {limits['max_file_mb']} MB and {limits['max_pdf_pages']} pages",
            help="Scanned PDFs (photos of pages) are not supported in this version.",
        )
        if st.button("Scan file", type="primary", disabled=uploaded is None):
            run_scan(config, settings, file=uploaded.getvalue(), file_name=uploaded.name)

    with paste_tab:
        pasted = st.text_area("Paste your text here", height=180, placeholder="Paste an email, a message, a letter...")
        if st.button("Scan text", type="primary"):
            run_scan(config, settings, text=pasted)

    with sample_tab:
        samples_dir = resolve_path("data/samples")
        samples = sorted(path.name for path in samples_dir.glob("*") if path.suffix.lower() in (".pdf", ".txt"))
        if not samples:
            st.info("No sample files found. Create them with: `python scripts/03_make_synthetic_docs.py`")
        else:
            choice = st.selectbox("Fictional sample documents (generated, no real data)", samples)
            st.caption("`sample_scanned.pdf` is an image-only PDF: use it to see how unsupported files are handled.")
            if st.button("Scan sample", type="primary"):
                run_scan(config, settings, file=(samples_dir / choice).read_bytes(), file_name=choice)


def detections_tab(load_result: LoadResult, result: ScanResult) -> None:
    text = load_result.canonical_text
    if not result.detections:
        st.success("No sensitive information was detected in this document.")
        st.markdown(highlight_html(text[:MAX_PREVIEW_CHARACTERS], []), unsafe_allow_html=True)
        return

    present = [category for category in ALL_CATEGORIES if any(d.category == category for d in result.detections)]
    st.markdown(legend_html(present), unsafe_allow_html=True)
    st.markdown(highlight_html(text[:MAX_PREVIEW_CHARACTERS], result.detections), unsafe_allow_html=True)
    if len(text) > MAX_PREVIEW_CHARACTERS:
        st.caption(f"Preview shows the first {MAX_PREVIEW_CHARACTERS:,} of {len(text):,} characters. The table below lists everything.")

    st.markdown("##### All detected items")
    table = pd.DataFrame(
        {
            "Category": d.category,
            "Text": d.text,
            "Found by": {"regex": "Rule", "ner": "AI model", "propagated": "Repeat of a detected name"}[d.source],
            "Confidence": d.confidence,
            "Rule / label": d.raw_label,
            "Start": d.start,
            "End": d.end,
        }
        for d in result.detections
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
    )


def risk_tab(result: ScanResult, config: dict) -> None:
    risk = result.risk
    left, right = st.columns([1, 2])
    with left:
        st.metric("Privacy risk score", f"{risk.score} / 100")
        st.markdown(
            f'<span style="background:{LEVEL_COLORS[risk.level]};color:#111;padding:4px 14px;'
            f'border-radius:12px;font-weight:600">{risk.level} risk</span>',
            unsafe_allow_html=True,
        )
    with right:
        show_figure(plot_risk_gauge(risk, config["risk"]["bands"]))

    if not risk.breakdown:
        st.info("Nothing was detected, so the score is 0.")
        return

    chart_left, chart_right = st.columns(2)
    with chart_left:
        show_figure(plot_category_counts(result.detections))
    with chart_right:
        show_figure(plot_risk_breakdown(risk))

    st.markdown("##### How the score was calculated")
    breakdown = pd.DataFrame(
        {
            "Category": category,
            "Detections": details["detections"],
            "Unique items": details["unique_items"],
            "Base points": details["base_points"],
            "Points": details["points"],
            "Capped": "yes" if details["capped"] else "",
        }
        for category, details in risk.breakdown.items()
    )
    st.dataframe(breakdown, hide_index=True, width="stretch")
    st.caption(
        "Rule: the first unique item of a category gives its base points; each additional unique item adds 25% "
        "of the base; a category gives at most 2x its base; the total is capped at 100. "
        "This score is defined by this project. It is not an official privacy standard."
    )


def current_redacted_pdf(load_result: LoadResult, result: ScanResult, selected: set[str], show_labels: bool) -> bytes:
    """Build the redacted PDF only when the selection changed (it is cached)."""
    key = (tuple(sorted(selected)), show_labels, id(result))
    cache = st.session_state.get("pdf_cache")
    if cache is None or cache["key"] != key:
        pdf_bytes = redact_pdf(load_result, result.detections, selected, show_labels=show_labels)
        check = verify_pdf_redaction(pdf_bytes, load_result, result.detections, selected)
        cache = {"key": key, "bytes": pdf_bytes, "check": check}
        st.session_state["pdf_cache"] = cache
    return cache["bytes"]


def review_tab(load_result: LoadResult, result: ScanResult, settings: dict) -> None:
    if not result.detections:
        st.info("There is nothing to redact.")
        return

    selected: set[str] = st.session_state["selected_ids"]
    present = [category for category in ALL_CATEGORIES if any(d.category == category for d in result.detections)]

    st.markdown("Choose what should be removed. Everything is selected by default.")
    all_column, none_column, category_column, apply_column = st.columns([1, 1, 4, 1.3], vertical_alignment="bottom")
    new_selection = None
    if all_column.button("Select all", width="stretch"):
        new_selection = {d.id for d in result.detections}
    if none_column.button("Clear all", width="stretch"):
        new_selection = set()
    chosen_categories = category_column.multiselect("Select by category", present, placeholder="e.g. PERSON, PHONE")
    if apply_column.button("Apply categories", width="stretch", disabled=not chosen_categories):
        new_selection = {d.id for d in result.detections if d.category in chosen_categories}
    if new_selection is not None:
        st.session_state["selected_ids"] = selected = new_selection
        st.session_state["editor_version"] += 1  # rebuild the table with the new ticks

    table = pd.DataFrame(
        {
            "Redact": d.id in selected,
            "Category": d.category,
            "Text": d.text,
            "Confidence": d.confidence,
            "ID": d.id,
        }
        for d in result.detections
    )
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["Category", "Text", "Confidence", "ID"],
        column_config={"Redact": st.column_config.CheckboxColumn(help="Tick = remove this item from the output")},
        key=f"editor_{st.session_state['editor_version']}",
    )
    selected = set(edited.loc[edited["Redact"], "ID"])
    st.session_state["selected_ids"] = selected
    st.caption(f"{len(selected)} of {len(result.detections)} item(s) selected for redaction.")

    st.markdown("##### Preview")
    if load_result.source_type == "pdf":
        try:
            redacted_pdf = current_redacted_pdf(load_result, result, selected, settings["show_labels"])
        except RedactionError as error:
            st.error(f"**Redaction problem:** {error}")
            return
        page_count = load_result.stats["pages"]
        page_number = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1) - 1 if page_count > 1 else 0
        outline = [
            (rect, hex_to_rgb(CATEGORY_COLORS[d.category]))
            for d in result.detections
            for rect_page, rect in detection_rectangles(load_result.char_boxes, d.start, d.end)
            if rect_page == page_number
        ]
        before, after = st.columns(2)
        before.caption("Original (detections outlined)")
        before.image(render_pdf_page(load_result.pdf_bytes, page_number, outline), width="stretch")
        after.caption("Redacted")
        after.image(render_pdf_page(redacted_pdf, page_number), width="stretch")
    else:
        text = load_result.canonical_text[:MAX_PREVIEW_CHARACTERS]
        st.markdown(highlight_html(text, result.detections, hidden_ids=selected), unsafe_allow_html=True)


def export_tab(load_result: LoadResult, result: ScanResult, settings: dict, config: dict) -> None:
    selected: set[str] = st.session_state["selected_ids"]
    base_name = Path(load_result.file_name).stem if load_result.file_name else "pasted_text"

    st.markdown(f"**{len(selected)}** of **{len(result.detections)}** detected item(s) will be removed from the exported files.")
    text_column, pdf_column, summary_column = st.columns(3)

    with text_column:
        st.markdown("##### Sanitized text")
        sanitized = redact(load_result.canonical_text, result.detections, selected, config["redaction"]["placeholder"])
        st.download_button("⬇️ Download .txt", sanitized.encode("utf-8"), f"{base_name}_sanitized.txt", "text/plain", width="stretch")
        st.caption("Plain text with placeholders such as [REDACTED-PHONE].")

    with pdf_column:
        st.markdown("##### Redacted PDF")
        if load_result.source_type != "pdf":
            st.caption("Available when the input is a PDF file.")
        else:
            try:
                redacted_pdf = current_redacted_pdf(load_result, result, selected, settings["show_labels"])
                check = st.session_state["pdf_cache"]["check"]
                st.download_button("⬇️ Download .pdf", redacted_pdf, f"{base_name}_redacted.pdf", "application/pdf", width="stretch")
                if check["passed"]:
                    st.success(f"Verified: the text of all {check['checked']} redacted item(s) was removed from the file.")
                else:
                    st.error(f"Verification failed for {len(check['failed_ids'])} item(s). Do not share this file.")
                st.caption("The text is deleted from the PDF, not just covered. Document metadata is cleared too.")
            except RedactionError as error:
                st.error(f"**Redaction problem:** {error}")

    with summary_column:
        st.markdown("##### Scan summary")
        include_text = st.checkbox("Include the detected text (not recommended)", value=False)
        summary = build_summary(
            load_result, result.detections, result.risk, selected, result.notes, result.model_name, include_sensitive_text=include_text
        )
        st.download_button("⬇️ Download summary (.json)", summary_to_json(summary), f"{base_name}_summary.json", "application/json", width="stretch")
        st.download_button("⬇️ Download detections (.csv)", detections_to_csv(summary), f"{base_name}_detections.csv", "text/csv", width="stretch")
        st.caption("Counts, categories, positions and score. By default the sensitive strings are left out.")


def scan_page(config: dict, settings: dict) -> None:
    st.title("PrivacyLens")
    st.markdown(
        "Scan a document for **names, organisations, places, addresses, emails, phone numbers, IDs and bank details**, "
        "see a privacy risk score, choose what to hide and download a redacted copy."
    )
    st.warning(
        "PrivacyLens uses rules and an AI model. Both can miss items or raise false alarms. "
        "It does **not** guarantee that a document is safe to share: always review the result yourself.",
        icon="⚠️",
    )
    input_section(config, settings)

    if "scan_result" not in st.session_state:
        return
    load_result: LoadResult = st.session_state["load_result"]
    result: ScanResult = st.session_state["scan_result"]

    st.divider()
    st.subheader("2. Results" + (f" for `{load_result.file_name}`" if load_result.file_name else ""))
    first, second, third, fourth = st.columns(4)
    first.metric("Items detected", len(result.detections))
    second.metric("Risk score", f"{result.risk.score} / 100")
    third.metric("Risk level", result.risk.level)
    fourth.metric("Pages" if load_result.source_type == "pdf" else "Words", load_result.stats.get("pages", load_result.stats["words"]))

    for warning in load_result.warnings:
        st.warning(warning)
    with st.expander("Processing details"):
        st.write({"input statistics": load_result.stats, "AI model": result.model_name, "seconds": result.seconds})
        for note in result.notes:
            st.write(f"- {note}")

    tabs = st.tabs(["🔎 Detections", "📊 Risk summary", "✂️ Review & redact", "⬇️ Export"])
    with tabs[0]:
        detections_tab(load_result, result)
    with tabs[1]:
        risk_tab(result, config)
    with tabs[2]:
        review_tab(load_result, result, settings)
    with tabs[3]:
        export_tab(load_result, result, settings, config)

    with st.expander("Limitations"):
        for limitation in LIMITATIONS:
            st.write(f"- {limitation}")


def read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def scores_table(per_category: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Category": category,
            "Precision": scores["precision"],
            "Recall": scores["recall"],
            "F1": scores["f1"],
            "Gold entities": scores["support"],
        }
        for category, scores in per_category.items()
        if scores["support"] > 0
    )


def insights_page(config: dict) -> None:
    """Show how the model was built: dataset, training and evaluation."""
    st.title("Model & data insights")
    st.caption("Where the AI model comes from and how well it works. All numbers are produced by the scripts in `scripts/`.")
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    figures_dir = resolve_path(config["data"]["figures_dir"])

    def figure(name: str, caption: str) -> None:
        path = figures_dir / name
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")

    data_tab, training_tab, evaluation_tab = st.tabs(["📚 Dataset", "🏋️ Training", "✅ Evaluation"])

    with data_tab:
        summary = read_json(metrics_dir / "data_summary.json")
        if summary is None:
            st.info("No dataset summary yet. Run `python scripts/01_prepare_data.py`.")
        else:
            dataset, stats = summary["dataset"], summary["exploration"]
            st.markdown(f"**Dataset:** [{dataset['name']}]({dataset['url']}) - license **{dataset['license']}** - fully synthetic (no real personal data)")
            first, second, third, fourth = st.columns(4)
            first.metric("Documents after cleaning", f"{stats['documents']:,}")
            second.metric("Label types in dataset", stats["raw_label_types"])
            third.metric("Domains", stats["domains"])
            fourth.metric("Median length (chars)", f"{stats['text_length']['50%']:.0f}")
            st.markdown("##### Documents used by this project")
            st.dataframe(
                pd.DataFrame({"Split": name, "Documents": split["documents"], **split["entities"]} for name, split in summary["splits"].items()),
                hide_index=True,
                width="stretch",
            )
            left, right = st.columns(2)
            with left:
                figure("01_dataset_label_distribution.png", "Labels in the dataset; blue ones are learned by our model")
                figure("03_text_length_distribution.png", "Document length")
            with right:
                figure("02_category_distribution.png", "PrivacyLens categories after label mapping")
                figure("04_documents_per_domain.png", "Documents per domain")
                figure("05_spans_per_document.png", "Annotations per document")
            with st.expander("Validation and cleaning report"):
                st.json({"validation": summary["validation"], "cleaning": summary["cleaning"]})
            with st.expander("Dataset limitations"):
                for limitation in summary["limitations"]:
                    st.write(f"- {limitation}")

    with training_tab:
        info = read_json(metrics_dir / "training_history.json")
        if info is None:
            st.info("No training record yet. Run `python scripts/02_train_model.py`.")
        else:
            st.markdown(
                f"The pretrained model **{info['base_model']}** was fine-tuned on **{info['train_documents']:,}** documents "
                f"({info['train_windows']:,} token windows) for **{len(info['history'])} epochs** on *{info['device']}* "
                f"in **{info['training_minutes']} minutes**."
            )
            figure("06_training_history.png", "Training history")
            st.dataframe(pd.DataFrame(info["history"]), hide_index=True, width="stretch")
            with st.expander("Hyperparameters and library versions"):
                st.json({"hyperparameters": info["hyperparameters"], "versions": info["versions"], "labels": info["labels"]})

    with evaluation_tab:
        results = read_json(metrics_dir / "evaluation_results.json")
        if results is None:
            st.info("No evaluation results yet. Run `python scripts/04_evaluate.py`.")
        else:
            public, synthetic = results["public_test_set"], results["synthetic_documents"]
            st.markdown(f"##### A. Public test set ({public['documents']:,} unseen documents)")
            first, second, third = st.columns(3)
            first.metric("Fine-tuned model F1", f"{public['fine_tuned_model']['overall']['f1']:.3f}")
            second.metric(
                "Baseline model F1",
                f"{public['baseline_model']['overall']['f1']:.3f}",
                help=f"{public['baseline_model']['name']} on PERSON / ORG / LOCATION",
            )
            third.metric("Sensitive characters covered", f"{public['fine_tuned_model']['character_recall']['overall']:.1%}")
            st.dataframe(scores_table(public["fine_tuned_model"]["per_category"]), hide_index=True, width="stretch")
            left, right = st.columns(2)
            with left:
                figure("07_test_scores_by_category.png", "Fine-tuned model per category")
                figure("11_regex_scores.png", "Regex layer on the dataset's own email / phone / card labels")
            with right:
                figure("08_model_comparison.png", "Does fine-tuning help? Comparison with an off-the-shelf model")

            st.markdown(f"##### B. Held-out synthetic documents ({synthetic['test_documents']} fictional letters, CVs, forms...)")
            pipeline = synthetic["full_pipeline_on_test"]
            first, second, third = st.columns(3)
            first.metric("Complete pipeline F1", f"{pipeline['overall']['f1']:.3f}")
            second.metric("Precision / Recall", f"{pipeline['overall']['precision']:.2f} / {pipeline['overall']['recall']:.2f}")
            third.metric("Sensitive characters covered", f"{pipeline['character_recall']['overall']:.1%}")
            st.dataframe(scores_table(pipeline["per_category"]), hide_index=True, width="stretch")
            left, right = st.columns(2)
            with left:
                figure("10_synthetic_test_scores.png", "Complete pipeline per category")
            with right:
                figure("09_threshold_study.png", "Confidence threshold study on the development set")

            errors_path = metrics_dir / "error_analysis_synthetic_test.csv"
            if errors_path.exists():
                st.markdown("##### Error analysis (synthetic test documents)")
                errors = pd.read_csv(errors_path)
                st.dataframe(errors, hide_index=True, width="stretch", height=280)
            st.caption(
                "Entity-level scores are strict: a prediction counts only if category and both boundaries match exactly. "
                "'Sensitive characters covered' is the share of gold sensitive characters hidden by any detection."
            )


def main() -> None:
    st.set_page_config(page_title="PrivacyLens", page_icon="🔍", layout="wide")
    try:
        config = load_config()
    except ConfigError as error:
        st.error(f"**Configuration problem:** {error}")
        st.stop()
    if "logging_ready" not in st.session_state:
        setup_logging(config)
        st.session_state["logging_ready"] = True

    page, settings = sidebar(config)
    if page == "Scan a document":
        scan_page(config, settings)
    else:
        insights_page(config)


main()
