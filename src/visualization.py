"""Charts for the Streamlit app and for the evaluation report.

Every function returns a matplotlib ``Figure``. The caller decides what to do
with it (show it in Streamlit with ``st.pyplot`` or save it with ``savefig``).
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.schemas import ALL_CATEGORIES, Detection, RiskResult

# One fixed colour per category, used in the charts AND in the text highlights.
CATEGORY_COLORS = {
    "PERSON": "#f4a261",
    "ORG": "#8ecae6",
    "LOCATION": "#90be6d",
    "ADDRESS": "#52b69a",
    "EMAIL": "#cdb4db",
    "PHONE": "#ffd166",
    "ID_LIKE": "#ef767a",
    "CREDIT_CARD": "#f28482",
    "IBAN": "#e5989b",
}
LEVEL_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#f4a261", "Critical": "#e63946"}


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)


def plot_category_counts(detections: list[Detection]) -> Figure:
    """Horizontal bar chart: number of detections per category."""
    counts = Counter(detection.category for detection in detections)
    categories = [category for category in ALL_CATEGORIES if counts[category]]
    figure, axis = plt.subplots(figsize=(6, max(2.2, 0.5 * len(categories) + 1)))
    bars = axis.barh(categories[::-1], [counts[c] for c in categories[::-1]], color=[CATEGORY_COLORS[c] for c in categories[::-1]])
    axis.bar_label(bars, padding=3)
    axis.set_xlabel("Number of detections")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))  # whole numbers only
    axis.set_title("Detections by category")
    _style(axis)
    figure.tight_layout()
    return figure


def plot_risk_breakdown(risk: RiskResult) -> Figure:
    """Horizontal bar chart: risk points contributed by each category."""
    categories = list(risk.breakdown)
    points = [risk.breakdown[category]["points"] for category in categories]
    figure, axis = plt.subplots(figsize=(6, max(2.2, 0.5 * len(categories) + 1)))
    bars = axis.barh(categories[::-1], points[::-1], color=[CATEGORY_COLORS.get(c, "#999999") for c in categories[::-1]])
    axis.bar_label(bars, fmt="%g", padding=3)
    axis.set_xlabel("Risk points")
    axis.set_title(f"Risk score breakdown (total {risk.score}/100 - {risk.level})")
    _style(axis)
    figure.tight_layout()
    return figure


def plot_risk_gauge(risk: RiskResult, bands: list[dict]) -> Figure:
    """A simple 0-100 scale with the risk bands and a marker for the score."""
    figure, axis = plt.subplots(figsize=(7, 1.3))
    left = 0
    for band in bands:
        axis.barh(0, band["max"] - left, left=left, color=LEVEL_COLORS[band["level"]], alpha=0.85, height=0.5)
        axis.text((left + band["max"]) / 2, -0.55, band["level"], ha="center", va="top", fontsize=9)
        left = band["max"]
    axis.plot([risk.score, risk.score], [-0.32, 0.32], color="black", linewidth=3)
    axis.text(risk.score, 0.42, str(risk.score), ha="center", va="bottom", fontsize=12, fontweight="bold")
    axis.set_xlim(0, 100)
    axis.set_ylim(-1.1, 0.9)
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_training_history(history: list[dict]) -> Figure:
    """Loss and validation F1 for every training epoch."""
    epochs = [record["epoch"] for record in history]
    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(10, 3.6))

    loss_axis.plot(epochs, [record["train_loss"] for record in history], marker="o", label="Training loss")
    loss_axis.plot(epochs, [record["validation_loss"] for record in history], marker="o", label="Validation loss")
    loss_axis.set_title("Loss per epoch")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_xticks(epochs)
    loss_axis.legend()

    for key, label in (("validation_precision", "Precision"), ("validation_recall", "Recall"), ("validation_f1", "F1")):
        f1_axis.plot(epochs, [record[key] for record in history], marker="o", label=label)
    f1_axis.set_title("Validation scores per epoch (entity-level, strict)")
    f1_axis.set_xlabel("Epoch")
    f1_axis.set_xticks(epochs)
    f1_axis.legend()

    for axis in (loss_axis, f1_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_scores_by_category(per_category: dict, title: str) -> Figure:
    """Grouped bars: precision, recall and F1 for every category."""
    categories = [category for category, scores in per_category.items() if scores["support"] > 0]
    figure, axis = plt.subplots(figsize=(max(6, 1.3 * len(categories) + 2), 3.8))
    width = 0.26
    for offset, (metric, color) in enumerate((("precision", "#457b9d"), ("recall", "#f4a261"), ("f1", "#2a9d8f"))):
        positions = [index + (offset - 1) * width for index in range(len(categories))]
        bars = axis.bar(positions, [per_category[c][metric] for c in categories], width=width, label=metric.capitalize(), color=color)
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axis.set_xticks(range(len(categories)))
    axis.set_xticklabels([f"{c}\n(n={per_category[c]['support']})" for c in categories], fontsize=8)
    axis.set_ylim(0, 1.12)
    axis.set_title(title)
    axis.legend(loc="lower right", ncols=3, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_model_comparison(results: dict[str, dict], categories: tuple[str, ...]) -> Figure:
    """F1 per category for several models side by side."""
    figure, axis = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / max(len(results), 1)
    colors = ["#2a9d8f", "#b6bfcc", "#457b9d"]
    for model_index, (model_name, scores) in enumerate(results.items()):
        positions = [index + (model_index - (len(results) - 1) / 2) * width for index in range(len(categories))]
        values = [scores["per_category"].get(category, {}).get("f1", 0) for category in categories]
        # A model that does not know a category at all gets "n/a", not 0.00.
        labels = [f"{value:.2f}" if category in scores["per_category"] else "n/a" for category, value in zip(categories, values)]
        bars = axis.bar(positions, values, width=width, label=model_name, color=colors[model_index % len(colors)])
        axis.bar_label(bars, labels=labels, fontsize=7, padding=2)
    axis.set_xticks(range(len(categories)))
    axis.set_xticklabels(categories)
    axis.set_ylim(0, 1.3)
    axis.set_ylabel("F1 (entity-level, strict)")
    axis.set_title("Fine-tuned model vs. off-the-shelf model")
    axis.legend(fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_threshold_curve(rows: list[dict], chosen: float) -> Figure:
    """Precision / recall / F1 on the development set for each confidence threshold."""
    thresholds = [row["threshold"] for row in rows]
    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    for metric in ("precision", "recall", "f1"):
        axis.plot(thresholds, [row[metric] for row in rows], marker="o", label=metric.capitalize())
    axis.axvline(chosen, color="grey", linestyle="--", linewidth=1, label=f"Chosen: {chosen}")
    axis.set_xlabel("Minimum confidence")
    axis.set_title("Effect of the confidence threshold (development set)")
    axis.legend(fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure
