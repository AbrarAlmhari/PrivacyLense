"""Step 4: evaluate the detectors and save metrics, figures and error tables.

Usage (from the project folder):
    python scripts/04_evaluate.py

What is measured:

A. Public test set (3,000 Nemotron-PII documents the model never saw)
   1. our fine-tuned model              -> PERSON, ORG, LOCATION, ADDRESS
   2. an off-the-shelf NER model        -> baseline for comparison
   3. the regex layer                   -> EMAIL, PHONE, CREDIT_CARD

B. Synthetic documents (fictional letters, CVs, forms... from data/synthetic)
   4. development part: effect of the confidence threshold
   5. held-out test part: the COMPLETE pipeline (regex + AI + merge)

Results go to reports/metrics/ and reports/figures/.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path, setup_logging
from src.data_preparation import load_processed
from src.evaluation import (
    character_recall,
    collect_errors,
    evaluate_entities,
    spans_from_detections,
    spans_from_entities,
)
from src.file_processor import load_input
from src.ner_detector import detect_with_model, load_model_from, load_ner_model
from src.pipeline import scan
from src.regex_detector import detect_regex
from src.schemas import ALL_CATEGORIES, NER_CATEGORIES, REGEX_CATEGORIES
from src.visualization import (
    plot_model_comparison,
    plot_scores_by_category,
    plot_threshold_curve,
    plot_training_history,
)

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def predict_documents(documents: list[dict], loaded, config: dict, min_confidence: float) -> list[set]:
    """Run one model over many documents and return the predicted spans."""
    return [
        spans_from_detections(
            detect_with_model(
                document["text"],
                loaded,
                min_confidence=min_confidence,
                stride=int(config["model"]["stride"]),
                batch_size=int(config["ner"]["inference_batch_size"]),
            )
        )
        for document in documents
    ]


def evaluate_public_test_set(config: dict) -> tuple[dict, list[dict]]:
    """Part A: fine-tuned model, baseline model and regex layer on public test data."""
    documents = load_processed("test", config)
    threshold = float(config["ner"]["min_confidence"])
    gold_ner = [spans_from_entities(document["entities"]) for document in documents]

    # 1. Our fine-tuned model
    fine_tuned = load_ner_model()
    if fine_tuned.is_fallback:
        raise SystemExit("The fine-tuned model is missing. Run scripts/02_train_model.py first.")
    predicted = predict_documents(documents, fine_tuned, config, threshold)
    fine_tuned_scores = evaluate_entities(gold_ner, predicted, NER_CATEGORIES)
    fine_tuned_scores["character_recall"] = character_recall(gold_ner, predicted, NER_CATEGORIES)

    errors: list[dict] = []
    for document, gold, prediction in zip(documents[:150], gold_ner, predicted):
        errors.extend(collect_errors(document["text"], gold, prediction, document["uid"][:8]))

    # 2. Off-the-shelf baseline (it has no ADDRESS label)
    baseline_name = config["ner"]["fallback_model"]
    baseline = load_model_from(baseline_name, config["model"]["max_length"], is_fallback=True)
    baseline_categories = ("PERSON", "ORG", "LOCATION")
    baseline_predicted = predict_documents(documents, baseline, config, threshold)
    baseline_scores = evaluate_entities(gold_ner, baseline_predicted, baseline_categories)
    baseline_scores["character_recall"] = character_recall(gold_ner, baseline_predicted, baseline_categories)

    # 3. Regex layer against the dataset's own email / phone / card labels
    regex_categories = ("EMAIL", "PHONE", "CREDIT_CARD")
    gold_regex = [spans_from_entities(document["regex_entities"]) for document in documents]
    regex_predicted = [spans_from_detections(detect_regex(document["text"])) for document in documents]
    regex_scores = evaluate_entities(gold_regex, regex_predicted, regex_categories)
    regex_scores["character_recall"] = character_recall(gold_regex, regex_predicted, regex_categories)

    results = {
        "documents": len(documents),
        "min_confidence": threshold,
        "fine_tuned_model": fine_tuned_scores,
        "baseline_model": {"name": baseline_name, **baseline_scores},
        "regex_layer": regex_scores,
    }
    return results, errors


def load_synthetic(split: str, config: dict) -> list[dict]:
    """Read the synthetic documents of one split together with their gold labels."""
    gold_dir = resolve_path(config["evaluation"]["gold_dir"])
    synthetic_dir = resolve_path(config["evaluation"]["synthetic_dir"])
    with open(gold_dir / "ner_gold.json", encoding="utf-8") as file:
        ner_gold = json.load(file)
    with open(gold_dir / "regex_gold.json", encoding="utf-8") as file:
        regex_gold = json.load(file)

    documents = []
    for document_id, entry in ner_gold.items():
        if entry["split"] != split:
            continue
        text = (synthetic_dir / split / f"{document_id}.txt").read_text(encoding="utf-8")
        documents.append(
            {
                "id": document_id,
                "text": text,
                "entities": entry["entities"] + regex_gold[document_id]["entities"],
            }
        )
    return documents


def evaluate_synthetic(config: dict) -> tuple[dict, list[dict]]:
    """Part B: threshold study on the dev part, full pipeline on the test part."""
    model = load_ner_model()

    # 4. Development set: how does the confidence threshold change the result?
    dev_documents = load_synthetic("dev", config)
    dev_gold = [spans_from_entities(document["entities"]) for document in dev_documents]
    threshold_rows = []
    for threshold in THRESHOLDS:
        predicted = predict_documents(dev_documents, model, config, threshold)
        overall = evaluate_entities(dev_gold, predicted, NER_CATEGORIES)["overall"]
        threshold_rows.append({"threshold": threshold, **{key: overall[key] for key in ("precision", "recall", "f1")}})

    # 5. Held-out test set: the complete pipeline, exactly as the app runs it.
    test_documents = load_synthetic("test", config)
    test_gold = [spans_from_entities(document["entities"]) for document in test_documents]
    predicted, errors = [], []
    for document, gold in zip(test_documents, test_gold):
        result = scan(load_input(text=document["text"]), config)
        spans = spans_from_detections(result.detections)
        predicted.append(spans)
        errors.extend(collect_errors(document["text"], gold, spans, document["id"]))

    pipeline_scores = evaluate_entities(test_gold, predicted, ALL_CATEGORIES)
    pipeline_scores["character_recall"] = character_recall(test_gold, predicted, ALL_CATEGORIES)
    results = {
        "dev_documents": len(dev_documents),
        "test_documents": len(test_documents),
        "threshold_study_on_dev": threshold_rows,
        "min_confidence_used": float(config["ner"]["min_confidence"]),
        "full_pipeline_on_test": pipeline_scores,
        "ai_layer_only": evaluate_entities(test_gold, predicted, NER_CATEGORIES)["overall"],
        "regex_layer_only": evaluate_entities(test_gold, predicted, REGEX_CATEGORIES)["overall"],
    }
    return results, errors


def write_errors(path: Path, rows: list[dict]) -> None:
    """Save an error-analysis table as CSV (synthetic / public data only)."""
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["document", "error", "expected", "predicted", "context"])
        writer.writeheader()
        writer.writerows(rows)


def save_figures(public: dict, synthetic: dict, config: dict) -> None:
    """Create the evaluation figures."""
    figures_dir = resolve_path(config["data"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])

    figures = {
        "07_test_scores_by_category.png": plot_scores_by_category(
            public["fine_tuned_model"]["per_category"], "Fine-tuned model on the public test set (strict entity-level)"
        ),
        "08_model_comparison.png": plot_model_comparison(
            {"PrivacyLens (fine-tuned)": public["fine_tuned_model"], f"Baseline ({public['baseline_model']['name']})": public["baseline_model"]},
            NER_CATEGORIES,
        ),
        "09_threshold_study.png": plot_threshold_curve(synthetic["threshold_study_on_dev"], synthetic["min_confidence_used"]),
        "10_synthetic_test_scores.png": plot_scores_by_category(
            synthetic["full_pipeline_on_test"]["per_category"], "Complete pipeline on the held-out synthetic documents"
        ),
        "11_regex_scores.png": plot_scores_by_category(
            public["regex_layer"]["per_category"], "Regex layer on the public test set"
        ),
    }
    history_path = metrics_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, encoding="utf-8") as file:
            figures["06_training_history.png"] = plot_training_history(json.load(file)["history"])

    for name, figure in figures.items():
        figure.savefig(figures_dir / name, dpi=130)


def print_scores(title: str, scores: dict) -> None:
    print(f"\n{title}")
    for category, values in scores["per_category"].items():
        if values["support"]:
            print(f"  {category:<12} P={values['precision']:.3f} R={values['recall']:.3f} F1={values['f1']:.3f} (n={values['support']})")
    overall = scores["overall"]
    print(f"  {'OVERALL':<12} P={overall['precision']:.3f} R={overall['recall']:.3f} F1={overall['f1']:.3f}")
    if "character_recall" in scores:
        print(f"  Character-level recall: {scores['character_recall']['overall']}")


def main() -> int:
    config = load_config()
    setup_logging(config)
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    public, public_errors = evaluate_public_test_set(config)
    synthetic, synthetic_errors = evaluate_synthetic(config)

    with open(metrics_dir / "evaluation_results.json", "w", encoding="utf-8") as file:
        json.dump({"public_test_set": public, "synthetic_documents": synthetic}, file, indent=2)
    write_errors(metrics_dir / "error_analysis_public_test.csv", public_errors)
    write_errors(metrics_dir / "error_analysis_synthetic_test.csv", synthetic_errors)
    save_figures(public, synthetic, config)

    print("=== Evaluation finished ===")
    print_scores("A1. Fine-tuned model, public test set", public["fine_tuned_model"])
    print_scores(f"A2. Baseline {public['baseline_model']['name']}, public test set", public["baseline_model"])
    print_scores("A3. Regex layer, public test set", public["regex_layer"])
    print("\nB4. Threshold study (synthetic dev set):")
    for row in synthetic["threshold_study_on_dev"]:
        print(f"  {row}")
    print_scores("B5. Complete pipeline, synthetic held-out test set", synthetic["full_pipeline_on_test"])
    print(f"\nErrors listed: {len(public_errors)} (public sample), {len(synthetic_errors)} (synthetic test)")
    print("Details: reports/metrics/evaluation_results.json, figures in reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
