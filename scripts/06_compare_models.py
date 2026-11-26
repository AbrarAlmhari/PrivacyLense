"""Step 6: compare trained models on the same documents.

Usage (from the project folder):
    python scripts/06_compare_models.py --stage validation --model v1=models/archive/privacylens-ner-v1 --model v2=models/privacylens-ner
    python scripts/06_compare_models.py --stage test       --model v1=... --model v2=...

Two stages, on purpose:

* ``validation`` uses only development data (Nemotron validation split, the
  validation templates of the ORG supplement, the synthetic dev documents).
  A model is CHOSEN with these numbers only.
* ``test`` is run once, after the choice, on the held-out test sets
  (Nemotron test split, the test templates of the ORG supplement, and the
  complete pipeline on the synthetic test documents). It only REPORTS.

Results: reports/metrics/model_comparison_<stage>.json (+ a figure for the test stage).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path, setup_logging
from src.data_preparation import load_processed
from src.evaluation import evaluate_entities, load_synthetic, spans_from_detections, spans_from_entities
from src.file_processor import load_input
from src.ner_detector import detect_with_model, load_model_from
from src.pipeline import scan
from src.schemas import ALL_CATEGORIES, NER_CATEGORIES
from src.visualization import plot_model_comparison


def read_jsonl(name: str, config: dict) -> list[dict]:
    path = resolve_path(config["data"]["processed_dir"]) / name
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def model_scores(documents: list[dict], loaded, config: dict) -> dict:
    """Strict entity scores of the AI model alone (PERSON, ORG, LOCATION, ADDRESS)."""
    gold = [spans_from_entities([e for e in d["entities"] if e["label"] in NER_CATEGORIES]) for d in documents]
    predicted = [
        spans_from_detections(
            detect_with_model(
                d["text"],
                loaded,
                min_confidence=float(config["ner"]["min_confidence"]),
                stride=int(config["model"]["stride"]),
                batch_size=int(config["ner"]["inference_batch_size"]),
            )
        )
        for d in documents
    ]
    return evaluate_entities(gold, predicted, NER_CATEGORIES)


def pipeline_scores(documents: list[dict], loaded, config: dict) -> dict:
    """Strict entity scores of the complete pipeline (rules + AI + merge), as in the app."""
    gold = [spans_from_entities(d["entities"]) for d in documents]
    predicted = [spans_from_detections(scan(load_input(text=d["text"]), config, model=loaded).detections) for d in documents]
    return evaluate_entities(gold, predicted, ALL_CATEGORIES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PrivacyLens models.")
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    parser.add_argument("--model", action="append", required=True, help="name=folder, can be given several times")
    arguments = parser.parse_args()

    config = load_config()
    setup_logging(config)
    if arguments.stage == "validation":
        sets = {
            "nemotron_validation": (load_processed("validation", config), "model"),
            "org_supplement_validation": (read_jsonl("org_supplement_validation.jsonl", config), "model"),
            "synthetic_dev": (load_synthetic("dev", config), "model"),
        }
    else:
        sets = {
            "nemotron_test": (load_processed("test", config), "model"),
            "org_supplement_test": (read_jsonl("org_supplement_test.jsonl", config), "model"),
            "synthetic_test_ai_layer": (load_synthetic("test", config), "model"),
            "synthetic_test_full_pipeline": (load_synthetic("test", config), "pipeline"),
        }

    results: dict[str, dict] = {}
    for item in arguments.model:
        name, folder = item.split("=", 1)
        loaded = load_model_from(str(resolve_path(folder)), config["model"]["max_length"], name=name)
        results[name] = {"folder": folder}
        for set_name, (documents, kind) in sets.items():
            started = time.time()
            scorer = model_scores if kind == "model" else pipeline_scores
            scores = scorer(documents, loaded, config)
            results[name][set_name] = {"documents": len(documents), **scores}
            overall, org = scores["overall"], scores["per_category"]["ORG"]
            print(
                f"{name:<12} {set_name:<30} overall F1={overall['f1']:.4f}  ORG P={org['precision']:.3f} "
                f"R={org['recall']:.3f} F1={org['f1']:.4f} (n={org['support']})  [{time.time() - started:.0f}s]"
            )

    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    with open(metrics_dir / f"model_comparison_{arguments.stage}.json", "w", encoding="utf-8") as file:
        json.dump({"stage": arguments.stage, "min_confidence": config["ner"]["min_confidence"], "models": results}, file, indent=2)

    if arguments.stage == "test":
        save_test_figures(results, config)
    print(f"Saved reports/metrics/model_comparison_{arguments.stage}.json")
    return 0


def save_test_figures(results: dict, config: dict) -> None:
    """One bar chart per test set: F1 per category, the first model in grey."""
    figures_dir = resolve_path(config["data"]["figures_dir"])
    for set_name, figure_name, title in (
        ("nemotron_test", "13_v1_v2_public_test.png", "Public test set (3,000 Nemotron-PII documents)"),
        ("org_supplement_test", "14_v1_v2_org_supplement_test.png", "Team's ORG documents: 4 held-out templates (100 documents)"),
        ("synthetic_test_ai_layer", "15_v1_v2_synthetic_test.png", "Synthetic test documents (40), AI model only"),
    ):
        scores = {name: results[name][set_name] for name in results}
        categories = tuple(c for c in NER_CATEGORIES if any(s["per_category"][c]["support"] for s in scores.values()))
        figure = plot_model_comparison(scores, categories, title=title, colors=["#b6bfcc", "#2a9d8f", "#457b9d"])
        figure.savefig(figures_dir / figure_name, dpi=130)


if __name__ == "__main__":
    sys.exit(main())
