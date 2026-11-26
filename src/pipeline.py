"""The complete scan workflow in one place.

``scan`` connects the modules in the order of the Transform step:

    canonical text -> regex layer + AI layer -> merge -> propagate -> risk score

The Streamlit app, the tests and the evaluation script all call this one
function, so they always behave the same way.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.config import load_config
from src.ner_detector import LoadedModel, detect_ner, detect_with_model, load_ner_model
from src.regex_detector import detect_regex
from src.risk_analyzer import merge_detections, propagate_detections, score
from src.schemas import Detection, LoadResult, RiskResult

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Everything produced by one scan."""

    detections: list[Detection]
    risk: RiskResult
    notes: list[str] = field(default_factory=list)  # decisions made on the way
    model_name: str = ""
    used_fallback_model: bool = False
    seconds: float = 0.0


def scan(
    load_result: LoadResult,
    config: dict | None = None,
    use_ner: bool = True,
    model: LoadedModel | None = None,
) -> ScanResult:
    """Detect sensitive information in a loaded document and score the risk.

    Args:
        load_result: Output of ``file_processor.load_input``.
        config: Optional configuration dictionary.
        use_ner: Set to False to run the regex layer only (used in tests).
        model: Use this loaded model instead of the app's model (used to compare models).

    Raises:
        ModelNotAvailableError: if ``use_ner`` is True and no model can be loaded.
    """
    config = config or load_config()
    text = load_result.canonical_text
    started = time.time()
    notes: list[str] = []

    regex_detections = detect_regex(text)

    ner_detections: list[Detection] = []
    model_name, used_fallback = "", False
    if use_ner and model is not None:
        model_name, used_fallback = model.name, model.is_fallback
        ner_detections = detect_with_model(
            text,
            model,
            min_confidence=float(config["ner"]["min_confidence"]),
            stride=int(config["model"]["stride"]),
            batch_size=int(config["ner"]["inference_batch_size"]),
        )
    elif use_ner:
        loaded = load_ner_model()
        model_name, used_fallback = loaded.name, loaded.is_fallback
        ner_detections = detect_ner(text, config)
        if used_fallback:
            notes.append(
                f"The fine-tuned PrivacyLens model was not found, so the public model "
                f"'{model_name}' was used. ADDRESS detection is not available with it."
            )

    detections = merge_detections(regex_detections, ner_detections, notes)
    if config["redaction"].get("propagate_repeated_names", True):
        detections = propagate_detections(text, detections, notes=notes)

    risk = score(detections, config)
    notes.extend(risk.notes)

    seconds = round(time.time() - started, 2)
    logger.info("Scan finished: %d detection(s), risk %d (%s), %.2fs", len(detections), risk.score, risk.level, seconds)
    return ScanResult(detections, risk, notes, model_name, used_fallback, seconds)
