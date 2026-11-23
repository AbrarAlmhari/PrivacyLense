"""AI-based detection: Named Entity Recognition (NER) with a transformer.

This module loads the model that we fine-tuned in ``train_ner.py`` and uses
it to find PERSON, ORG, LOCATION and ADDRESS entities in the canonical text.

How a token-classification model works, in short:

1. The tokenizer splits the text into small pieces ("tokens") and remembers
   the character offsets of each token.
2. The model gives every token a label such as ``B-PERSON`` (beginning of a
   person name), ``I-PERSON`` (inside a person name) or ``O`` (nothing).
3. We glue neighbouring tokens with the same entity type back together and
   use the saved offsets to know exactly where the entity is in the text.

If the fine-tuned model folder is missing, a public pretrained NER model
(``ner.fallback_model`` in config.yaml) is used instead, so the app still
works on a fresh machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.config import load_config, resolve_path
from src.schemas import NER_CATEGORIES, Detection

logger = logging.getLogger(__name__)

# Model label -> application category. Our fine-tuned model already uses the
# application names; the short names (PER, LOC) belong to the fallback model.
LABEL_TO_CATEGORY = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "ADDRESS": "ADDRESS",
}


class ModelNotAvailableError(Exception):
    """Raised when no NER model can be loaded."""


@dataclass
class LoadedModel:
    """A model that is ready for inference."""

    name: str
    tokenizer: object
    model: torch.nn.Module
    device: torch.device
    id_to_label: dict[int, str]
    max_length: int
    is_fallback: bool


def load_model_from(source: str, max_length: int, name: str | None = None, is_fallback: bool = False) -> LoadedModel:
    """Load a token-classification model from a local folder or the Hugging Face Hub."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForTokenClassification.from_pretrained(source)
    except Exception as error:  # network problems, damaged files, ...
        raise ModelNotAvailableError(
            f"The NER model '{source}' could not be loaded. If this is the first run, "
            "check your internet connection or train the model with: python scripts/02_train_model.py"
        ) from error

    # Use the GPU when there is one; otherwise the CPU works fine (just slower).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # inference mode: switches off dropout

    logger.info("NER model loaded: %s on %s", source, device)
    return LoadedModel(
        name=name or source,
        tokenizer=tokenizer,
        model=model,
        device=device,
        id_to_label={int(index): label for index, label in model.config.id2label.items()},
        max_length=min(int(max_length), 512),
        is_fallback=is_fallback,
    )


@lru_cache(maxsize=1)
def load_ner_model() -> LoadedModel:
    """Load the fine-tuned model (or the fallback) once and cache it."""
    config = load_config()
    model_dir = resolve_path(config["model"]["output_dir"])
    fallback_name = config["ner"].get("fallback_model")
    max_length = config["model"]["max_length"]

    hub_model = config["ner"].get("hub_model")

    if (model_dir / "config.json").exists():
        return load_model_from(str(model_dir), max_length, name=model_dir.name)
    if hub_model:
        # Our own fine-tuned model, downloaded from the Hugging Face Hub.
        logger.info("Local model not found. Downloading fine-tuned model %s from the Hub.", hub_model)
        return load_model_from(hub_model, max_length)
    if fallback_name:
        logger.warning("Fine-tuned model not found in %s. Using fallback model %s.", model_dir, fallback_name)
        return load_model_from(fallback_name, max_length, is_fallback=True)
    raise ModelNotAvailableError(
        f"No NER model found in '{model_dir}'. Train it with: python scripts/02_train_model.py"
    )


def predict_token_labels(text: str, loaded: LoadedModel, stride: int, batch_size: int) -> list[tuple[int, int, str, float]]:
    """Run the model and return ``(start, end, label, probability)`` per token.

    Long texts do not fit into the model at once, so the tokenizer cuts them
    into overlapping windows. A token that appears in two windows keeps the
    prediction with the higher probability.
    """
    encoded = loaded.tokenizer(
        text,
        truncation=True,
        max_length=loaded.max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping", None)

    best: dict[tuple[int, int], tuple[str, float]] = {}
    window_count = encoded["input_ids"].shape[0]

    with torch.no_grad():  # no gradients needed for inference
        for batch_start in range(0, window_count, batch_size):
            batch = {
                name: tensor[batch_start : batch_start + batch_size].to(loaded.device)
                for name, tensor in encoded.items()
            }
            logits = loaded.model(**batch).logits
            probabilities = torch.softmax(logits, dim=-1)
            top_probabilities, top_ids = probabilities.max(dim=-1)

            top_probabilities = top_probabilities.cpu().tolist()
            top_ids = top_ids.cpu().tolist()
            batch_offsets = offsets[batch_start : batch_start + batch_size].tolist()

            for window_offsets, window_ids, window_probabilities in zip(batch_offsets, top_ids, top_probabilities):
                for (start, end), label_id, probability in zip(window_offsets, window_ids, window_probabilities):
                    if start == end:  # special tokens ([CLS], [SEP], padding)
                        continue
                    key = (start, end)
                    if key not in best or probability > best[key][1]:
                        best[key] = (loaded.id_to_label[label_id], probability)

    return [(start, end, label, probability) for (start, end), (label, probability) in sorted(best.items())]


def group_tokens_into_entities(text: str, token_labels: list[tuple[int, int, str, float]]) -> list[dict]:
    """Glue labelled tokens into entities using the BIO scheme.

    ``B-X`` starts a new entity of type X, ``I-X`` continues it, ``O`` ends it.
    """
    entities: list[dict] = []
    current: dict | None = None

    for start, end, label, probability in token_labels:
        if label == "O":
            current = None
            continue

        prefix, _, entity_type = label.partition("-")
        gap = text[current["end"] : start] if current else ""
        continues_current = (
            current is not None
            and current["type"] == entity_type
            and gap.strip() == ""
            and "\n\n" not in gap
            # "I-" continues an entity; a "B-" in the middle of a word
            # (no gap) is also treated as a continuation.
            and (prefix == "I" or gap == "")
        )
        if continues_current:
            current["end"] = end
            current["probabilities"].append(probability)
        else:
            current = {"type": entity_type, "start": start, "end": end, "probabilities": [probability]}
            entities.append(current)

    return entities


def _expand_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """Make sure an entity never starts or ends in the middle of a word."""
    while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum() and text[end - 1].isalnum():
        end += 1
    # Trim whitespace that slipped in at the edges.
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def detect_with_model(
    text: str,
    loaded: LoadedModel,
    min_confidence: float = 0.5,
    stride: int = 32,
    batch_size: int = 16,
) -> list[Detection]:
    """Run a loaded model on ``text`` and return clean ``Detection`` objects.

    This function is shared by the app (``detect_ner``) and by the evaluation
    scripts, so the numbers we report describe exactly what the app does.
    """
    if not text or not text.strip():
        return []

    token_labels = predict_token_labels(text, loaded, stride=stride, batch_size=batch_size)

    detections: list[Detection] = []
    for entity in group_tokens_into_entities(text, token_labels):
        category = LABEL_TO_CATEGORY.get(entity["type"])
        if category not in NER_CATEGORIES:
            continue  # e.g. the MISC label of the fallback model
        confidence = sum(entity["probabilities"]) / len(entity["probabilities"])
        if confidence < min_confidence:
            continue
        start, end = _expand_to_word_boundaries(text, entity["start"], entity["end"])
        if end - start < 2:
            continue
        # Skip an entity that now overlaps the previous one (after expansion).
        if detections and start < detections[-1].end:
            continue
        detections.append(
            Detection(
                id=f"n-{len(detections) + 1:04d}",
                category=category,
                text=text[start:end],
                start=start,
                end=end,
                source="ner",
                confidence=round(confidence, 4),
                raw_label=entity["type"],
            )
        )
    return detections


def detect_ner(text: str, config: dict | None = None) -> list[Detection]:
    """Find PERSON, ORG, LOCATION and ADDRESS entities with the AI model.

    Args:
        text: The canonical text.
        config: Optional configuration dictionary.

    Returns:
        Detections sorted by position; offsets refer to ``text``.

    Raises:
        ModelNotAvailableError: if no model can be loaded.
    """
    if not text or not text.strip():
        return []

    config = config or load_config()
    detections = detect_with_model(
        text,
        load_ner_model(),
        min_confidence=float(config["ner"]["min_confidence"]),
        stride=int(config["model"]["stride"]),
        batch_size=int(config["ner"]["inference_batch_size"]),
    )
    # Privacy rule: log counts only.
    logger.info("NER layer found %d item(s)", len(detections))
    return detections
