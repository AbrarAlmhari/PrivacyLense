"""Fine-tune a pretrained transformer for PII entity recognition.

We start from ``distilbert-base-cased`` (a small pretrained language model
from the Hugging Face Hub) and train it on our prepared dataset so that it
labels every token as PERSON / ORG / LOCATION / ADDRESS / nothing.

The training loop is written in plain PyTorch on purpose, so every step
from the lectures is visible:

    Dataset -> DataLoader -> forward pass -> loss -> backward() -> optimizer step

Run it with:  ``python scripts/02_train_model.py``
"""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.config import resolve_path
from src.data_preparation import load_processed
from src.evaluation import evaluate_entities, spans_from_detections, spans_from_entities
from src.ner_detector import LoadedModel, detect_with_model
from src.schemas import NER_CATEGORIES

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100  # PyTorch's loss function skips positions with this label


def set_seed(seed: int) -> None:
    """Fix all random number generators so training is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_list(categories: tuple[str, ...] = NER_CATEGORIES) -> list[str]:
    """BIO label scheme: ``O``, then ``B-X`` and ``I-X`` for every category."""
    labels = ["O"]
    for category in categories:
        labels += [f"B-{category}", f"I-{category}"]
    return labels


# ------------------------------------------------------------ data encoding
def encode_documents(documents: list[dict], tokenizer, label_to_id: dict, max_length: int, stride: int) -> list[dict]:
    """Turn documents with character-level entities into labelled token windows.

    The tokenizer cuts every text into windows of at most ``max_length``
    tokens (long documents give several overlapping windows). It also returns
    the character offsets of every token, which we use to give each token
    its BIO label.
    """
    texts = [document["text"] for document in documents]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    windows = []
    for window_index, document_index in enumerate(encoded["overflow_to_sample_mapping"]):
        document = documents[document_index]
        # entity_at[i] = index of the entity that covers character i (or -1).
        entity_at = [-1] * len(document["text"])
        for entity_index, entity in enumerate(document["entities"]):
            for position in range(entity["start"], entity["end"]):
                entity_at[position] = entity_index

        labels = []
        for start, end in encoded["offset_mapping"][window_index]:
            if start == end:  # special tokens such as [CLS] and [SEP]
                labels.append(IGNORE_INDEX)
                continue
            entity_index = entity_at[start]
            if entity_index == -1:
                labels.append(label_to_id["O"])
                continue
            entity = document["entities"][entity_index]
            prefix = "B" if start == entity["start"] else "I"
            labels.append(label_to_id[f"{prefix}-{entity['label']}"])

        windows.append(
            {
                "input_ids": encoded["input_ids"][window_index],
                "attention_mask": encoded["attention_mask"][window_index],
                "labels": labels,
            }
        )
    return windows


class TokenWindowDataset(Dataset):
    """A PyTorch Dataset: it knows how to return ONE training sample."""

    def __init__(self, windows: list[dict]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        return self.windows[index]


def make_collate_function(pad_token_id: int):
    """Create the function the DataLoader uses to build one batch.

    Windows have different lengths, so we pad them to the longest window in
    the batch and stack them into tensors of shape [batch_size, tokens].
    """

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        longest = max(len(sample["input_ids"]) for sample in batch)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (longest - len(values))

        return {
            "input_ids": torch.tensor([pad(sample["input_ids"], pad_token_id) for sample in batch]),
            "attention_mask": torch.tensor([pad(sample["attention_mask"], 0) for sample in batch]),
            "labels": torch.tensor([pad(sample["labels"], IGNORE_INDEX) for sample in batch]),
        }

    return collate


# --------------------------------------------------------------- validation
def average_loss(model, loader: DataLoader, device: torch.device) -> float:
    """Average loss over a DataLoader (no training, no gradients)."""
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                total += model(**batch).loss.item()
            batches += 1
    return total / max(batches, 1)


def entity_scores(loaded: LoadedModel, documents: list[dict], config: dict) -> dict:
    """Strict entity-level precision/recall/F1 using the app's own inference code."""
    gold = [spans_from_entities(document["entities"]) for document in documents]
    predicted = [
        spans_from_detections(
            detect_with_model(
                document["text"],
                loaded,
                min_confidence=float(config["ner"]["min_confidence"]),
                stride=int(config["model"]["stride"]),
                batch_size=int(config["ner"]["inference_batch_size"]),
            )
        )
        for document in documents
    ]
    return evaluate_entities(gold, predicted, NER_CATEGORIES)


# ----------------------------------------------------------------- training
def train(config: dict) -> dict:
    """Fine-tune the model, keep the best epoch and save it with its history."""
    settings = config["model"]
    set_seed(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision: faster, less GPU memory
    logger.info("Training on %s", torch.cuda.get_device_name(0) if use_amp else "CPU")

    labels = build_label_list()
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(settings["base_model"])
    model = AutoModelForTokenClassification.from_pretrained(
        settings["base_model"], num_labels=len(labels), id2label=id_to_label, label2id=label_to_id
    ).to(device)

    train_documents = load_processed("train", config)
    validation_documents = load_processed("validation", config)
    max_length, stride = settings["max_length"], settings["stride"]
    train_windows = encode_documents(train_documents, tokenizer, label_to_id, max_length, stride)
    validation_windows = encode_documents(validation_documents, tokenizer, label_to_id, max_length, stride)
    logger.info("Training windows: %d, validation windows: %d", len(train_windows), len(validation_windows))

    collate = make_collate_function(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(settings["seed"])
    train_loader = DataLoader(
        TokenWindowDataset(train_windows), batch_size=settings["batch_size"], shuffle=True, collate_fn=collate, generator=generator
    )
    validation_loader = DataLoader(
        TokenWindowDataset(validation_windows), batch_size=settings["batch_size"] * 2, shuffle=False, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    total_steps = len(train_loader) * settings["epochs"]
    warmup_steps = int(total_steps * settings["warmup_ratio"])

    def learning_rate_factor(step: int) -> float:
        """Linear warm-up, then linear decay to zero."""
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    loaded = LoadedModel("in-training", tokenizer, model, device, id_to_label, max_length, False)
    output_dir = resolve_path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_f1 = -1.0
    started = time.time()

    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model(**batch).loss           # forward pass
            scaler.scale(loss).backward()            # autograd computes gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)                   # update the weights
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % 200 == 0:
                logger.info("Epoch %d step %d/%d - loss %.4f", epoch, step, len(train_loader), running_loss / step)

        train_loss = running_loss / len(train_loader)
        validation_loss = average_loss(model, validation_loader, device)
        scores = entity_scores(loaded, validation_documents, config)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "validation_loss": round(validation_loss, 5),
            "validation_precision": scores["overall"]["precision"],
            "validation_recall": scores["overall"]["recall"],
            "validation_f1": scores["overall"]["f1"],
            "elapsed_minutes": round((time.time() - started) / 60, 2),
        }
        history.append(epoch_record)
        logger.info("Epoch summary: %s", epoch_record)

        # Keep only the best model (highest validation F1).
        if scores["overall"]["f1"] > best_f1:
            best_f1 = scores["overall"]["f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("New best model saved to %s (F1 %.4f)", output_dir, best_f1)

    training_info = {
        "base_model": settings["base_model"],
        "labels": labels,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_minutes": round((time.time() - started) / 60, 2),
        "device": torch.cuda.get_device_name(0) if use_amp else "cpu",
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "train_windows": len(train_windows),
        "hyperparameters": {key: settings[key] for key in ("max_length", "stride", "batch_size", "learning_rate", "weight_decay", "warmup_ratio", "epochs", "seed")},
        "best_validation_f1": best_f1,
        "history": history,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    _write_json(output_dir / "training_info.json", training_info)
    _write_json(resolve_path(config["evaluation"]["metrics_dir"]) / "training_history.json", training_info)
    return training_info


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)
