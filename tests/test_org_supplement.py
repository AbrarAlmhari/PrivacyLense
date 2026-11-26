"""Tests for the team's extra ORG documents (src/org_supplement.py)."""

import copy
import json

import pytest

from src.config import load_config, resolve_path
from src.data_preparation import DataError
from src.org_supplement import complete_labels, entity_problems, organisation_family, split_by_template
from src.train_ner import load_extra_training_documents

CONFIG = load_config()
SETTINGS = CONFIG["org_supplement"]


def make_document(text, entities, document_type="Invoice", document_format="structured"):
    return {
        "uid": "x",
        "domain": "Finance",
        "document_type": document_type,
        "document_format": document_format,
        "locale": "intl",
        "text": text,
        "entities": entities,
    }


# ------------------------------------------------------------ label completion
def test_missing_person_and_city_labels_are_added():
    text = "Supplier: Aster Labs\nContact Person: Sarah Mitchell\nOffice: London"
    document = make_document(text, [{"start": 10, "end": 20, "label": "ORG"}])
    completed = complete_labels(document, ["Sarah Mitchell"], ["London"])
    found = [(e["label"], text[e["start"] : e["end"]]) for e in completed["entities"]]
    assert found == [("ORG", "Aster Labs"), ("PERSON", "Sarah Mitchell"), ("LOCATION", "London")]


def test_city_inside_a_company_name_stays_part_of_the_org():
    text = "Riyadh Smart Services opened an office in Riyadh."
    document = make_document(text, [{"start": 0, "end": 21, "label": "ORG"}])
    completed = complete_labels(document, [], ["Riyadh"])
    locations = [(e["start"], e["end"]) for e in completed["entities"] if e["label"] == "LOCATION"]
    assert locations == [(42, 48)]


def test_only_whole_words_are_labelled():
    document = make_document("Londoner Group and London", [])
    completed = complete_labels(document, [], ["London"])
    assert [(e["start"], e["end"]) for e in completed["entities"]] == [(19, 25)]


# ------------------------------------------------------------------ validation
def test_valid_document_has_no_problems():
    assert entity_problems(make_document("Aster Labs", [{"start": 0, "end": 10, "label": "ORG"}])) == []


@pytest.mark.parametrize(
    "entities, expected",
    [
        ([{"start": 0, "end": 99, "label": "ORG"}], "offsets"),
        ([{"start": 0, "end": 5, "label": "COMPANY"}], "unknown label"),
        ([{"start": 0, "end": 6, "label": "ORG"}], "spaces"),
        ([{"start": 0, "end": 5, "label": "ORG"}, {"start": 2, "end": 10, "label": "ORG"}], "overlapping"),
    ],
)
def test_broken_documents_are_reported(entities, expected):
    problems = entity_problems(make_document("Aster Labs", entities))
    assert problems and expected in problems[0]


def test_family_of_a_company_name():
    assert organisation_family("Al Noor Networks") == "Al Noor"
    assert organisation_family("Aster Digital Systems") == "Aster"


# ------------------------------------------------------------- template split
def test_whole_templates_go_to_one_split_and_each_split_has_both_formats():
    documents = []
    for index in range(10):
        for document_format in ("structured", "unstructured"):
            for copy_number in range(3):
                documents.append(make_document(f"text {index} {document_format} {copy_number}", [], f"T{index}-{document_format}", document_format))
    config = copy.deepcopy(CONFIG)
    splits = split_by_template(documents, config)

    templates = {name: {d["document_type"] for d in items} for name, items in splits.items()}
    assert not templates["train"] & templates["test"]
    assert not templates["train"] & templates["validation"]
    assert not templates["validation"] & templates["test"]
    assert len(templates["test"]) == SETTINGS["test_templates"]
    assert len(templates["validation"]) == SETTINGS["validation_templates"]
    for name in ("validation", "test"):
        assert {d["document_format"] for d in splits[name]} == {"structured", "unstructured"}
    assert split_by_template(documents, config) == splits  # same seed, same result


def test_missing_extra_training_file_gives_a_clear_error():
    config = copy.deepcopy(CONFIG)
    config["model"]["extra_train_files"] = ["data/processed/does_not_exist.jsonl"]
    with pytest.raises(DataError):
        load_extra_training_documents(config)


# --------------------------------------------- the prepared files themselves
PROCESSED = resolve_path(CONFIG["data"]["processed_dir"])
needs_files = pytest.mark.skipif(
    not (PROCESSED / "org_supplement_test.jsonl").exists(),
    reason="Run scripts/05_prepare_org_supplement.py first",
)


def read_split(name):
    with open(PROCESSED / f"org_supplement_{name}.jsonl", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


@needs_files
def test_prepared_files_keep_templates_apart_and_label_every_listed_name():
    splits = {name: read_split(name) for name in ("train", "validation", "test")}
    assert sum(len(items) for items in splits.values()) == 500
    seen_templates = set()
    for items in splits.values():
        templates = {d["document_type"] for d in items}
        assert not templates & seen_templates
        seen_templates |= templates
        for document in items:
            assert entity_problems(document) == []
            labelled = {document["text"][e["start"] : e["end"]] for e in document["entities"]}
            for name in SETTINGS["person_names"]:
                if name in document["text"]:
                    assert name in labelled
