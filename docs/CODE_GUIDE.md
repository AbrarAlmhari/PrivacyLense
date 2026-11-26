# PrivacyLens - Code Guide

A tour of the code for someone who wants to **understand and explain** it.
Suggested reading order: follow one document through the system.

```
schemas.py -> file_processor.py -> regex_detector.py -> ner_detector.py
           -> risk_analyzer.py -> redactor.py -> report_generator.py -> pipeline.py -> app.py
then the "training side":  data_preparation.py -> train_ner.py -> evaluation.py -> synthetic_data.py
```

---

## 1. `src/schemas.py` - the shared vocabulary

Three dataclasses that every module agrees on (the "interface contract" of the plan).

- **`Detection`**: one found item. `category`, `text`, `start`, `end` (character offsets, end is
  exclusive, so `canonical_text[start:end] == text`), `source` (`regex`, `ner` or `propagated`),
  `confidence` (model only), `raw_label` (rule name or model label), `id` (`r-0001`, `n-0001`).
- **`LoadResult`**: the canonical text, statistics, warnings, and for PDFs `char_boxes`
  (one `CharBox` per character: page, line, x0, y0, x1, y1) plus the original bytes.
- **`RiskResult`**: score, level and a per-category breakdown.

*Why dataclasses?* They are simple containers with named fields; a typo in a field name is an error
instead of a silent bug, which is what would happen with plain dictionaries.

## 2. `src/config.py` - configuration and logging

- `load_config()` reads `config/config.yaml` once (cached) and raises `ConfigError` with a clear
  message if the file is missing or broken.
- `setup_logging()` writes to the console and to `outputs/logs/privacylens.log`.
- **Privacy rule used everywhere:** log events and counts, never the text or the detected strings.

## 3. `src/file_processor.py` - Load, Validate, Clean

`load_input(text=None, file=None, file_name=None)` is the only entry point.

1. **Validate**: extension (`.txt` / `.pdf`), size, empty file, real PDF header, password, page limit.
   Each problem has its own exception class (`UnsupportedFileError`, `ScannedPdfError`, ...), all
   children of `InputError`, so the app can catch one type and show the message.
2. **Extract (PDF)**: `page.get_text("rawdict")` gives blocks -> lines -> spans -> characters with a
   bounding box for each character. We append every character to `raw_chars` and its box to
   `raw_boxes`. Separators we insert ourselves (line breaks) get `None` as box.
3. **Clean**: `clean_text_with_map(raw)` walks through the raw text one character at a time,
   decides to keep, replace or skip it, and records `index_map[i]` = position of kept character `i`
   in the raw text. With that map, `char_boxes` for the canonical text is one line:
   `[raw_boxes[raw_index] for raw_index in index_map]`.

*Key idea:* the canonical text and `char_boxes` have the same length. That is the bridge between
"the model found characters 120-135" and "draw a box at these coordinates on page 2".

## 4. `src/regex_detector.py` - the rule layer

- `RULES` is an ordered tuple of `Rule(name, category, pattern, description, validator)`.
- Order = priority. If two rules match overlapping text, the earlier rule wins
  (an IBAN is never also reported as an ID number).
- Look-arounds such as `(?<![\w.,/\-])` stop a rule from matching in the middle of a longer
  number, a price or a date. This is what keeps `1,250.00` or `2026-11-16` from matching.
- `validator` adds a check that a regular expression cannot do: the Luhn checksum for bank cards,
  or "between 9 and 15 digits" for international phone numbers.
- `normalize_digits` maps Arabic-Indic digits (٠-٩) to 0-9 before matching. It is 1 character to
  1 character, so offsets do not move, and the original characters are kept in `Detection.text`.

## 5. `src/ner_detector.py` - the AI layer (inference)

1. `load_ner_model()` loads the tokenizer and model once (cached), moves the model to the GPU if
   there is one, and calls `model.eval()`. Order of preference: local fine-tuned folder ->
   `ner.hub_model` -> public fallback model.
2. `predict_token_labels()`: the tokenizer cuts the text into overlapping windows of 256 tokens
   and returns each token's character offsets. Inside `torch.no_grad()` the model produces logits,
   `softmax` turns them into probabilities, and the best label per token is kept.
3. `group_tokens_into_entities()`: BIO decoding. `B-PERSON` starts an entity, `I-PERSON` continues it.
4. `detect_with_model()`: maps model labels to categories, averages token probabilities into a
   confidence, drops entities below the threshold, and widens an entity to whole words so that half
   a name is never redacted.

## 6. `src/risk_analyzer.py` - merge, propagate, score

- `merge_detections`: regex detections go in first; an NER detection that overlaps one is dropped.
- `propagate_detections`: for each detected PERSON / ORG text, find other exact occurrences with a
  word-boundary regex and add them with `source="propagated"`.
- `score`: unique items per category (after `normalize_for_counting`), base points for the first,
  25% for each extra one, cap per category at 2x base, cap total at 100, then map to a level.

## 7. `src/redactor.py` - removing the information

- `redact` (text): replaces selected detections **from the end backward**. If we went forward,
  the first replacement would change the length and every later offset would be wrong.
- `detection_rectangles`: takes `char_boxes[start:end]`, groups neighbours on the same page and
  line, and unions their boxes -> one rectangle per line.
- `redact_pdf`: adds a redaction annotation per rectangle, deletes links under it,
  `apply_redactions()` removes the text from the page content, metadata is cleared, and the file is
  saved with `garbage=4` so removed objects do not stay inside the file.
- `verify_pdf_redaction`: opens the *new* PDF and extracts text inside each redacted rectangle.
  Anything other than our own white label means failure.

## 8. `src/report_generator.py` and `src/pipeline.py`

- `build_summary` creates the JSON summary: counts, categories, positions, score, notes and
  limitations. The matched strings are only included if `include_sensitive_text=True`.
- `pipeline.scan(load_result)` runs regex -> NER -> merge -> propagate -> score and returns a
  `ScanResult`. The app, the tests and the evaluation all call this same function.
  The optional argument `model=` lets `scripts/06_compare_models.py` run the same workflow with
  another loaded model (for example the previous version), without changing the app's model.

## 9. `app.py` - the Streamlit interface

- Streamlit re-runs the whole script on every click, so results are kept in `st.session_state`
  (`load_result`, `scan_result`, `selected_ids`).
- `@st.cache_resource` keeps the model in memory between re-runs.
- `run_scan` wraps the workflow in `try / except`: `InputError` and `ModelNotAvailableError` show
  their message; anything unexpected is logged and a generic message is shown.
- The tick-box table is `st.data_editor`. The buttons "Select all / Clear all / Apply categories"
  change `selected_ids` and increase `editor_version`, which gives the table a new key so it is
  rebuilt with the new ticks.
- `current_redacted_pdf` builds the redacted PDF only when the selection changes.

---

## The training side

## 10. `src/data_preparation.py` - dataset pipeline

One function per workflow step, called in order by `run_pipeline`:
`download_dataset` / `load_raw` -> `validate` -> `clean` -> `explore` -> `visualize` ->
`transform` -> `export` -> `summarize`. Each step returns a small report dictionary, and
`summarize` stores them together in `reports/metrics/data_summary.json`.

The interesting function is `map_spans_to_entities`: it renames dataset labels with the
`label_map` from the config and merges a first name that is directly followed by a last name
into one `PERSON` entity.

## 11. `src/train_ner.py` - the training loop

Follows the PyTorch lectures step by step:

| Code | Concept |
|---|---|
| `set_seed` | reproducibility (Lecture 13-14) |
| `encode_documents` | tokenization with offsets; each token gets a BIO label id; special tokens get `-100` so the loss ignores them |
| `TokenWindowDataset` | custom `Dataset` with `__len__` and `__getitem__` (Lecture 8) |
| `make_collate_function` | pads a batch to the same length and builds tensors of shape `[batch, tokens]` |
| `DataLoader(..., shuffle=True)` | batching and shuffling (Lecture 8) |
| `model.to(device)` / `batch.to(device)` | CPU vs GPU (Lecture 9) |
| `loss = model(**batch).loss` | forward pass (cross-entropy over token labels) |
| `loss.backward()` | autograd computes gradients (Lecture 9) |
| `optimizer.step()` | AdamW updates the weights |
| `scheduler.step()` | learning rate: linear warm-up, then linear decay |
| `torch.autocast` + `GradScaler` | mixed precision: faster and less GPU memory |
| `model.save_pretrained` | serialization of the best epoch (Lecture 9) |
| `training_info.json` | experiment record: hyperparameters, versions, history (Lecture 14) |

After each epoch the model is evaluated on the validation set with the same inference code the app
uses (`detect_with_model`), and only the epoch with the best F1 is saved.

Extra training data (version 2): `load_extra_training_documents` reads the files listed under
`model.extra_train_files` in the config and adds them to the Nemotron training documents,
`model.extra_train_repeat` times each. `scripts/02_train_model.py` has options for experiments:
`--output-dir` (save somewhere else, so the app's model is not replaced), `--extra-repeat` and
`--no-extra`. Only a run that saves to `models/privacylens-ner` updates
`reports/metrics/training_history.json`.

## 11b. `src/org_supplement.py` - the team's extra ORG documents

The same workflow idea as `data_preparation.py`, for the 500 documents in
`data/extra/org_supplement_raw.jsonl`, called in order by `run_pipeline`
(script: `scripts/05_prepare_org_supplement.py`):

| Function | What it does |
|---|---|
| `load_supplement` | reads the JSON Lines file |
| `entity_problems` / `validate_supplement` | checks keys, offsets, labels, overlaps; counts duplicates and overlap with the Nemotron splits |
| `clean` | drops duplicates, documents already in the Nemotron data, over-long texts |
| `complete_labels` | adds `PERSON` and `LOCATION` labels for the names listed in the config (the source file labelled only `ORG`); a name inside a company name is left alone |
| `split_by_template` | puts whole templates into train / validation / test with a fixed seed, so test sentences were never seen in training |
| `family_overlap` | reports how many company-name families of validation/test also occur in training |
| `export` | writes `data/processed/org_supplement_<split>.jsonl` |

Why each step is needed is explained in `docs/DATASET_UPDATE.md`.

## 12. `src/evaluation.py` - metrics

- `evaluate_entities`: strict entity-level precision / recall / F1 per category plus the
  micro-average ("overall"). A span is `(category, start, end)`; it is correct only if it is in the gold set.
- `character_recall`: share of gold sensitive characters covered by any prediction.
- `collect_errors`: builds the error-analysis rows (missed, false alarm, wrong boundary, wrong category).
- `load_synthetic`: reads the synthetic documents of one split with their gold labels (used by
  scripts 04 and 06).

`scripts/06_compare_models.py` compares model folders in two separate stages: `--stage validation`
(development data only, used to choose a model) and `--stage test` (held-out data, run once after
the choice, only to report).

## 13. `src/synthetic_data.py` - fictional evaluation documents

Templates contain slots like `{PERSON_1}`, `{PHONE_2}`, `{DATE_1}`. `fill_template` replaces each
slot with a random fictional value and, because it knows where it wrote each value, records the
gold label at the same time. The same slot number gives the same value (a name that repeats).
`DATE`, `AMOUNT` and `ORDER` slots are distractors that must *not* be detected.

---

## Tests (`tests/`)

| File | What it proves |
|---|---|
| `test_file_processor.py` | cleaning rules, index map, TXT/PDF loading, every error type (empty, unsupported, corrupt, too large, scanned, password) |
| `test_regex_detector.py` | positive and negative cases for every rule, offsets, priority, Arabic-Indic digits |
| `test_ner_detector.py` | BIO decoding without a model; with the model: names, places, long texts, clean text, `scan(model=...)` gives the same result |
| `test_org_supplement.py` | label completion (also a city inside a company name), validation errors, template split, the prepared files |
| `test_risk_redaction.py` | scoring cases from the plan (empty, one item, repeats, caps, bands), merge, propagation, text redaction |
| `test_pdf_redaction.py` | text really removed, neighbours kept, text does not move, metadata cleared, verification works |
| `test_app_workflow.py` | end-to-end: load -> scan -> redact -> summary, synthetic generator, metrics |

Run all of them with `pytest`.
