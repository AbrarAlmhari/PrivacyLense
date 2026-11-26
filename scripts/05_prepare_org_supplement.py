"""Step 5: prepare the team's extra ORG documents for training and testing.

Usage (from the project folder, after step 1):
    python scripts/05_prepare_org_supplement.py

Reads data/extra/org_supplement_raw.jsonl and writes
data/processed/org_supplement_train.jsonl, _validation.jsonl and _test.jsonl.
Then train again with scripts/02_train_model.py (the training file is listed
under model.extra_train_files in config/config.yaml).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging
from src.data_preparation import DataError
from src.org_supplement import run_pipeline


def main() -> int:
    config = load_config()
    setup_logging(config)
    try:
        summary = run_pipeline(config)
    except DataError as error:
        print(f"ERROR: {error}")
        return 1

    print("\n=== ORG supplement prepared ===")
    print("Validation:", json.dumps(summary["validation"], indent=2))
    print("Cleaning:  ", json.dumps(summary["cleaning"], indent=2))
    print("Labels before:", summary["labels_before_completion"], " after:", summary["labels_after_completion"])
    print("Splits:    ", json.dumps(summary["splits"], indent=2))
    print("Company-name families:", summary["company_name_families"])
    print("Full summary in reports/metrics/org_supplement_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
