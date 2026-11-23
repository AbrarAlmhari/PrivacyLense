"""Step 1: download, validate, clean, explore and export the training data.

Usage (from the project folder):
    python scripts/01_prepare_data.py
"""

import json
import sys
from pathlib import Path

# Allow "from src ..." imports when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging
from src.data_preparation import DataError, run_pipeline


def main() -> int:
    config = load_config()
    setup_logging(config)
    try:
        summary = run_pipeline(config)
    except DataError as error:
        print(f"ERROR: {error}")
        return 1

    print("\n=== Data preparation finished ===")
    print("Validation:", json.dumps(summary["validation"], indent=2))
    print("Cleaning:  ", json.dumps(summary["cleaning"], indent=2))
    print("Splits:    ", json.dumps(summary["splits"], indent=2))
    print("Figures saved in reports/figures, full summary in reports/metrics/data_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
