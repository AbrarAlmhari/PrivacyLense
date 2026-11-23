"""Step 3: generate the fictional evaluation documents and sample files.

Usage (from the project folder):
    python scripts/03_make_synthetic_docs.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging
from src.synthetic_data import build_evaluation_set


def main() -> int:
    config = load_config()
    setup_logging(config)
    summary = build_evaluation_set(config)
    print("\n=== Synthetic evaluation set created ===")
    print(json.dumps(summary, indent=2))
    print("Documents: data/synthetic/   Gold labels: data/gold/   Sample files: data/samples/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
