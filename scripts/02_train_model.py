"""Step 2: fine-tune the NER model on the prepared data.

Usage (from the project folder):
    python scripts/02_train_model.py

A GPU is strongly recommended (about 15-20 minutes on an RTX 3050).
On a CPU it works but takes several hours; reduce ``data.train_size`` in
config/config.yaml and re-run step 1 if you only have a CPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging
from src.data_preparation import DataError
from src.train_ner import train


def main() -> int:
    config = load_config()
    setup_logging(config)
    try:
        info = train(config)
    except DataError as error:
        print(f"ERROR: {error}")
        return 1

    print("\n=== Training finished ===")
    for record in info["history"]:
        print(record)
    print(f"Best validation F1: {info['best_validation_f1']:.4f}")
    print(f"Model saved in: {config['model']['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
