"""Configuration and logging helpers.

All tunable values (file limits, model settings, risk weights, ...) live in
``config/config.yaml`` so that nothing important is hard-coded in the modules.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

# Project root = the folder that contains app.py, src/, config/ ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML configuration file and return it as a dictionary.

    The result is cached, so the file is only read once per run.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Configuration file is not valid YAML: {error}") from error

    if not isinstance(config, dict):
        raise ConfigError("Configuration file must contain a YAML mapping.")
    return config


def resolve_path(relative_path: str | Path) -> Path:
    """Turn a path from the config file into an absolute path."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logging(config: dict | None = None) -> None:
    """Configure logging once for the whole application.

    Privacy rule: modules log *events and counts only*. They never log the
    extracted text or the detected sensitive strings.
    """
    config = config or load_config()
    log_settings = config.get("logging", {})
    level = getattr(logging, str(log_settings.get("level", "INFO")).upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = log_settings.get("file")
    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    # Third-party libraries are very chatty at INFO level; keep our log clean.
    for noisy_logger in ("httpx", "httpx2", "huggingface_hub", "matplotlib", "PIL"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
