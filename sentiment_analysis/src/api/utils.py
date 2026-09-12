"""
Utility Module
--------------
Shared helpers: logging setup, config loading, path resolution.
"""

import logging
import yaml
import json
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Configure and return a named logger."""
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_json_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
