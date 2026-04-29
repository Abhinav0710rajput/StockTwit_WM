from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a unified YAML config and return the full nested dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "train", "eval"):
        if section not in cfg:
            raise KeyError(f"Config {path} is missing required section '{section}'")
    return cfg
