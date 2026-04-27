"""Load project configuration from YAML files.

Usage:
    from src.utils.config import load_settings, load_assets
    settings = load_settings()
    tickers  = load_assets()
"""

from typing import Any

import yaml

from .paths import ASSETS_FILE, SETTINGS_FILE


def load_settings() -> dict[str, Any]:
    """Load global settings from config/settings.yaml.

    Returns:
        Dictionary with all project settings.

    Raises:
        FileNotFoundError: If settings.yaml does not exist.
    """
    if not SETTINGS_FILE.exists():
        raise FileNotFoundError(f"Settings file not found: {SETTINGS_FILE}")

    with SETTINGS_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_assets() -> list[str]:
    """Load the list of tickers from config/assets.yaml.

    Returns:
        List of ticker strings (upper-cased).

    Raises:
        FileNotFoundError: If assets.yaml does not exist.
        ValueError: If the tickers list is missing or empty.
    """
    if not ASSETS_FILE.exists():
        raise FileNotFoundError(f"Assets file not found: {ASSETS_FILE}")

    with ASSETS_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tickers = data.get("tickers", [])
    if not tickers:
        raise ValueError("No tickers found in config/assets.yaml")

    return [str(t).upper() for t in tickers]
