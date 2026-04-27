"""Centralized path definitions for the project.

All modules import paths from here so that moving the project root
requires a change in only one place.
"""

from pathlib import Path

# Project root: two levels above src/utils/paths.py  →  portfolio_lab/
ROOT: Path = Path(__file__).resolve().parents[2]

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_METADATA: Path = DATA_DIR / "metadata"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUTS_DIR: Path = ROOT / "outputs"
OUTPUTS_TABLES: Path = OUTPUTS_DIR / "tables"
OUTPUTS_FIGURES: Path = OUTPUTS_DIR / "figures"
OUTPUTS_REPORTS: Path = OUTPUTS_DIR / "reports"

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG_DIR: Path = ROOT / "config"
SETTINGS_FILE: Path = CONFIG_DIR / "settings.yaml"
ASSETS_FILE: Path = CONFIG_DIR / "assets.yaml"
