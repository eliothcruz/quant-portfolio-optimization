"""Clean, validate, align price series and compute simple returns.

Reads:   data/raw/<ticker>.csv  (output of run_download.py)
Writes:  data/processed/prices_aligned.csv  — aligned price matrix
         data/processed/returns.csv         — simple daily returns

Usage (from portfolio_lab/ directory):
    python scripts/run_prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.returns import compute_simple_returns
from src.data.cleaner import align_to_common_period, clean_multiple_assets
from src.data.loader import load_all_raw_assets
from src.data.validator import validate_all_raw, validate_temporal_alignment
from src.utils.config import load_assets, load_settings
from src.utils.logger import get_logger
from src.utils.paths import DATA_PROCESSED

logger = get_logger("run_prepare_data")


def main() -> None:
    settings = load_settings()
    tickers = load_assets()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    logger.info("Step 1/5 — Loading raw data")
    raw = load_all_raw_assets(tickers)

    if not raw:
        logger.error("No raw data found. Run run_download.py first.")
        sys.exit(1)

    # ── Step 2: Validate raw data ─────────────────────────────────────────────
    logger.info("Step 2/5 — Validating raw data")
    reports = validate_all_raw(raw, settings["start_date"], settings["end_date"])

    # Log consolidated validation summary
    for r in reports:
        ticker = r["ticker"]
        flags = []
        if r["n_missing"] > 0:
            flags.append(f"{r['n_missing']} missing")
        if r["n_duplicates"] > 0:
            flags.append(f"{r['n_duplicates']} dupes")
        if r["gap_start_days"] > 5:
            flags.append(f"start gap {r['gap_start_days']}d")
        if r["gap_end_days"] > 5:
            flags.append(f"end gap {r['gap_end_days']}d")
        status = "OK" if not flags else "WARN: " + ", ".join(flags)
        logger.info(
            f"  {ticker:8s} {r['actual_start']} -> {r['actual_end']}  [{status}]"
        )

    # ── Step 3: Clean each series ─────────────────────────────────────────────
    logger.info("Step 3/5 — Cleaning series")
    clean = clean_multiple_assets(raw)

    # ── Step 4: Align to common period (inner join, no imputation) ────────────
    logger.info("Step 4/5 — Aligning to common period")
    prices = align_to_common_period(clean)
    validate_temporal_alignment(prices)

    prices_path = DATA_PROCESSED / "prices_aligned.csv"
    prices.to_csv(prices_path)
    logger.info(f"  Saved -> {prices_path.name}  ({prices.shape[0]} rows x {prices.shape[1]} assets)")

    # ── Step 5: Compute simple returns ────────────────────────────────────────
    logger.info("Step 5/5 — Computing simple returns")
    returns = compute_simple_returns(prices)

    returns_path = DATA_PROCESSED / "returns.csv"
    returns.to_csv(returns_path)
    logger.info(f"  Saved -> {returns_path.name}  ({returns.shape[0]} rows x {returns.shape[1]} assets)")

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
