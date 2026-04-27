"""Download raw price data for all configured assets.

Reads:   config/settings.yaml  (start_date, end_date, price_field)
         config/assets.yaml    (tickers)
Writes:  data/raw/<ticker>.csv for each asset

Usage (from portfolio_lab/ directory):
    python scripts/run_download.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path regardless of where
# the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.downloader import download_multiple_assets
from src.utils.config import load_assets, load_settings
from src.utils.logger import get_logger
from src.utils.paths import DATA_RAW

logger = get_logger("run_download")


def main() -> None:
    settings = load_settings()
    tickers = load_assets()

    logger.info(f"Tickers    : {tickers}")
    logger.info(f"Period     : {settings['start_date']} -> {settings['end_date']}")
    logger.info(f"Price field: {settings['price_field']}")

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    results = download_multiple_assets(
        tickers=tickers,
        start_date=settings["start_date"],
        end_date=settings["end_date"],
        price_field=settings["price_field"],
        save_raw=True,
    )

    if not results:
        logger.error("No data was downloaded. Check your tickers and internet connection.")
        sys.exit(1)

    logger.info(f"Done. Raw files saved to data/raw/  ({len(results)} assets)")


if __name__ == "__main__":
    main()
