"""Load raw and processed price data from disk.

Raw files live in data/raw/<ticker>.csv.
Processed files live in data/processed/<filename>.csv.
"""

import pandas as pd

from ..utils.logger import get_logger
from ..utils.paths import DATA_PROCESSED, DATA_RAW

logger = get_logger(__name__)


def load_raw_asset(ticker: str) -> pd.DataFrame:
    """Load the raw CSV file for a single ticker from data/raw/.

    Args:
        ticker: Ticker symbol. File is expected at data/raw/<ticker>.csv.

    Returns:
        DataFrame with DatetimeIndex and one price column.

    Raises:
        FileNotFoundError: If the file does not exist (run run_download.py first).
    """
    path = DATA_RAW / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}  —  run run_download.py first."
        )

    df = pd.read_csv(path, index_col="date", parse_dates=True)
    logger.info(
        f"Loaded raw {ticker}: {len(df)} rows  "
        f"({df.index.min().date()} – {df.index.max().date()})"
    )
    return df


def load_all_raw_assets(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load raw CSVs for a list of tickers.

    Tickers with missing files are skipped with a warning so that one
    missing file does not abort the entire pipeline.

    Args:
        tickers: List of ticker symbols to load.

    Returns:
        Dictionary mapping ticker -> DataFrame for successfully loaded assets.
    """
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            results[ticker] = load_raw_asset(ticker)
        except FileNotFoundError as exc:
            logger.warning(str(exc))

    logger.info(f"Loaded {len(results)}/{len(tickers)} raw assets from data/raw/")
    return results


def load_processed(filename: str) -> pd.DataFrame:
    """Load a processed CSV from data/processed/.

    Args:
        filename: File name within data/processed/
            (e.g. 'prices_aligned.csv', 'returns.csv').

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = DATA_PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}  —  run run_prepare_data.py first."
        )

    df = pd.read_csv(path, index_col="date", parse_dates=True)
    logger.info(
        f"Loaded '{filename}': {len(df)} rows, {len(df.columns)} assets"
    )
    return df
