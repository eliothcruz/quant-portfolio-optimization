"""Download historical price data from Yahoo Finance via yfinance.

All downloads use the same source and the same price field to ensure
cross-asset comparability. Each ticker is downloaded individually
so that one failure does not block the rest.
"""

import pandas as pd
import yfinance as yf

from ..utils.logger import get_logger
from ..utils.paths import DATA_RAW

logger = get_logger(__name__)


def download_asset_data(
    ticker: str,
    start_date: str,
    end_date: str,
    price_field: str = "Adj Close",
    save_raw: bool = True,
) -> pd.DataFrame:
    """Download historical price data for a single asset.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. 'AAPL').
        start_date: Start date in 'YYYY-MM-DD' format (inclusive).
        end_date: End date in 'YYYY-MM-DD' format (exclusive per yfinance).
        price_field: Column to extract. Default 'Adj Close' (split- and
            dividend-adjusted). Must be consistent across all assets.
        save_raw: If True, save the raw series to data/raw/<ticker>.csv
            before any transformation.

    Returns:
        DataFrame with DatetimeIndex named 'date' and one column
        named after the ticker.

    Raises:
        ValueError: If yfinance returns no data or the requested
            price_field is not available for this ticker.
    """
    logger.info(f"Downloading {ticker}  [{start_date} -> {end_date}]  field='{price_field}'")

    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker: '{ticker}'")

    # yfinance >= 0.2.x may return MultiIndex columns (price_field, ticker).
    # Flatten to a simple Index of price field names.
    if isinstance(raw.columns, pd.MultiIndex):
        if price_field in raw.columns.get_level_values(0):
            raw.columns = raw.columns.get_level_values(0)
        elif price_field in raw.columns.get_level_values(1):
            raw.columns = raw.columns.get_level_values(1)
        else:
            raw.columns = raw.columns.get_level_values(0)

    if price_field not in raw.columns:
        raise ValueError(
            f"Field '{price_field}' not found for '{ticker}'. "
            f"Available columns: {list(raw.columns)}"
        )

    df = raw[[price_field]].copy()
    df.columns = [ticker]
    df.index.name = "date"

    if save_raw:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        out_path = DATA_RAW / f"{ticker}.csv"
        df.to_csv(out_path)
        logger.info(f"  Saved raw -> data/raw/{ticker}.csv  ({len(df)} rows)")

    return df


def download_multiple_assets(
    tickers: list[str],
    start_date: str,
    end_date: str,
    price_field: str = "Adj Close",
    save_raw: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download historical price data for multiple assets.

    Each ticker is downloaded individually to produce one raw file per asset.
    Failed tickers are skipped and reported; they do not abort the run.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        price_field: Price column to extract (must be the same for all assets).
        save_raw: If True, save each raw series to data/raw/.

    Returns:
        Dictionary mapping ticker -> DataFrame for successfully downloaded tickers.
    """
    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        try:
            df = download_asset_data(ticker, start_date, end_date, price_field, save_raw)
            results[ticker] = df
        except Exception as exc:
            logger.warning(f"Skipping '{ticker}': {exc}")
            failed.append(ticker)

    status = f"{len(results)}/{len(tickers)} assets succeeded"
    if failed:
        logger.warning(f"Download complete: {status}  |  failed: {failed}")
    else:
        logger.info(f"Download complete: {status}")

    return results
