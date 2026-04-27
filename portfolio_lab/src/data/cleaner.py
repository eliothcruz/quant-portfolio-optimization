"""Clean and align raw price series.

Cleaning rules (Phase 1):
- Normalize the date index to DatetimeIndex.
- Sort chronologically.
- Remove duplicate dates (keep first occurrence).
- Drop rows with NaN prices.

No imputation is performed. Missing prices are dropped, not filled.
Alignment across assets uses an inner join so only dates common to ALL
assets are kept — ensuring a strictly comparable analysis window.
"""

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def clean_asset_series(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standardize a raw single-asset price series.

    Args:
        df: Raw DataFrame with a date-like index and one price column.
        ticker: Ticker symbol — used only in log messages.

    Returns:
        Cleaned DataFrame with a sorted, deduplicated DatetimeIndex
        and no NaN values.
    """
    df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Sort chronologically
    df = df.sort_index()

    # Remove duplicate dates
    n_dupes = int(df.index.duplicated().sum())
    if n_dupes > 0:
        logger.warning(f"{ticker}: removing {n_dupes} duplicate date(s) — keeping first")
        df = df[~df.index.duplicated(keep="first")]

    # Drop NaN prices (no imputation)
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"{ticker}: dropped {n_dropped} NaN price row(s)")

    return df


def clean_multiple_assets(
    raw_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply clean_asset_series to every asset in a dictionary.

    Args:
        raw_dict: Dictionary mapping ticker -> raw DataFrame.

    Returns:
        Dictionary mapping ticker -> cleaned DataFrame.
    """
    return {
        ticker: clean_asset_series(df, ticker)
        for ticker, df in raw_dict.items()
    }


def align_to_common_period(
    clean_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge all clean price series on their common date index (inner join).

    Only dates present in ALL assets are retained. No imputation is applied.
    This is a deliberate design choice: portfolio analysis must rest on
    a fully comparable, gap-free dataset.

    Args:
        clean_dict: Dictionary mapping ticker -> cleaned price DataFrame.

    Returns:
        DataFrame with DatetimeIndex and one column per asset,
        covering only the dates shared by all assets.

    Raises:
        ValueError: If no assets are provided or the inner join produces
            an empty result (no common dates).
    """
    if not clean_dict:
        raise ValueError("align_to_common_period requires at least one asset")

    # Use iloc[:, 0] instead of squeeze() to be explicit about single-column
    # extraction and avoid ambiguity in pandas >= 2.0.
    series_list = [
        df.iloc[:, 0].rename(ticker)
        for ticker, df in clean_dict.items()
    ]
    aligned = pd.concat(series_list, axis=1, join="inner").sort_index()

    if aligned.empty:
        raise ValueError(
            "No common dates found across assets after inner join. "
            "Check that all tickers share an overlapping trading period."
        )

    logger.info(
        f"Aligned {len(clean_dict)} assets | "
        f"common period: {aligned.index.min().date()} -> {aligned.index.max().date()} "
        f"({len(aligned)} observations)"
    )
    return aligned
