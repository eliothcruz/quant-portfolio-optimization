"""Load and align Fama-French factor data from local CSV files.

Factors are NOT downloaded automatically. Place the factor CSV in
data/factors/ before running the analysis. The Fama-French Data Library
(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
distributes factors in percentage form (e.g., 0.5 = 0.5%). Asset returns in
this pipeline are in decimal form (e.g., 0.005 = 0.5%). Set
convert_percent_to_decimal=True when using Kenneth French raw files.

Expected CSV format:
    date,MKT_RF,SMB,HML,RF
    2019-01-02,0.31,-0.10,0.05,0.01
    ...

date: YYYY-MM-DD
MKT_RF: market excess return (market return minus risk-free rate)
SMB:    Small Minus Big (size factor)
HML:    High Minus Low (value factor)
RF:     daily risk-free rate
"""

import pandas as pd
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_COLUMNS: list[str] = ["MKT_RF", "SMB", "HML", "RF"]
_MIN_OBSERVATIONS: int = 252


def load_factor_data(
    path: str | Path,
    convert_percent_to_decimal: bool = False,
) -> pd.DataFrame:
    """Load a Fama-French factor CSV from disk.

    Args:
        path: Path to the factor CSV file.
            Expected columns: date, MKT_RF, SMB, HML, RF.
        convert_percent_to_decimal: If True, divide all factor columns by 100.
            Use this when loading raw Kenneth French files, which express
            factors as percentages (0.5 means 0.5%, not 50%).

    Returns:
        DataFrame with DatetimeIndex named 'date' and columns
        [MKT_RF, SMB, HML, RF], sorted chronologically, no duplicates.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Factor file not found: {path}\n"
            "Download Fama-French factors from Kenneth French Data Library "
            "and place the CSV in data/factors/. "
            "See docs/theory.md for the expected format."
        )

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")

    if df.empty:
        raise ValueError(f"load_factor_data: factor file is empty: {path}")

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"load_factor_data: missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[_REQUIRED_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    n_dupes = int(df.index.duplicated().sum())
    if n_dupes > 0:
        logger.warning(
            f"load_factor_data: removing {n_dupes} duplicate date(s)"
        )
        df = df[~df.index.duplicated(keep="first")]

    if convert_percent_to_decimal:
        df = df / 100.0
        logger.info("Factor values converted from percent to decimal (÷ 100)")

    logger.info(
        f"Factor data loaded: {len(df)} observations  "
        f"({df.index.min().date()} -> {df.index.max().date()})"
    )
    return df


def align_returns_with_factors(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join asset returns and factor data on their shared date index.

    Only dates present in both DataFrames are kept. No imputation is applied.
    This mirrors the inner-join convention used for asset alignment in the
    main data pipeline.

    Args:
        returns_df: Asset return series (DatetimeIndex, columns = tickers).
        factors_df: Factor data (DatetimeIndex, columns include MKT_RF etc.).

    Returns:
        Tuple (aligned_returns, aligned_factors), both with the same
        DatetimeIndex covering the common period.

    Raises:
        ValueError: If either input is empty, the join produces no rows,
            fewer than _MIN_OBSERVATIONS rows, or if NaN values remain
            after alignment.
    """
    if returns_df.empty:
        raise ValueError("align_returns_with_factors: returns_df is empty")
    if factors_df.empty:
        raise ValueError("align_returns_with_factors: factors_df is empty")

    common_dates = returns_df.index.intersection(factors_df.index)

    if len(common_dates) == 0:
        raise ValueError(
            "align_returns_with_factors: no common dates between returns "
            f"({returns_df.index.min().date()} – {returns_df.index.max().date()}) "
            f"and factors ({factors_df.index.min().date()} – {factors_df.index.max().date()}). "
            "Check that date formats match and the periods overlap."
        )

    aligned_ret = returns_df.loc[common_dates].sort_index()
    aligned_fac = factors_df.loc[common_dates].sort_index()

    if len(aligned_ret) < _MIN_OBSERVATIONS:
        logger.warning(
            f"align_returns_with_factors: only {len(aligned_ret)} overlapping "
            f"observations (minimum recommended: {_MIN_OBSERVATIONS}). "
            "Factor regression estimates may be unreliable."
        )

    nan_ret = int(aligned_ret.isnull().sum().sum())
    nan_fac = int(aligned_fac.isnull().sum().sum())
    if nan_ret > 0 or nan_fac > 0:
        raise ValueError(
            f"align_returns_with_factors: NaN values remain after alignment — "
            f"returns: {nan_ret}  factors: {nan_fac}. "
            "Check the source data for gaps."
        )

    logger.info(
        f"Returns and factors aligned: {len(aligned_ret)} observations  "
        f"({aligned_ret.index.min().date()} -> {aligned_ret.index.max().date()})  "
        f"assets={list(aligned_ret.columns)}"
    )
    return aligned_ret, aligned_fac
