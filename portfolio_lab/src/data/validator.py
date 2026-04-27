"""Validate raw and processed price series.

Validation checks (Phase 1):
- Missing values per asset.
- Temporal coverage vs. requested window.
- Duplicate dates in the index.
- Post-alignment integrity (no gaps, no missing values).

Validation is advisory: it logs warnings but does not modify data.
The alignment check is the most critical — an empty or gapped aligned
DataFrame will cause downstream failures.
"""

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Tolerance for coverage gaps: weekends + public holidays
_COVERAGE_TOLERANCE_DAYS: int = 5


def check_missing_values(df: pd.DataFrame, ticker: str) -> dict:
    """Count missing values in a price series.

    Args:
        df: Price DataFrame for one asset.
        ticker: Ticker symbol for log messages.

    Returns:
        Dict with keys: ticker, n_missing, pct_missing.
    """
    n_missing = int(df.isnull().sum().sum())
    pct_missing = round(n_missing / max(len(df), 1) * 100, 4)

    if n_missing > 0:
        logger.warning(f"{ticker}: {n_missing} missing value(s) ({pct_missing:.2f}%)")
    else:
        logger.info(f"{ticker}: no missing values")

    return {"ticker": ticker, "n_missing": n_missing, "pct_missing": pct_missing}


def check_temporal_coverage(
    df: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Verify that the series covers the requested date range.

    Gaps up to _COVERAGE_TOLERANCE_DAYS calendar days are expected
    (weekends, market holidays). Larger gaps are flagged as warnings.

    Args:
        df: Price DataFrame with DatetimeIndex.
        ticker: Ticker symbol for log messages.
        start_date: Expected start date ('YYYY-MM-DD').
        end_date: Expected end date ('YYYY-MM-DD').

    Returns:
        Dict with actual vs. requested boundaries and gap sizes in days.
    """
    actual_start = df.index.min()
    actual_end = df.index.max()
    req_start = pd.Timestamp(start_date)
    req_end = pd.Timestamp(end_date)

    gap_start = int((actual_start - req_start).days)
    gap_end = int((req_end - actual_end).days)

    # gap_start < 0 means data begins before the requested window (not an error).
    if gap_start < 0:
        logger.info(
            f"{ticker}: data starts {abs(gap_start)} day(s) before requested start "
            f"(extra history available)"
        )
    elif gap_start > _COVERAGE_TOLERANCE_DAYS:
        logger.warning(
            f"{ticker}: data starts {gap_start} calendar days after requested start"
        )

    if gap_end > _COVERAGE_TOLERANCE_DAYS:
        logger.warning(
            f"{ticker}: data ends {gap_end} calendar days before requested end"
        )

    return {
        "ticker": ticker,
        "requested_start": str(req_start.date()),
        "actual_start": str(actual_start.date()),
        "requested_end": str(req_end.date()),
        "actual_end": str(actual_end.date()),
        "gap_start_days": gap_start,
        "gap_end_days": gap_end,
    }


def check_duplicates(df: pd.DataFrame, ticker: str) -> dict:
    """Check for duplicate date entries in the index.

    Args:
        df: Price DataFrame with DatetimeIndex.
        ticker: Ticker symbol for log messages.

    Returns:
        Dict with keys: ticker, n_duplicates.
    """
    n_dupes = int(df.index.duplicated().sum())
    if n_dupes > 0:
        logger.warning(f"{ticker}: {n_dupes} duplicate date(s) in index")
    return {"ticker": ticker, "n_duplicates": n_dupes}


def validate_temporal_alignment(aligned_df: pd.DataFrame) -> dict:
    """Validate the aligned price DataFrame produced by align_to_common_period.

    This is the critical check before any portfolio computation. An empty
    result or one with missing values indicates a data pipeline problem
    that must be resolved before proceeding.

    Args:
        aligned_df: Output of cleaner.align_to_common_period().

    Returns:
        Dict summarizing n_assets, n_observations, common period,
        and whether any missing values remain.

    Raises:
        ValueError: If the aligned DataFrame is empty.
    """
    if aligned_df.empty:
        raise ValueError(
            "Aligned DataFrame is empty. "
            "No dates are common to all assets — check data coverage."
        )

    any_missing = bool(aligned_df.isnull().any().any())
    result = {
        "n_assets": len(aligned_df.columns),
        "n_observations": len(aligned_df),
        "common_start": str(aligned_df.index.min().date()),
        "common_end": str(aligned_df.index.max().date()),
        "any_missing": any_missing,
    }

    if any_missing:
        logger.warning(
            "Aligned DataFrame still contains missing values — review cleaner output"
        )
    else:
        logger.info(
            f"Alignment validated: {result['n_assets']} assets | "
            f"{result['n_observations']} observations | "
            f"{result['common_start']} -> {result['common_end']}"
        )

    return result


def validate_all_raw(
    raw_dict: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Run all per-asset validations and return a consolidated report.

    Args:
        raw_dict: Dictionary mapping ticker -> raw DataFrame.
        start_date: Expected start date for coverage check.
        end_date: Expected end date for coverage check.

    Returns:
        List of per-asset result dicts, one entry per ticker.
    """
    reports = []
    for ticker, df in raw_dict.items():
        report: dict = {}
        report.update(check_missing_values(df, ticker))
        report.update(check_temporal_coverage(df, ticker, start_date, end_date))
        report.update(check_duplicates(df, ticker))
        reports.append(report)
    return reports
