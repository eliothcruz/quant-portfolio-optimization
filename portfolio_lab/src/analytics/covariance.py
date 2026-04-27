"""Covariance and correlation matrices from return series.

The covariance matrix is the core input to portfolio optimization.
Annualization follows the standard scaling: Sigma_annual = Sigma_daily * 252.
"""

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PERIODS: int = 252


def compute_covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = _DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Compute the sample covariance matrix of returns.

    The result is used directly in portfolio variance calculations:
    sigma_p^2 = w' * Sigma * w

    Args:
        returns: DataFrame of simple returns with DatetimeIndex.
            All columns must be numeric (one per asset).
        annualize: If True, multiply the daily covariance matrix by
            periods_per_year. Default True.
        periods_per_year: Trading days per year (default 252).

    Returns:
        DataFrame of shape (n_assets, n_assets) with tickers as both
        index and columns.

    Raises:
        ValueError: If returns is empty or has fewer than 2 observations.
    """
    if returns.empty:
        raise ValueError("compute_covariance_matrix: returns DataFrame is empty")
    if len(returns) < 2:
        raise ValueError(
            "compute_covariance_matrix: need at least 2 observations to estimate covariance"
        )

    factor = periods_per_year if annualize else 1
    cov = returns.cov() * factor

    label = "annualized" if annualize else "period"
    logger.info(
        f"Covariance matrix computed ({label}): "
        f"{cov.shape[0]}x{cov.shape[1]}  assets={list(cov.columns)}"
    )
    return cov


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix of returns.

    Correlation is scale-invariant and does not require annualization.

    Args:
        returns: DataFrame of simple returns with DatetimeIndex.

    Returns:
        DataFrame of shape (n_assets, n_assets) with values in [-1, 1].

    Raises:
        ValueError: If returns is empty or has fewer than 2 observations.
    """
    if returns.empty:
        raise ValueError("compute_correlation_matrix: returns DataFrame is empty")
    if len(returns) < 2:
        raise ValueError(
            "compute_correlation_matrix: need at least 2 observations to estimate correlation"
        )

    corr = returns.corr()
    logger.info(
        f"Correlation matrix computed: "
        f"{corr.shape[0]}x{corr.shape[1]}  assets={list(corr.columns)}"
    )
    return corr
