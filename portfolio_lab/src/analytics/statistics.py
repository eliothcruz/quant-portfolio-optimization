"""Descriptive statistics for individual asset return series.

All annualized metrics assume 252 trading days per year, which is the
standard convention for daily equity returns.
"""

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PERIODS: int = 252


def compute_mean_returns(
    returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = _DEFAULT_PERIODS,
) -> pd.Series:
    """Compute the mean return for each asset.

    Args:
        returns: DataFrame of simple returns with DatetimeIndex.
        annualize: If True, scale daily mean by periods_per_year.
        periods_per_year: Trading days per year (default 252).

    Returns:
        Series indexed by ticker with mean returns.

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("compute_mean_returns: returns DataFrame is empty")

    factor = periods_per_year if annualize else 1
    means = returns.mean() * factor
    label = "annualized" if annualize else "period"
    logger.info(f"Mean returns computed ({label}): {means.round(4).to_dict()}")
    return means


def compute_variances(
    returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = _DEFAULT_PERIODS,
) -> pd.Series:
    """Compute the variance of returns for each asset.

    Variance scales linearly with time (unlike volatility which scales
    with the square root), so annualized variance = daily variance * 252.

    Args:
        returns: DataFrame of simple returns.
        annualize: If True, scale by periods_per_year.
        periods_per_year: Trading days per year (default 252).

    Returns:
        Series indexed by ticker with variances.

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("compute_variances: returns DataFrame is empty")

    factor = periods_per_year if annualize else 1
    variances = returns.var() * factor
    return variances


def compute_volatilities(
    returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = _DEFAULT_PERIODS,
) -> pd.Series:
    """Compute annualized volatility (standard deviation) for each asset.

    Volatility scales with the square root of time, so:
    annualized_vol = daily_std * sqrt(periods_per_year).

    Args:
        returns: DataFrame of simple returns.
        annualize: If True, multiply daily std by sqrt(periods_per_year).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Series indexed by ticker with volatilities.

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("compute_volatilities: returns DataFrame is empty")

    factor = np.sqrt(periods_per_year) if annualize else 1
    vols = returns.std() * factor
    label = "annualized" if annualize else "period"
    logger.info(f"Volatilities computed ({label}): {vols.round(4).to_dict()}")
    return vols


def summarize_asset_statistics(
    returns: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = _DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Produce a descriptive statistics table for all assets.

    Returns a DataFrame where each row is an asset and columns include
    annualized mean return, volatility, variance, plus daily extremes
    and distributional shape statistics.

    Args:
        returns: DataFrame of simple returns with DatetimeIndex.
        annualize: If True, annualize mean, volatility, and variance.
        periods_per_year: Trading days per year (default 252).

    Returns:
        DataFrame with shape (n_assets, n_statistics).

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("summarize_asset_statistics: returns DataFrame is empty")

    mean_ret = compute_mean_returns(returns, annualize=annualize, periods_per_year=periods_per_year)
    vols = compute_volatilities(returns, annualize=annualize, periods_per_year=periods_per_year)
    variances = compute_variances(returns, annualize=annualize, periods_per_year=periods_per_year)

    label = "ann_" if annualize else ""
    summary = pd.DataFrame(
        {
            f"{label}mean_return": mean_ret,
            f"{label}volatility": vols,
            f"{label}variance": variances,
            "min_daily_return": returns.min(),
            "max_daily_return": returns.max(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurt(),
            "n_observations": returns.count(),
        }
    )

    logger.info(f"Asset statistics summary computed for {len(summary)} assets")
    return summary
