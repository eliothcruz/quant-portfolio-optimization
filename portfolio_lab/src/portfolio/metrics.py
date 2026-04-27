"""Portfolio-level return and risk metrics using matrix algebra.

All functions operate on numpy arrays for efficiency and clarity.
The standard portfolio formulas are:
  - Expected return : μ_p = w' μ
  - Variance        : σ²_p = w' Σ w
  - Volatility      : σ_p  = sqrt(w' Σ w)

Inputs (weights, mean_returns, cov_matrix) must be aligned in length
and consistent in their annualization — use the same periods_per_year
across all analytics functions.

portfolio_returns() works differently: it produces a period-by-period
time series of realized portfolio returns from historical return data.
"""

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def portfolio_return(
    weights: np.ndarray,
    mean_returns: np.ndarray,
) -> float:
    """Compute the expected return of the portfolio: w' μ.

    Args:
        weights: (n,) array of portfolio weights. Must sum to 1.
        mean_returns: (n,) array of per-asset expected returns.
            Use annualized values if you want an annualized result.

    Returns:
        Scalar portfolio expected return.

    Raises:
        ValueError: If weights and mean_returns have different lengths.
    """
    weights = np.asarray(weights, dtype=float)
    mean_returns = np.asarray(mean_returns, dtype=float)

    if weights.shape != mean_returns.shape:
        raise ValueError(
            f"weights shape {weights.shape} != mean_returns shape {mean_returns.shape}"
        )

    return float(weights @ mean_returns)


def portfolio_variance(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Compute portfolio variance: w' Σ w.

    Args:
        weights: (n,) array of portfolio weights.
        cov_matrix: (n, n) covariance matrix.
            Use annualized values for an annualized result.

    Returns:
        Scalar portfolio variance (non-negative).

    Raises:
        ValueError: If dimensions are inconsistent.
    """
    weights = np.asarray(weights, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    n = len(weights)
    if cov_matrix.shape != (n, n):
        raise ValueError(
            f"cov_matrix shape {cov_matrix.shape} is incompatible with {n} weights"
        )

    var = float(weights @ cov_matrix @ weights)

    # Numerical precision can produce tiny negatives; clip to zero
    return max(var, 0.0)


def portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Compute portfolio volatility: sqrt(w' Σ w).

    Args:
        weights: (n,) array of portfolio weights.
        cov_matrix: (n, n) covariance matrix.

    Returns:
        Scalar portfolio volatility (non-negative).
    """
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))


def portfolio_returns(
    returns_df: pd.DataFrame,
    weights: np.ndarray | pd.Series,
) -> pd.Series:
    """Compute the period-by-period realized portfolio return time series.

    Each period's portfolio return is the weighted sum of individual asset
    returns: r_p_t = sum(w_i * r_i_t) = returns_df @ weights.

    When weights is a pd.Series, alignment is performed by column name to
    prevent silent order errors. This is the expected path when loading
    weights from portfolio_weights.csv.

    Args:
        returns_df: DataFrame of asset returns with DatetimeIndex and one
            column per asset.
        weights: (n,) array or pd.Series of portfolio weights.
            If Series, its index must match returns_df column names.

    Returns:
        pd.Series of portfolio returns with the same DatetimeIndex as
        returns_df, named 'portfolio_return'.

    Raises:
        ValueError: If returns_df is empty, weights are empty,
            or dimensions are inconsistent.
    """
    if returns_df.empty:
        raise ValueError("portfolio_returns: returns_df is empty")

    if isinstance(weights, pd.Series):
        # Align by ticker name — order of weights in the file may differ
        w_aligned = weights.reindex(returns_df.columns)
        if w_aligned.isnull().any():
            missing = w_aligned[w_aligned.isnull()].index.tolist()
            raise ValueError(
                f"portfolio_returns: weights missing for tickers: {missing}. "
                f"Ensure weights cover all assets in returns_df."
            )
        w_arr = w_aligned.to_numpy(dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float)
        if len(w_arr) == 0:
            raise ValueError("portfolio_returns: weights array is empty")
        if len(w_arr) != len(returns_df.columns):
            raise ValueError(
                f"portfolio_returns: weights length ({len(w_arr)}) != "
                f"number of assets ({len(returns_df.columns)})"
            )

    port_ret: pd.Series = returns_df @ w_arr
    port_ret.name = "portfolio_return"

    logger.info(
        f"Portfolio returns computed: {len(port_ret)} periods | "
        f"mean={port_ret.mean():.6f}  std={port_ret.std():.6f}"
    )
    return port_ret
