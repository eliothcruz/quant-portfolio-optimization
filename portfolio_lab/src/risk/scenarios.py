"""Stress scenarios for portfolio risk analysis.

Phase 3 scope: simple, transparent shocks applied to return series
or covariance matrices. No Monte Carlo simulation.

Scenarios are designed to be auditable: each function applies a single,
clearly documented transformation. Complex scenario engines are out of scope.
"""

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def apply_return_shock(
    returns: pd.Series | pd.DataFrame,
    shock: float,
) -> pd.Series | pd.DataFrame:
    """Apply a constant additive shock to every return observation.

    Shifts the entire return distribution by a fixed amount. Useful for
    simple stress tests such as:
    - shock = -0.02: simulate an extra 2% daily loss on every period
    - shock = +0.01: simulate a 1% daily tailwind

    This does not change the shape or variance of the distribution,
    only its location.

    Args:
        returns: Return series (pd.Series) or matrix (pd.DataFrame).
        shock: Additive constant applied to every observation.
            Negative values simulate adverse conditions.

    Returns:
        Shocked returns of the same type and shape as the input.

    Raises:
        ValueError: If returns is empty.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("apply_return_shock: returns DataFrame is empty")
    else:
        if returns.empty:
            raise ValueError("apply_return_shock: returns Series is empty")

    shocked = returns + shock
    label = f"+{shock:.4f}" if shock >= 0 else f"{shock:.4f}"
    logger.info(f"Return shock applied ({label}) to {type(returns).__name__}")
    return shocked


def apply_volatility_shock(
    cov_matrix: np.ndarray,
    shock_factor: float,
) -> np.ndarray:
    """Scale the covariance matrix by a constant multiplicative factor.

    A shock_factor > 1 simulates increased volatility and co-movement:
    - shock_factor = 1.5: 50% increase in all variances and covariances
    - shock_factor = 2.0: simulate a volatility-doubling stress event

    Because the covariance matrix scales linearly, the implied volatility
    of each asset increases by sqrt(shock_factor).

    Args:
        cov_matrix: (n x n) symmetric positive-semidefinite covariance matrix.
        shock_factor: Multiplicative scaling factor. Must be > 0.

    Returns:
        Shocked covariance matrix of the same shape.

    Raises:
        ValueError: If shock_factor <= 0 or cov_matrix is not 2-dimensional.
    """
    cov = np.asarray(cov_matrix, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(
            f"cov_matrix must be a square 2D array, got shape {cov.shape}"
        )
    if shock_factor <= 0:
        raise ValueError(
            f"shock_factor must be > 0, got {shock_factor}"
        )

    shocked = cov * shock_factor
    implied_vol_increase = np.sqrt(shock_factor)
    logger.info(
        f"Volatility shock applied: factor={shock_factor:.2f} | "
        f"implied vol increase={implied_vol_increase:.4f}x"
    )
    return shocked
