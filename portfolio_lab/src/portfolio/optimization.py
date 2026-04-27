"""Portfolio optimization: minimum variance (Phase 2).

Uses scipy.optimize.minimize with the SLSQP solver.
Constraints: long-only (w_i >= 0) and fully invested (sum(w) == 1).

The minimum variance portfolio minimizes total portfolio variance:
    min  w' Σ w
    s.t. sum(w) = 1
         w_i >= 0  for all i

# TODO (Phase 3): implement maximum Sharpe ratio optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

from .constraints import full_investment_constraint, long_only_bounds
from ..utils.logger import get_logger

logger = get_logger(__name__)


def min_variance_portfolio(
    mean_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Find the minimum variance portfolio weights.

    The mean_returns argument is included for API consistency with future
    optimization functions (e.g. max Sharpe) but is not used in the
    minimum variance objective — only the covariance matrix is needed.

    Args:
        mean_returns: (n,) expected returns per asset (pd.Series or np.ndarray).
            Used only to infer n_assets.
        cov_matrix: (n, n) annualized covariance matrix (pd.DataFrame or np.ndarray).

    Returns:
        (n,) array of optimal weights, non-negative and summing to 1.
        Small negative values from numerical precision are clipped to 0
        and weights are re-normalized.

    Raises:
        ValueError: If inputs are inconsistent in shape.
        RuntimeError: If the SLSQP solver does not converge.
    """
    # Accept both pandas and numpy inputs
    mu = np.asarray(mean_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)

    n = len(mu)
    if sigma.shape != (n, n):
        raise ValueError(
            f"cov_matrix shape {sigma.shape} is incompatible with "
            f"mean_returns length {n}"
        )
    if n < 2:
        raise ValueError("Portfolio optimization requires at least 2 assets")

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    x0 = np.ones(n) / n  # equal-weight starting point
    bounds = long_only_bounds(n)
    constraints = [full_investment_constraint()]

    result: OptimizeResult = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        raise RuntimeError(
            f"Minimum variance optimization failed to converge: {result.message}"
        )

    # Clip tiny negative weights caused by floating-point precision and re-normalize
    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()

    logger.info(
        f"Minimum variance portfolio solved in {result.nit} iterations | "
        f"portfolio variance = {result.fun:.6f}"
    )
    return weights
