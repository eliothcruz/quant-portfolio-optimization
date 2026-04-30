"""Portfolio optimization: minimum variance (Phase 2) and
maximum Sharpe ratio + efficient frontier (Phase 6).

Uses scipy.optimize.minimize with the SLSQP solver.
Constraints: long-only (w_i >= 0) and fully invested (sum(w) == 1).
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


def max_sharpe_portfolio(
    mean_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    risk_free_rate: float = 0.0,
) -> dict:
    """Find the maximum Sharpe ratio portfolio.

    Maximizes (mu_p - r_f) / sigma_p by minimizing its negative.

    Args:
        mean_returns: (n,) expected returns per asset (pd.Series or np.ndarray).
        cov_matrix: (n, n) annualized covariance matrix.
        risk_free_rate: Annualized risk-free rate (default 0.0).

    Returns:
        Dict with keys:
            weights    - pd.Series indexed by asset name (or 0..n-1)
            return     - float, annualized portfolio expected return
            volatility - float, annualized portfolio volatility
            sharpe     - float, Sharpe ratio

    Raises:
        ValueError: If inputs are inconsistent in shape.
        RuntimeError: If the SLSQP solver does not converge.
    """
    asset_names = mean_returns.index if isinstance(mean_returns, pd.Series) else None

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

    def neg_sharpe(w: np.ndarray) -> float:
        p_ret = float(w @ mu)
        p_var = float(w @ sigma @ w)
        if p_var <= 0.0:
            return 0.0
        return -(p_ret - risk_free_rate) / np.sqrt(p_var)

    x0 = np.ones(n) / n
    bounds = long_only_bounds(n)
    constraints = [full_investment_constraint()]

    result: OptimizeResult = minimize(
        neg_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        raise RuntimeError(
            f"Max Sharpe optimization failed to converge: {result.message}"
        )

    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()

    p_ret = float(weights @ mu)
    p_var = max(float(weights @ sigma @ weights), 0.0)
    p_vol = float(np.sqrt(p_var))
    sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 0.0 else 0.0

    w_series = (
        pd.Series(weights, index=asset_names, name="weight")
        if asset_names is not None
        else pd.Series(weights, name="weight")
    )

    logger.info(
        f"Max Sharpe portfolio solved in {result.nit} iterations | "
        f"Sharpe={sharpe:.4f}  ret={p_ret:.4f}  vol={p_vol:.4f}"
    )
    return {"weights": w_series, "return": p_ret, "volatility": p_vol, "sharpe": sharpe}


def min_variance_target_return(
    mean_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    target_return: float,
) -> dict | None:
    """Find the minimum variance portfolio that achieves a target return.

    Minimizes portfolio variance subject to sum(w)=1, w>=0,
    and w'mu = target_return.

    Args:
        mean_returns: (n,) expected returns per asset.
        cov_matrix: (n, n) annualized covariance matrix.
        target_return: Required annualized portfolio expected return.

    Returns:
        Dict with keys 'weights' (pd.Series), 'return' (float),
        'volatility' (float), or None if the problem is infeasible.

    Raises:
        ValueError: If inputs are inconsistent in shape.
    """
    asset_names = mean_returns.index if isinstance(mean_returns, pd.Series) else None

    mu = np.asarray(mean_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)

    n = len(mu)
    if sigma.shape != (n, n):
        raise ValueError(
            f"cov_matrix shape {sigma.shape} is incompatible with "
            f"mean_returns length {n}"
        )

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    x0 = np.ones(n) / n
    bounds = long_only_bounds(n)
    constraints = [
        full_investment_constraint(),
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
    ]

    result: OptimizeResult = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        return None

    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()

    p_ret = float(weights @ mu)
    p_var = max(float(weights @ sigma @ weights), 0.0)
    p_vol = float(np.sqrt(p_var))

    w_series = (
        pd.Series(weights, index=asset_names, name="weight")
        if asset_names is not None
        else pd.Series(weights, name="weight")
    )
    return {"weights": w_series, "return": p_ret, "volatility": p_vol}


def risk_parity_portfolio(
    cov_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Find the risk parity (equal risk contribution) portfolio.

    Each asset is allocated such that it contributes an equal fraction
    of total portfolio variance. The risk contribution of asset i is:

        RC_i = w_i * (Σw)_i

    The objective is to minimize the variance of the RC vector, which is
    zero when all contributions are equal.

    Unlike mean-variance optimization, this strategy requires only the
    covariance matrix — no expected return estimates.

    Args:
        cov_matrix: (n, n) annualized covariance matrix (pd.DataFrame or
            np.ndarray). Must be positive semi-definite.

    Returns:
        (n,) array of optimal weights, non-negative and summing to 1.

    Raises:
        ValueError: If cov_matrix is not square or has fewer than 2 assets.
        RuntimeError: If the SLSQP solver does not converge.
    """
    sigma = np.asarray(cov_matrix, dtype=float)

    n = sigma.shape[0]
    if sigma.ndim != 2 or sigma.shape[1] != n:
        raise ValueError(
            f"risk_parity_portfolio: cov_matrix must be square, got shape {sigma.shape}"
        )
    if n < 2:
        raise ValueError("risk_parity_portfolio: requires at least 2 assets")

    def objective(w: np.ndarray) -> float:
        sigma_w = sigma @ w
        rc = w * sigma_w
        rc_mean = rc.mean()
        return float(np.sum((rc - rc_mean) ** 2))

    x0 = np.ones(n) / n
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
            f"Risk parity optimization failed to converge: {result.message}"
        )

    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()

    sigma_w = sigma @ weights
    rc = weights * sigma_w
    logger.info(
        f"Risk parity portfolio solved in {result.nit} iterations | "
        f"RC std={rc.std():.2e}  (lower = more equal)"
    )
    return weights


def efficient_frontier(
    mean_returns: pd.Series | np.ndarray,
    cov_matrix: pd.DataFrame | np.ndarray,
    n_points: int = 50,
) -> pd.DataFrame:
    """Compute the efficient frontier as a sequence of minimum-variance portfolios.

    Sweeps target returns from min(mean_returns) to max(mean_returns) and
    solves min_variance_target_return at each level. Infeasible points are
    silently dropped.

    Args:
        mean_returns: (n,) expected returns per asset.
        cov_matrix: (n, n) annualized covariance matrix.
        n_points: Number of evenly-spaced target returns to evaluate (default 50).

    Returns:
        DataFrame with columns 'return' and 'volatility', sorted by return.

    Raises:
        RuntimeError: If no feasible points were found.
    """
    mu = np.asarray(mean_returns, dtype=float)
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    records = []
    for target in target_returns:
        point = min_variance_target_return(mean_returns, cov_matrix, target)
        if point is not None:
            records.append({"return": point["return"], "volatility": point["volatility"]})

    if not records:
        raise RuntimeError("efficient_frontier: no feasible points found")

    frontier_df = (
        pd.DataFrame(records)
        .sort_values("return")
        .reset_index(drop=True)
    )

    logger.info(
        f"Efficient frontier computed: {len(frontier_df)}/{n_points} feasible points"
    )
    return frontier_df
