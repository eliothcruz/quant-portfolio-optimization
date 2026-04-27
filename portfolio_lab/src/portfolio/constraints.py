"""Portfolio optimization constraints compatible with scipy.optimize.

Phase 1 scope: long-only (w_i >= 0) and fully invested (sum(w) == 1).
These two constraints together define the feasible region for Phase 2
minimum variance optimization.
"""

from typing import Any

import numpy as np


def long_only_bounds(n_assets: int) -> list[tuple[float, float]]:
    """Return weight bounds for a long-only portfolio.

    Each weight is bounded to [0, 1], preventing short positions and
    preventing any single asset from exceeding full allocation.

    Args:
        n_assets: Number of assets in the portfolio.

    Returns:
        List of (lower, upper) tuples, one per asset.
        Compatible with the `bounds` argument of scipy.optimize.minimize.

    Raises:
        ValueError: If n_assets < 1.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1, got {n_assets}")

    return [(0.0, 1.0)] * n_assets


def full_investment_constraint() -> dict[str, Any]:
    """Return an equality constraint enforcing that weights sum to 1.

    This ensures the portfolio is fully invested (no cash position).

    Returns:
        Dictionary compatible with the `constraints` argument of
        scipy.optimize.minimize (SLSQP method):
        {"type": "eq", "fun": lambda w: sum(w) - 1}
    """
    return {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }
