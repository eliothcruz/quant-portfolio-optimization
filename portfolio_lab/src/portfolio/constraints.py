"""Portfolio optimization constraints compatible with scipy.optimize.

Phase 2 scope: long-only (w_i >= 0) and fully invested (sum(w) == 1).
Phase 7 adds weight_bounds() for custom per-asset min/max limits.
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


def weight_bounds(
    n_assets: int,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> list[tuple[float, float]]:
    """Return per-asset weight bounds [min_weight, max_weight].

    Generalizes long_only_bounds to support custom concentration limits.

    Args:
        n_assets: Number of assets in the portfolio.
        min_weight: Lower bound for each asset weight (default 0.0).
        max_weight: Upper bound for each asset weight (default 1.0).

    Returns:
        List of (min_weight, max_weight) tuples, one per asset.

    Raises:
        ValueError: If n_assets < 1 or bounds are invalid or infeasible.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1, got {n_assets}")
    if not 0.0 <= min_weight <= max_weight <= 1.0:
        raise ValueError(
            f"Require 0 <= min_weight <= max_weight <= 1, "
            f"got min={min_weight}, max={max_weight}"
        )
    if n_assets * min_weight > 1.0:
        raise ValueError(
            f"min_weight={min_weight} with {n_assets} assets implies "
            f"minimum weight sum {n_assets * min_weight:.4f} > 1 — infeasible"
        )
    if n_assets * max_weight < 1.0:
        raise ValueError(
            f"max_weight={max_weight} with {n_assets} assets implies "
            f"maximum weight sum {n_assets * max_weight:.4f} < 1 — infeasible"
        )

    return [(min_weight, max_weight)] * n_assets


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
