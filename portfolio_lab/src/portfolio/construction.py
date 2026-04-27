"""Portfolio weight vector construction and validation."""

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def validate_weights(
    weights: np.ndarray,
    tickers: list[str],
    allow_short: bool = False,
    tolerance: float = 1e-6,
) -> None:
    """Validate that a weight vector satisfies portfolio constraints.

    Checks:
    - Length matches the number of tickers.
    - Weights sum to 1 (fully invested), within numerical tolerance.
    - All weights are non-negative if allow_short is False (long-only).

    Args:
        weights: (n,) array of portfolio weights.
        tickers: List of asset tickers — must match len(weights).
        allow_short: Whether negative weights are permitted.
        tolerance: Absolute tolerance for the sum-to-one check.

    Raises:
        ValueError: If any constraint is violated.
    """
    weights = np.asarray(weights, dtype=float)

    if len(weights) != len(tickers):
        raise ValueError(
            f"weights length ({len(weights)}) != number of tickers ({len(tickers)})"
        )

    weight_sum = weights.sum()
    if abs(weight_sum - 1.0) > tolerance:
        raise ValueError(
            f"Weights sum to {weight_sum:.8f}, expected 1.0 "
            f"(tolerance {tolerance})"
        )

    if not allow_short:
        negative = weights[weights < -tolerance]
        if len(negative) > 0:
            offending = [tickers[i] for i in np.where(weights < -tolerance)[0]]
            raise ValueError(
                f"Negative weights found but allow_short=False: {offending}"
            )


def build_weight_series(
    weights: np.ndarray,
    tickers: list[str],
) -> pd.Series:
    """Return a labeled Series mapping tickers to portfolio weights.

    Args:
        weights: (n,) array of portfolio weights.
        tickers: List of ticker strings, aligned with weights.

    Returns:
        pd.Series indexed by ticker, named 'weight'.

    Raises:
        ValueError: If lengths do not match.
    """
    weights = np.asarray(weights, dtype=float)

    if len(weights) != len(tickers):
        raise ValueError(
            f"weights length ({len(weights)}) != tickers length ({len(tickers)})"
        )

    return pd.Series(weights, index=tickers, name="weight")
