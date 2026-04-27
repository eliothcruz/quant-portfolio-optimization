"""Tail Value at Risk (TVaR) — also known as CVaR or Expected Shortfall.

TVaR is the average return in the worst (1 - confidence_level) fraction of
outcomes — it tells us what to expect if we are already in the tail:

  TVaR_alpha = E[ r | r <= VaR_alpha ]

Like VaR, TVaR is expressed as a return value (typically negative).
Use tvar_loss() to convert to a positive loss magnitude.

TVaR is always <= var_return and therefore always >= var_loss in absolute
terms, making it a more conservative and coherent risk measure than VaR.
"""

import pandas as pd

from ..utils.logger import get_logger
from .var import historical_var

logger = get_logger(__name__)


# ── Internal validators (local copies to keep modules self-contained) ──────────

def _validate_confidence_level(confidence_level: float) -> None:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, "
            f"got {confidence_level}"
        )


def _validate_returns(returns: pd.Series | pd.DataFrame) -> None:
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
    else:
        if returns.empty:
            raise ValueError("returns Series is empty")


# ── Public functions ───────────────────────────────────────────────────────────

def historical_tvar(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
) -> float | pd.Series:
    """Compute historical TVaR (Expected Shortfall / Conditional VaR).

    Procedure:
    1. Compute historical VaR: q = quantile(1 - confidence_level).
    2. Select the tail: all returns <= q.
    3. Return the mean of the tail.

    The result is a return value, typically negative. To express as a
    positive loss magnitude, call tvar_loss().

    TVaR is always <= var_return (i.e., deeper in the loss territory),
    so tvar_loss >= var_loss at any given confidence level.

    Args:
        returns: pd.Series (single asset or portfolio) or pd.DataFrame
            (one column per asset). NaN values are dropped before computation.
        confidence_level: Confidence level in (0, 1). E.g. 0.95 for 95% TVaR.

    Returns:
        float if returns is a Series.
        pd.Series indexed by asset name if returns is a DataFrame.

    Raises:
        ValueError: If confidence_level is out of range, returns is empty,
            or the tail contains no observations.
    """
    _validate_confidence_level(confidence_level)
    _validate_returns(returns)

    if isinstance(returns, pd.DataFrame):
        # Apply column-by-column; result is a Series indexed by asset name
        result = returns.apply(
            lambda col: historical_tvar(col.dropna(), confidence_level)
        )
        result.name = f"historical_tvar_{confidence_level:.2f}"
        return result

    # Series path
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("returns contains only NaN values after dropping NaN")

    var = historical_var(clean, confidence_level)
    tail = clean[clean <= var]

    if tail.empty:
        raise ValueError(
            f"TVaR tail is empty: no observations at or below VaR threshold "
            f"({var:.6f}). Increase the dataset size or lower confidence_level."
        )

    tvar = float(tail.mean())
    logger.info(
        f"Historical TVaR ({confidence_level:.0%}): {tvar:.4f}  "
        f"(tail observations: {len(tail)})"
    )
    return tvar


def tvar_loss(tvar_return: float | pd.Series) -> float | pd.Series:
    """Convert TVaR from a return value to a positive loss magnitude.

    Convention mirrors var_loss: loss = -tvar_return.
    A negative tvar_return (average tail loss) becomes a positive number.

    TVaR loss >= VaR loss at any given confidence level, reflecting
    that the expected tail loss is always at least as large as the
    threshold loss.

    Args:
        tvar_return: TVaR expressed as a return (output of historical_tvar).

    Returns:
        Loss magnitude — positive when tvar_return < 0.
    """
    return -tvar_return
