"""Value at Risk (VaR) — historical and parametric methods.

Convention used throughout this module:
  - var_return : the VaR expressed as a return value (typically negative).
                 Example: -0.035 means the 5th-percentile daily return is -3.5%.
  - var_loss   : the VaR expressed as a loss magnitude (positive when there is loss).
                 Formula: var_loss = -var_return.

Use historical_var() or parametric_var() to obtain var_return, then var_loss()
to convert for reporting.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ── Internal validators ────────────────────────────────────────────────────────

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
        n_nan = int(returns.isnull().sum().sum())
    else:
        if returns.empty:
            raise ValueError("returns Series is empty")
        n_nan = int(returns.isnull().sum())

    if n_nan > 0:
        logger.warning(
            f"returns contains {n_nan} NaN value(s) — "
            "they are excluded from VaR computation via quantile/mean"
        )


# ── Public functions ───────────────────────────────────────────────────────────

def historical_var(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
) -> float | pd.Series:
    """Compute historical (empirical) VaR as the left-tail quantile.

    Uses alpha = 1 - confidence_level as the empirical quantile level.
    Example: confidence_level=0.95 -> alpha=0.05 -> 5th percentile of returns.

    The result is a return value, typically negative for loss-generating tails.
    To express as a positive loss magnitude, call var_loss(result).

    Args:
        returns: pd.Series (single asset or portfolio) or pd.DataFrame
            (one column per asset). NaN values are excluded by pandas quantile.
        confidence_level: Confidence level in (0, 1). E.g. 0.95 for 95% VaR.

    Returns:
        float if returns is a Series.
        pd.Series indexed by asset name if returns is a DataFrame.

    Raises:
        ValueError: If confidence_level is out of range or returns is empty.
    """
    _validate_confidence_level(confidence_level)
    _validate_returns(returns)

    alpha = 1.0 - confidence_level

    if isinstance(returns, pd.DataFrame):
        result = returns.quantile(alpha)
        result.name = f"historical_var_{confidence_level:.2f}"
        logger.info(
            f"Historical VaR ({confidence_level:.0%}): "
            + "  ".join(f"{k}={v:.4f}" for k, v in result.items())
        )
        return result

    var = float(returns.quantile(alpha))
    logger.info(f"Historical VaR ({confidence_level:.0%}): {var:.4f}")
    return var


def parametric_var(
    mean: float | pd.Series,
    std: float | pd.Series,
    confidence_level: float = 0.95,
) -> float | pd.Series:
    """Compute parametric (Gaussian) VaR.

    Assumes the return distribution is normal with the given mean and std.
    The result is a return value (typically negative).

    Formula: var_return = mean + z_alpha * std
    where z_alpha = norm.ppf(1 - confidence_level) < 0 for standard CL values.

    Example: mean=0.0003, std=0.012, CL=0.95
      -> z = norm.ppf(0.05) = -1.6449
      -> var_return = 0.0003 + (-1.6449 * 0.012) = -0.0194

    Args:
        mean: Per-period expected return (scalar or pd.Series per asset).
        std: Per-period standard deviation (scalar or pd.Series per asset).
            Must be strictly positive.
        confidence_level: Confidence level in (0, 1).

    Returns:
        float if mean and std are scalars.
        pd.Series if mean and std are Series (aligned by index).

    Raises:
        ValueError: If confidence_level is out of range or std <= 0.
    """
    _validate_confidence_level(confidence_level)

    if isinstance(std, pd.Series):
        if (std <= 0).any():
            bad = std[std <= 0].index.tolist()
            raise ValueError(
                f"std must be > 0 for parametric VaR. "
                f"Non-positive values found for: {bad}"
            )
    else:
        if float(std) <= 0:
            raise ValueError(
                f"std must be > 0 for parametric VaR, got {std}"
            )

    alpha = 1.0 - confidence_level
    z = float(norm.ppf(alpha))  # e.g. z(0.05) = -1.6449

    var = mean + z * std

    if isinstance(var, pd.Series):
        var.name = f"parametric_var_{confidence_level:.2f}"
        logger.info(
            f"Parametric VaR ({confidence_level:.0%}, z={z:.4f}): "
            + "  ".join(f"{k}={v:.4f}" for k, v in var.items())
        )
    else:
        logger.info(
            f"Parametric VaR ({confidence_level:.0%}, z={z:.4f}): {float(var):.4f}"
        )

    return var


def var_loss(var_return: float | pd.Series) -> float | pd.Series:
    """Convert VaR from a return value to a loss magnitude (sign flip).

    Convention:
      - var_return is typically negative (e.g. -0.035 = 3.5% loss).
      - var_loss = -var_return = 0.035 (positive loss magnitude).

    Edge case: if var_return >= 0, the distribution's tail yields a gain
    rather than a loss at the given confidence level. In this case,
    var_loss will be <= 0. This is reported as-is — do not use abs().

    Args:
        var_return: VaR expressed as a return (output of historical_var
            or parametric_var).

    Returns:
        Loss magnitude — positive when var_return < 0.
    """
    return -var_return
