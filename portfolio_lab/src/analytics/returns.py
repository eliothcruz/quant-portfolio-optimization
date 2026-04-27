"""Compute return series from aligned price series.

Phase 1 implements simple (arithmetic) returns only.
Log returns are stubbed for Phase 2.
"""

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute period-over-period simple returns: r_t = (P_t / P_{t-1}) - 1.

    The first row (NaN produced by pct_change) is dropped so the
    returned DataFrame contains only complete observations.

    Args:
        prices: DataFrame with DatetimeIndex and one column per asset.
            All assets must share the same date index (use
            cleaner.align_to_common_period first).

    Returns:
        DataFrame of simple returns with the same columns,
        one fewer row than prices.

    Raises:
        ValueError: If prices is empty.
    """
    if prices.empty:
        raise ValueError("Cannot compute returns: prices DataFrame is empty")

    returns = prices.pct_change().dropna()
    logger.info(
        f"Simple returns computed: {len(returns)} observations, "
        f"{len(returns.columns)} assets"
    )
    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: r_t = ln(P_t / P_{t-1}).

    Placeholder — scheduled for Phase 2.
    Log returns are additive over time and more suitable for
    multi-period aggregation than simple returns.
    """
    raise NotImplementedError("Log returns are not implemented in Phase 1")
