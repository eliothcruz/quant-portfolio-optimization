"""Performance analytics for backtested return series (Phases 7-8).

All metrics assume annualization by 252 trading days.
"""

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PERIODS: int = 252


def compute_performance_metrics(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _DEFAULT_PERIODS,
) -> dict:
    """Compute summary performance metrics for a daily return series.

    Args:
        portfolio_returns: pd.Series of daily portfolio returns.
        risk_free_rate: Annualized risk-free rate used in Sharpe ratio
            (default 0.0).
        periods_per_year: Trading days per year for annualization (default 252).

    Returns:
        Dict with keys:
            cumulative_return     - total return over the full period
            annualized_return     - geometric annualized return
            annualized_volatility - annualized standard deviation
            sharpe_ratio          - (ann_return - r_f) / ann_vol
            max_drawdown          - worst peak-to-trough decline (negative)
            n_periods             - number of daily observations used

    Raises:
        ValueError: If portfolio_returns is empty after dropping NaN.
    """
    clean = portfolio_returns.dropna()
    if clean.empty:
        raise ValueError(
            "compute_performance_metrics: portfolio_returns is empty after dropping NaN"
        )

    n = len(clean)

    cum_ret = float((1 + clean).prod() - 1)
    ann_ret = float((1 + cum_ret) ** (periods_per_year / n) - 1)
    ann_vol = float(clean.std() * np.sqrt(periods_per_year))
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0.0 else 0.0

    cum_series = (1 + clean).cumprod()
    rolling_max = cum_series.expanding().max()
    max_dd = float(((cum_series - rolling_max) / rolling_max).min())

    metrics = {
        "cumulative_return": round(cum_ret, 6),
        "annualized_return": round(ann_ret, 6),
        "annualized_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown": round(max_dd, 6),
        "n_periods": n,
    }

    logger.info(
        f"Performance metrics: cum={cum_ret:.4f}  ann_ret={ann_ret:.4f}  "
        f"ann_vol={ann_vol:.4f}  sharpe={sharpe:.4f}  max_dd={max_dd:.4f}"
    )
    return metrics


def compute_strategy_metrics(
    portfolio_returns: pd.Series,
    weights_history: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _DEFAULT_PERIODS,
) -> dict:
    """Extend compute_performance_metrics with average turnover per rebalance.

    Turnover at each rebalance = sum(|w_t - w_{t-1}|). The reported metric
    is the mean across all rebalance periods (excluding the first, which has
    no prior weights to compare against).

    Args:
        portfolio_returns: pd.Series of daily portfolio returns.
        weights_history: DataFrame of weights at each rebalance date
            (output of backtest_portfolio).
        risk_free_rate: Annualized risk-free rate (default 0.0).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Dict with all keys from compute_performance_metrics plus:
            avg_turnover — mean one-way turnover per rebalance period.
    """
    metrics = compute_performance_metrics(portfolio_returns, risk_free_rate, periods_per_year)

    if len(weights_history) > 1:
        avg_turnover = float(
            weights_history.diff().abs().sum(axis=1).iloc[1:].mean()
        )
    else:
        avg_turnover = 0.0

    metrics["avg_turnover"] = round(avg_turnover, 6)
    return metrics
