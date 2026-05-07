"""Build risk, summary, and factor exposure tables for assets and the portfolio.

All risk metrics in these tables are computed at the daily horizon,
consistent with the return series frequency. Mean return and volatility
are also reported as daily values to maintain internal consistency.

Column naming convention:
  - *_return : the metric expressed as a return value (negative = loss)
  - *_loss   : the metric expressed as a positive loss magnitude
"""

import pandas as pd

from ..analytics.statistics import compute_mean_returns, compute_volatilities
from ..risk.tvar import historical_tvar, tvar_loss
from ..risk.var import historical_var, parametric_var, var_loss
from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_asset_risk_table(
    returns_df: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Build a per-asset risk metrics table.

    All metrics are computed at the daily horizon (no annualization).
    Each row corresponds to one asset; columns are the risk statistics.

    Columns:
      mean_return            — daily arithmetic mean return
      volatility             — daily standard deviation
      historical_var_return  — empirical left-tail quantile (negative)
      historical_var_loss    — |historical_var_return| (positive)
      parametric_var_return  — Gaussian VaR = mean + z*std (negative)
      parametric_var_loss    — |parametric_var_return| (positive)
      historical_tvar_return — average return in the VaR tail (negative)
      historical_tvar_loss   — |historical_tvar_return| (positive)

    Args:
        returns_df: DataFrame of asset returns with DatetimeIndex.
        confidence_level: VaR/TVaR confidence level in (0, 1).

    Returns:
        DataFrame of shape (n_assets, 8), indexed by ticker.

    Raises:
        ValueError: If returns_df is empty or confidence_level is invalid.
    """
    if returns_df.empty:
        raise ValueError("build_asset_risk_table: returns_df is empty")

    # Daily statistics (no annualization factor)
    daily_mean = returns_df.mean()
    daily_std = returns_df.std()

    hist_var = historical_var(returns_df, confidence_level)
    param_var = parametric_var(daily_mean, daily_std, confidence_level)
    hist_tvar = historical_tvar(returns_df, confidence_level)

    table = pd.DataFrame(
        {
            "mean_return": daily_mean,
            "volatility": daily_std,
            "historical_var_return": hist_var,
            "historical_var_loss": var_loss(hist_var),
            "parametric_var_return": param_var,
            "parametric_var_loss": var_loss(param_var),
            "historical_tvar_return": hist_tvar,
            "historical_tvar_loss": tvar_loss(hist_tvar),
        }
    )

    logger.info(
        f"Asset risk table built: {len(table)} assets, "
        f"confidence_level={confidence_level:.0%}"
    )
    return table


def build_portfolio_risk_table(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Build a risk metrics table for the portfolio return series.

    Produces a single-row DataFrame with the same column schema as
    build_asset_risk_table, applied to the portfolio's return series.

    The row label is 'portfolio'.

    Args:
        portfolio_returns: pd.Series of portfolio returns (output of
            portfolio/metrics.portfolio_returns).
        confidence_level: VaR/TVaR confidence level in (0, 1).

    Returns:
        DataFrame of shape (1, 8) with row index ['portfolio'].

    Raises:
        ValueError: If portfolio_returns is empty.
    """
    if portfolio_returns.empty:
        raise ValueError("build_portfolio_risk_table: portfolio_returns is empty")

    daily_mean = float(portfolio_returns.mean())
    daily_std = float(portfolio_returns.std())

    hist_var = historical_var(portfolio_returns, confidence_level)
    param_var = parametric_var(daily_mean, daily_std, confidence_level)
    hist_tvar = historical_tvar(portfolio_returns, confidence_level)

    table = pd.DataFrame(
        {
            "mean_return": [daily_mean],
            "volatility": [daily_std],
            "historical_var_return": [hist_var],
            "historical_var_loss": [float(var_loss(hist_var))],
            "parametric_var_return": [float(param_var)],
            "parametric_var_loss": [float(var_loss(param_var))],
            "historical_tvar_return": [hist_tvar],
            "historical_tvar_loss": [float(tvar_loss(hist_tvar))],
        },
        index=["portfolio"],
    )

    logger.info(
        f"Portfolio risk table built: confidence_level={confidence_level:.0%}"
    )
    return table


# ── Columns extracted for the comparison view ──────────────────────────────────
_COMPARISON_COLS: list[str] = [
    "mean_return",
    "volatility",
    "historical_var_loss",
    "parametric_var_loss",
    "historical_tvar_loss",
]


def build_risk_comparison_table(
    asset_risk_table: pd.DataFrame,
    portfolio_risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a side-by-side comparison of risk metrics for assets and portfolio.

    Selects five key metrics from each input table and concatenates them into
    a single DataFrame so individual assets and the portfolio can be evaluated
    in one view. The portfolio row appears last, clearly separated from assets.

    Columns in the result:
      mean_return          — daily arithmetic mean return
      volatility           — daily standard deviation
      historical_var_loss  — empirical VaR expressed as a positive loss
      parametric_var_loss  — Gaussian VaR expressed as a positive loss
      historical_tvar_loss — Expected Shortfall expressed as a positive loss

    All metrics share the same daily horizon so comparisons are valid.
    Lower loss values indicate lower risk; the portfolio should show lower
    risk than the individual assets if diversification is effective.

    Args:
        asset_risk_table: Output of build_asset_risk_table (n_assets x 8).
            Must contain the five comparison columns.
        portfolio_risk_table: Output of build_portfolio_risk_table (1 x 8).
            Must contain the five comparison columns.

    Returns:
        DataFrame of shape (n_assets + 1, 5) with asset tickers as the
        first n rows and 'portfolio' as the last row.

    Raises:
        ValueError: If either table is empty or is missing required columns.
    """
    if asset_risk_table.empty:
        raise ValueError(
            "build_risk_comparison_table: asset_risk_table is empty"
        )
    if portfolio_risk_table.empty:
        raise ValueError(
            "build_risk_comparison_table: portfolio_risk_table is empty"
        )

    for label, tbl in [
        ("asset_risk_table", asset_risk_table),
        ("portfolio_risk_table", portfolio_risk_table),
    ]:
        missing = [c for c in _COMPARISON_COLS if c not in tbl.columns]
        if missing:
            raise ValueError(
                f"build_risk_comparison_table: {label} is missing columns: {missing}. "
                f"Available: {list(tbl.columns)}"
            )

    combined = pd.concat(
        [
            asset_risk_table[_COMPARISON_COLS],
            portfolio_risk_table[_COMPARISON_COLS],
        ],
        axis=0,
    )

    logger.info(
        f"Risk comparison table built: {len(combined)} rows "
        f"({len(asset_risk_table)} assets + 1 portfolio)"
    )
    return combined


def build_strategy_comparison_table(results: dict) -> pd.DataFrame:
    """Build a strategies × metrics comparison table.

    Args:
        results: Dict mapping strategy_name -> {"metrics": dict, ...}
            as returned by run_strategy_comparison after augmenting each
            entry with a "metrics" key from compute_strategy_metrics.

    Returns:
        DataFrame with strategy names as the index and metric names as
        columns, sorted by sharpe_ratio descending.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("build_strategy_comparison_table: results dict is empty")

    rows = {name: data["metrics"] for name, data in results.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "strategy"

    if "sharpe_ratio" in df.columns:
        df = df.sort_values("sharpe_ratio", ascending=False)

    logger.info(
        f"Strategy comparison table built: {len(df)} strategies, "
        f"{len(df.columns)} metrics"
    )
    return df


def build_factor_summary_table(factor_results_df: pd.DataFrame) -> pd.DataFrame:
    """Augment a factor regression table with a significance flag for alpha.

    Adds a boolean column 'alpha_significant' (True when p_alpha < 0.05)
    to the regression results DataFrame produced by
    factors.metrics.run_factor_analysis_for_assets or
    factors.metrics.run_factor_analysis_for_strategies.

    The table is returned sorted by annualized alpha descending so that
    the highest-alpha assets/strategies appear first.

    Args:
        factor_results_df: DataFrame from run_factor_analysis_for_assets or
            run_factor_analysis_for_strategies. Must contain at least the
            columns 'alpha' and 'p_alpha'.

    Returns:
        Copy of factor_results_df with an additional 'alpha_significant'
        column (bool), sorted by 'alpha' descending.

    Raises:
        ValueError: If factor_results_df is empty or missing required columns.
    """
    if factor_results_df.empty:
        raise ValueError("build_factor_summary_table: factor_results_df is empty")

    for col in ("alpha", "p_alpha"):
        if col not in factor_results_df.columns:
            raise ValueError(
                f"build_factor_summary_table: missing required column '{col}'. "
                f"Available: {list(factor_results_df.columns)}"
            )

    result = factor_results_df.copy()
    result["alpha_significant"] = result["p_alpha"] < 0.05
    result = result.sort_values("alpha", ascending=False)

    n_sig = int(result["alpha_significant"].sum())
    logger.info(
        f"Factor summary table built: {len(result)} rows  "
        f"alpha significant (p<0.05): {n_sig}/{len(result)}"
    )
    return result
