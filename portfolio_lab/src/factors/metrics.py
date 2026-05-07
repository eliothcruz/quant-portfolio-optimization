"""Batch factor analysis across multiple assets or strategies.

Orchestrates regression.run_factor_regression for each column of a return
DataFrame and collects results into a tidy summary table.
"""

import pandas as pd

from .regression import run_factor_regression
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Column ordering for the output tables
_CAPM_COLS: list[str] = [
    "alpha", "beta_mkt",
    "t_alpha", "p_alpha",
    "t_beta_mkt", "p_beta_mkt",
    "r_squared", "n_obs",
]

_FF3_COLS: list[str] = [
    "alpha", "beta_mkt", "beta_smb", "beta_hml",
    "t_alpha", "p_alpha",
    "t_beta_mkt", "p_beta_mkt",
    "t_beta_smb", "p_beta_smb",
    "t_beta_hml", "p_beta_hml",
    "r_squared", "n_obs",
]


def run_factor_analysis_for_assets(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    model: str = "ff3",
) -> pd.DataFrame:
    """Run factor regression for each asset in returns_df.

    Args:
        returns_df: DataFrame of asset daily returns (DatetimeIndex, columns=tickers).
            Must already be aligned with factors_df — call
            loader.align_returns_with_factors() first.
        factors_df: Aligned factor DataFrame (same index as returns_df).
        model: 'capm' or 'ff3' (default 'ff3').

    Returns:
        DataFrame with one row per asset. Columns depend on model:
            - CAPM: alpha, beta_mkt, t_alpha, p_alpha, t_beta_mkt, p_beta_mkt,
                    r_squared, n_obs
            - FF3:  above + beta_smb, beta_hml and their t/p statistics

    Raises:
        ValueError: If returns_df is empty.
    """
    if returns_df.empty:
        raise ValueError("run_factor_analysis_for_assets: returns_df is empty")

    records: dict[str, dict] = {}
    for ticker in returns_df.columns:
        try:
            records[ticker] = run_factor_regression(
                returns_df[ticker], factors_df, model=model
            )
            logger.info(
                f"  {ticker:8s} | alpha={records[ticker]['alpha']:.4f}  "
                f"beta_mkt={records[ticker]['beta_mkt']:.4f}  "
                f"R²={records[ticker]['r_squared']:.4f}"
            )
        except Exception as exc:
            logger.warning(f"  {ticker}: regression failed — {exc}")

    if not records:
        raise RuntimeError(
            "run_factor_analysis_for_assets: all regressions failed"
        )

    df = pd.DataFrame(records).T
    df.index.name = "asset"
    col_order = _FF3_COLS if model == "ff3" else _CAPM_COLS
    df = df[[c for c in col_order if c in df.columns]]

    logger.info(
        f"Factor analysis ({model.upper()}) complete: "
        f"{len(df)} assets  model={model}"
    )
    return df


def run_factor_analysis_for_strategies(
    strategy_returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    model: str = "ff3",
) -> pd.DataFrame:
    """Run factor regression for each strategy return series.

    Identical logic to run_factor_analysis_for_assets but semantically
    applied to strategy-level return series (one column per strategy).

    Args:
        strategy_returns_df: DataFrame of strategy daily returns
            (DatetimeIndex, columns=strategy names).
            Must already be aligned with factors_df.
        factors_df: Aligned factor DataFrame.
        model: 'capm' or 'ff3' (default 'ff3').

    Returns:
        DataFrame with one row per strategy, same column schema as
        run_factor_analysis_for_assets.

    Raises:
        ValueError: If strategy_returns_df is empty.
    """
    if strategy_returns_df.empty:
        raise ValueError(
            "run_factor_analysis_for_strategies: strategy_returns_df is empty"
        )

    logger.info(
        f"Running {model.upper()} factor analysis for "
        f"{len(strategy_returns_df.columns)} strategies"
    )
    result = run_factor_analysis_for_assets(strategy_returns_df, factors_df, model=model)
    result.index.name = "strategy"
    return result
