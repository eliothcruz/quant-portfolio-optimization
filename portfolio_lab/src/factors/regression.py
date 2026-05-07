"""OLS factor regressions: CAPM and Fama-French 3-Factor.

All regressions are estimated via statsmodels OLS with heteroskedasticity-
consistent standard errors (HC3). The dependent variable is the asset excess
return (asset_return − RF), and the regressors are the factor returns.

Regression results are returned as plain dicts so that callers can build
DataFrames without coupling to statsmodels object internals.

Scale requirement:
    Returns and factors must be in the same unit (both decimal or both
    percent). The pipeline uses decimal returns; factor files from Kenneth
    French use percent and must be converted before calling these functions.
    See loader.load_factor_data(convert_percent_to_decimal=True).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..utils.logger import get_logger

logger = get_logger(__name__)

_VALID_MODELS = ("capm", "ff3")


def run_capm(
    asset_returns: pd.Series,
    factors_df: pd.DataFrame,
) -> dict:
    """Estimate the CAPM single-factor regression for one asset.

    Model:
        r_i − RF = α + β_mkt · MKT_RF + ε

    Args:
        asset_returns: Daily return series for one asset (DatetimeIndex).
            Must be aligned with factors_df (same index).
        factors_df: DataFrame with columns MKT_RF and RF, same index
            as asset_returns.

    Returns:
        Dict with keys:
            alpha        — annualized intercept (daily α × 252)
            beta_mkt     — market beta
            t_alpha      — t-statistic for alpha (HC3)
            p_alpha      — two-sided p-value for alpha
            t_beta_mkt   — t-statistic for beta_mkt (HC3)
            p_beta_mkt   — two-sided p-value for beta_mkt
            r_squared    — R² of the regression
            n_obs        — number of observations used

    Raises:
        ValueError: If required columns are missing or the series is empty.
    """
    _validate_factor_columns(factors_df, ["MKT_RF", "RF"])
    _validate_aligned(asset_returns, factors_df)

    excess_ret = asset_returns - factors_df["RF"]
    X = sm.add_constant(factors_df[["MKT_RF"]], has_constant="add")

    result = sm.OLS(excess_ret, X).fit(cov_type="HC3")

    alpha_daily = float(result.params["const"])
    beta_mkt = float(result.params["MKT_RF"])

    return {
        "alpha": alpha_daily * 252,         # annualized
        "beta_mkt": beta_mkt,
        "beta_smb": np.nan,
        "beta_hml": np.nan,
        "t_alpha": float(result.tvalues["const"]),
        "p_alpha": float(result.pvalues["const"]),
        "t_beta_mkt": float(result.tvalues["MKT_RF"]),
        "p_beta_mkt": float(result.pvalues["MKT_RF"]),
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
    }


def run_ff3(
    asset_returns: pd.Series,
    factors_df: pd.DataFrame,
) -> dict:
    """Estimate the Fama-French 3-Factor regression for one asset.

    Model:
        r_i − RF = α + β_mkt · MKT_RF + β_smb · SMB + β_hml · HML + ε

    Args:
        asset_returns: Daily return series for one asset (DatetimeIndex).
            Must be aligned with factors_df (same index).
        factors_df: DataFrame with columns MKT_RF, SMB, HML, RF, same index
            as asset_returns.

    Returns:
        Dict with keys:
            alpha        — annualized intercept (daily α × 252)
            beta_mkt     — market beta
            beta_smb     — size factor loading
            beta_hml     — value factor loading
            t_alpha      — t-statistic for alpha (HC3)
            p_alpha      — two-sided p-value for alpha
            t_beta_mkt   — t-statistic for beta_mkt (HC3)
            p_beta_mkt   — two-sided p-value for beta_mkt
            t_beta_smb   — t-statistic for beta_smb (HC3)
            p_beta_smb   — two-sided p-value for beta_smb
            t_beta_hml   — t-statistic for beta_hml (HC3)
            p_beta_hml   — two-sided p-value for beta_hml
            r_squared    — R² of the regression
            n_obs        — number of observations used

    Raises:
        ValueError: If required columns are missing or the series is empty.
    """
    _validate_factor_columns(factors_df, ["MKT_RF", "SMB", "HML", "RF"])
    _validate_aligned(asset_returns, factors_df)

    excess_ret = asset_returns - factors_df["RF"]
    X = sm.add_constant(factors_df[["MKT_RF", "SMB", "HML"]], has_constant="add")

    result = sm.OLS(excess_ret, X).fit(cov_type="HC3")

    alpha_daily = float(result.params["const"])

    return {
        "alpha": alpha_daily * 252,
        "beta_mkt": float(result.params["MKT_RF"]),
        "beta_smb": float(result.params["SMB"]),
        "beta_hml": float(result.params["HML"]),
        "t_alpha": float(result.tvalues["const"]),
        "p_alpha": float(result.pvalues["const"]),
        "t_beta_mkt": float(result.tvalues["MKT_RF"]),
        "p_beta_mkt": float(result.pvalues["MKT_RF"]),
        "t_beta_smb": float(result.tvalues["SMB"]),
        "p_beta_smb": float(result.pvalues["SMB"]),
        "t_beta_hml": float(result.tvalues["HML"]),
        "p_beta_hml": float(result.pvalues["HML"]),
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
    }


def run_factor_regression(
    asset_returns: pd.Series,
    factors_df: pd.DataFrame,
    model: str = "ff3",
) -> dict:
    """Dispatch a factor regression by model name.

    Args:
        asset_returns: Daily return series for one asset.
        factors_df: Factor DataFrame aligned to asset_returns.
        model: One of 'capm' or 'ff3' (default 'ff3').

    Returns:
        Regression result dict (see run_capm or run_ff3 for keys).

    Raises:
        ValueError: If model is not one of the supported options.
    """
    if model not in _VALID_MODELS:
        raise ValueError(
            f"run_factor_regression: unknown model '{model}'. "
            f"Valid options: {_VALID_MODELS}"
        )
    if model == "capm":
        return run_capm(asset_returns, factors_df)
    return run_ff3(asset_returns, factors_df)


# ── Internal validators ────────────────────────────────────────────────────────

def _validate_factor_columns(factors_df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in factors_df.columns]
    if missing:
        raise ValueError(
            f"factors_df is missing required columns: {missing}. "
            f"Available: {list(factors_df.columns)}"
        )


def _validate_aligned(series: pd.Series, factors_df: pd.DataFrame) -> None:
    if series.empty:
        raise ValueError("asset_returns is empty")
    if not series.index.equals(factors_df.index):
        raise ValueError(
            "asset_returns and factors_df have different indices. "
            "Call loader.align_returns_with_factors() first."
        )
