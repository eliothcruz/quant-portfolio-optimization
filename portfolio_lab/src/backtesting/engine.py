"""Rolling-window portfolio backtest with rebalancing and transaction costs.

Each rebalance period uses only historical data ending on the rebalance date
(no look-ahead bias). Transaction costs are applied as a flat turnover charge
on the first day of each new period.

Supported strategies: "max_sharpe", "min_variance", "max_sharpe_shrinkage",
    "risk_parity", "factor_alpha_weighted".
Supported rebalance frequencies: "D" (daily), "M" (monthly), "Q" (quarterly).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..analytics.covariance import compute_covariance_matrix, compute_shrinkage_covariance
from ..analytics.statistics import compute_mean_returns
from ..factors.metrics import run_factor_analysis_for_assets
from ..portfolio.constraints import full_investment_constraint, weight_bounds
from ..portfolio.optimization import factor_alpha_weighted_portfolio
from ..utils.logger import get_logger

logger = get_logger(__name__)

_MIN_WINDOW_OBSERVATIONS = 30  # absolute floor to estimate covariance

# Pandas 2.2+ renamed frequency aliases: "M" -> "ME", "Q" -> "QE"
_FREQ_ALIASES: dict[str, str] = {"M": "ME", "Q": "QE"}

_VALID_STRATEGIES = frozenset(
    {
        "max_sharpe",
        "min_variance",
        "max_sharpe_shrinkage",
        "risk_parity",
        "factor_alpha_weighted",
    }
)


def _optimize_constrained(
    mu: pd.Series,
    cov: pd.DataFrame,
    strategy: str,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """Optimize portfolio weights with custom bounds.

    Returns equal weights on convergence failure rather than raising, so that
    the backtest loop can continue across all periods.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(cov, dtype=float)
    n = len(mu_arr)

    bounds = weight_bounds(n, min_weight, max_weight)
    constraints = [full_investment_constraint()]
    x0 = np.ones(n) / n

    if strategy == "max_sharpe":
        def objective(w: np.ndarray) -> float:
            p_ret = float(w @ mu_arr)
            p_var = float(w @ sigma_arr @ w)
            if p_var <= 0.0:
                return 0.0
            return -(p_ret / np.sqrt(p_var))
    elif strategy == "risk_parity":
        def objective(w: np.ndarray) -> float:
            sigma_w = sigma_arr @ w
            rc = w * sigma_w
            rc_mean = rc.mean()
            return float(np.sum((rc - rc_mean) ** 2))
    else:  # min_variance
        def objective(w: np.ndarray) -> float:
            return float(w @ sigma_arr @ w)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500, "disp": False},
    )

    if not result.success:
        logger.warning(
            f"Optimizer did not converge ({result.message}); "
            "falling back to equal weights for this period"
        )
        return x0

    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()
    return weights


def backtest_portfolio(
    returns_df: pd.DataFrame,
    rebalance_freq: str = "M",
    window: int = 252,
    strategy: str = "max_sharpe",
    max_weight: float = 0.4,
    min_weight: float = 0.0,
    transaction_cost: float = 0.001,
    factors_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate a rolling portfolio with periodic rebalancing.

    At each rebalance date the model is re-estimated using the preceding
    `window` trading days. Weights are then applied to the subsequent period
    until the next rebalance. Transaction costs are deducted from the first
    day of each new period.

    Args:
        returns_df: DataFrame of daily asset returns (DatetimeIndex, one column
            per asset). Must have no NaN values in the data used for estimation.
        rebalance_freq: Rebalance frequency — "D" daily, "M" monthly,
            "Q" quarterly. Passed to pd.DataFrame.resample().
        window: Number of trading days used to estimate mu and Sigma.
            Must be > number of assets.
        strategy: One of "max_sharpe", "min_variance", "max_sharpe_shrinkage",
            "risk_parity", "factor_alpha_weighted".
        max_weight: Maximum weight per asset (0 < max_weight <= 1).
        min_weight: Minimum weight per asset (0 <= min_weight).
        transaction_cost: Proportional cost applied to portfolio turnover
            at each rebalance (e.g. 0.001 = 10 bps).
        factors_df: Factor DataFrame (DatetimeIndex, columns include MKT_RF,
            SMB, HML, RF in decimal units). Required when
            strategy="factor_alpha_weighted"; ignored otherwise.

    Returns:
        Tuple of:
            portfolio_returns  - pd.Series of daily portfolio returns
                                 (DatetimeIndex, name="portfolio_return")
            weights_history    - pd.DataFrame of weights at each rebalance date
                                 (DatetimeIndex, columns=tickers)

    Raises:
        ValueError: If strategy is unknown, window is too small, or
            strategy="factor_alpha_weighted" without factors_df.
        RuntimeError: If no valid rebalance periods could be generated.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use one of: {sorted(_VALID_STRATEGIES)}"
        )
    if strategy == "factor_alpha_weighted" and factors_df is None:
        raise ValueError(
            "strategy='factor_alpha_weighted' requires factors_df. "
            "Pass a factor DataFrame aligned to the same date range as returns_df."
        )

    n_assets = len(returns_df.columns)
    if window <= n_assets:
        raise ValueError(
            f"window={window} must be greater than n_assets={n_assets} "
            "to ensure the covariance matrix is full rank"
        )

    tickers = list(returns_df.columns)
    dates = returns_df.index

    # Normalize frequency alias for pandas 2.2+ compatibility
    freq = _FREQ_ALIASES.get(rebalance_freq.upper(), rebalance_freq)

    # Rebalance dates = last trading day of each period
    rebalance_dates = (
        returns_df.resample(freq)
        .last()
        .dropna(how="all")
        .index
    )

    port_returns_list: list[tuple] = []
    weights_records: dict = {}
    prev_weights = np.ones(n_assets) / n_assets  # equal weights as prior

    n_rebalances = 0

    for k, rebal_date in enumerate(rebalance_dates):
        # Position of the day AFTER rebal_date in returns_df
        pos = int(dates.searchsorted(rebal_date, side="right"))

        if pos < window:
            continue  # not enough history yet

        # Training window: [pos-window, pos) — ends exactly at rebal_date
        window_data = returns_df.iloc[pos - window : pos]

        if len(window_data) < max(_MIN_WINDOW_OBSERVATIONS, n_assets + 1):
            continue

        fallback_weights = np.ones(n_assets) / n_assets

        if strategy == "factor_alpha_weighted":
            common_dates = window_data.index.intersection(factors_df.index)
            if len(common_dates) < _MIN_WINDOW_OBSERVATIONS:
                logger.warning(
                    f"Insufficient factor overlap at {rebal_date} "
                    f"({len(common_dates)} obs < {_MIN_WINDOW_OBSERVATIONS}); "
                    "falling back to equal weights"
                )
                new_weights = fallback_weights
            else:
                aligned_rets = window_data.loc[common_dates]
                aligned_facs = factors_df.loc[common_dates]
                try:
                    factor_results = run_factor_analysis_for_assets(
                        aligned_rets, aligned_facs, model="ff3"
                    )
                    w_series = factor_alpha_weighted_portfolio(
                        factor_results, max_weight=max_weight, min_weight=min_weight
                    )
                    reindexed = w_series.reindex(tickers).fillna(0.0)
                    total = reindexed.sum()
                    new_weights = (reindexed / total).values if total > 0 else fallback_weights
                except Exception as exc:
                    logger.warning(
                        f"factor_alpha_weighted failed at {rebal_date} ({exc}); "
                        "falling back to equal weights"
                    )
                    new_weights = fallback_weights
        else:
            # Estimate parameters from training window
            mu = compute_mean_returns(window_data, annualize=True)
            if strategy == "max_sharpe_shrinkage":
                cov_opt = compute_shrinkage_covariance(window_data, annualize=True)
                new_weights = _optimize_constrained(mu, cov_opt, "max_sharpe", min_weight, max_weight)
            else:
                cov = compute_covariance_matrix(window_data, annualize=True)
                new_weights = _optimize_constrained(mu, cov, strategy, min_weight, max_weight)

        weights_records[rebal_date] = dict(zip(tickers, new_weights))

        # Application period: (rebal_date, next_rebal_date]
        if k + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[k + 1]
            next_pos = int(dates.searchsorted(next_rebal, side="right"))
        else:
            next_pos = len(dates)

        period_data = returns_df.iloc[pos:next_pos]
        if period_data.empty:
            continue

        # Transaction cost = turnover * rate
        turnover = float(np.sum(np.abs(new_weights - prev_weights)))
        cost = transaction_cost * turnover

        for j, (date, row) in enumerate(period_data.iterrows()):
            p_ret = float(np.dot(new_weights, row.values))
            if j == 0:
                p_ret -= cost
            port_returns_list.append((date, p_ret))

        prev_weights = new_weights.copy()
        n_rebalances += 1

    if not port_returns_list:
        raise RuntimeError(
            "backtest_portfolio: no valid periods generated. "
            "Try reducing 'window' or using a longer return series."
        )

    portfolio_returns = pd.Series(
        {d: r for d, r in port_returns_list},
        name="portfolio_return",
    )
    portfolio_returns.index.name = "date"

    weights_df = pd.DataFrame(weights_records).T
    weights_df.index.name = "date"
    weights_df = weights_df[tickers]

    logger.info(
        f"Backtest complete: strategy={strategy}  freq={rebalance_freq}  "
        f"window={window}  rebalances={n_rebalances}  "
        f"periods={len(portfolio_returns)}"
    )
    return portfolio_returns, weights_df


def backtest_portfolio_multi(
    returns_df: pd.DataFrame,
    strategies: list[str] | None = None,
    rebalance_freq: str = "M",
    window: int = 252,
    max_weight: float = 0.4,
    transaction_cost: float = 0.001,
    factors_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Run backtest_portfolio for multiple strategies under identical conditions.

    All strategies receive the same data, rebalancing schedule, weight limits,
    and transaction cost model, ensuring a fair comparison.

    Args:
        returns_df: DataFrame of daily asset returns.
        strategies: List of strategy names to run. Defaults to the four MVO
            strategies: ["min_variance", "max_sharpe", "max_sharpe_shrinkage",
            "risk_parity"]. "factor_alpha_weighted" requires factors_df.
        rebalance_freq: Rebalance frequency passed to backtest_portfolio.
        window: Estimation window in trading days.
        max_weight: Maximum weight per asset (applied uniformly).
        transaction_cost: Proportional transaction cost per unit of turnover.
        factors_df: Factor DataFrame required when "factor_alpha_weighted" is
            in strategies. Ignored for all other strategies. Default None.

    Returns:
        Dict mapping strategy_name -> {"returns": pd.Series, "weights": pd.DataFrame}.

    Raises:
        ValueError: If any strategy name is invalid or "factor_alpha_weighted"
            is requested without factors_df.
    """
    if strategies is None:
        strategies = ["min_variance", "max_sharpe", "max_sharpe_shrinkage", "risk_parity"]

    invalid = [s for s in strategies if s not in _VALID_STRATEGIES]
    if invalid:
        raise ValueError(
            f"Unknown strategies: {invalid}. Valid: {sorted(_VALID_STRATEGIES)}"
        )

    results: dict[str, dict] = {}
    for strategy in strategies:
        logger.info(f"Starting backtest for strategy='{strategy}'")
        port_ret, weights_df = backtest_portfolio(
            returns_df=returns_df,
            rebalance_freq=rebalance_freq,
            window=window,
            strategy=strategy,
            max_weight=max_weight,
            min_weight=0.0,
            transaction_cost=transaction_cost,
            factors_df=factors_df,
        )
        results[strategy] = {"returns": port_ret, "weights": weights_df}

    logger.info(
        f"Multi-strategy backtest complete: {len(results)} strategies, "
        f"freq={rebalance_freq}  window={window}"
    )
    return results
