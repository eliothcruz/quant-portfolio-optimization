"""Multi-strategy comparison backtest (Phases 8 + 10B).

Compares min_variance, max_sharpe, max_sharpe_shrinkage, risk_parity, and
(when factor data is available) factor_alpha_weighted under identical
conditions: same data, rebalancing schedule, weight limits, and transaction
costs.

Reads:   data/processed/returns.csv
         data/factors/fama_french_3_factors.csv  (optional — enables
             factor_alpha_weighted strategy and latest-weights output)
         config/factors.yaml                      (optional — factor settings)
Writes:  outputs/tables/strategy_comparison.csv
         outputs/tables/strategy_returns.csv
         outputs/figures/strategy_comparison.png
         outputs/tables/factor_alpha_weights_latest.csv  (if factor data found)
         outputs/figures/factor_alpha_weights_latest.png (if factor data found)

Usage (from portfolio_lab/ directory):
    python scripts/run_strategy_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd

from src.analytics.performance import compute_strategy_metrics
from src.backtesting.engine import backtest_portfolio_multi
from src.data.loader import load_processed
from src.factors.loader import load_factor_data
from src.factors.metrics import run_factor_analysis_for_assets
from src.reporting.export import save_figure, save_table
from src.reporting.plots import (
    plot_factor_alpha_weights,
    plot_strategy_comparison,
)
from src.reporting.tables import build_strategy_comparison_table
from src.utils.logger import get_logger
from src.utils.paths import CONFIG_DIR, OUTPUTS_FIGURES, OUTPUTS_TABLES

logger = get_logger("run_strategy_comparison")

# ── Fixed parameters (identical for all strategies) ───────────────────────────
_BASE_STRATEGIES = [
    "min_variance",
    "max_sharpe",
    "max_sharpe_shrinkage",
    "risk_parity",
]
REBALANCE_FREQ = "M"
WINDOW = 252
MAX_WEIGHT = 0.4
TRANSACTION_COST = 0.001
BENCHMARK_TICKER = "SPY"
# ──────────────────────────────────────────────────────────────────────────────


def _load_factor_config() -> dict:
    config_path = CONFIG_DIR / "factors.yaml"
    if not config_path.exists():
        return {
            "factor_file": "data/factors/fama_french_3_factors.csv",
            "convert_percent_to_decimal": True,
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _try_load_factors() -> pd.DataFrame | None:
    """Load factor data; return None with a warning if the file is missing."""
    config = _load_factor_config()
    factor_path = Path(config["factor_file"])
    convert = bool(config.get("convert_percent_to_decimal", True))
    if not factor_path.exists():
        logger.warning(
            f"Factor file not found at '{factor_path}'. "
            "factor_alpha_weighted strategy will be skipped. "
            "Place Fama-French 3 Factor daily data there to enable it."
        )
        return None
    factors = load_factor_data(factor_path, convert_percent_to_decimal=convert)
    logger.info(
        f"Factor data loaded: {len(factors)} observations  "
        f"columns={list(factors.columns)}"
    )
    return factors


def _build_factor_alpha_weights_table(
    returns: pd.DataFrame,
    factors_df: pd.DataFrame,
    last_rebal_date: pd.Timestamp,
    weights_series: pd.Series,
) -> pd.DataFrame:
    """Recompute FF3 for the last rolling window and join with final weights."""
    pos = int(returns.index.searchsorted(last_rebal_date, side="right"))
    window_data = returns.iloc[max(0, pos - WINDOW) : pos]
    common = window_data.index.intersection(factors_df.index)
    if len(common) < 30:
        logger.warning(
            "Insufficient factor overlap to build latest-weights table. Skipping."
        )
        return pd.DataFrame()
    aligned_rets = window_data.loc[common]
    aligned_facs = factors_df.loc[common]
    factor_results = run_factor_analysis_for_assets(aligned_rets, aligned_facs, model="ff3")
    cols = [c for c in ("alpha", "p_alpha", "beta_mkt", "beta_smb", "beta_hml", "r_squared")
            if c in factor_results.columns]
    export_df = factor_results[cols].copy()
    export_df.insert(0, "weight", weights_series.reindex(export_df.index).fillna(0.0))
    return export_df


def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load returns ──────────────────────────────────────────────────
    logger.info("Step 1/6 — Loading returns")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations")
    logger.info(f"  Date range: {returns.index[0].date()} — {returns.index[-1].date()}")

    if BENCHMARK_TICKER not in returns.columns:
        raise ValueError(f"Benchmark '{BENCHMARK_TICKER}' not found. Available: {tickers}")
    benchmark_returns = returns[BENCHMARK_TICKER]

    # ── Step 2: Attempt to load factor data ──────────────────────────────────
    logger.info("Step 2/6 — Attempting to load factor data")
    factors_df = _try_load_factors()
    strategies = _BASE_STRATEGIES.copy()
    if factors_df is not None:
        strategies.append("factor_alpha_weighted")
        logger.info("  factor_alpha_weighted strategy enabled")
    else:
        logger.info("  factor_alpha_weighted strategy disabled (no factor file)")

    # ── Step 3: Run multi-strategy backtest ───────────────────────────────────
    logger.info(
        f"Step 3/6 — Running {len(strategies)} strategies  "
        f"freq={REBALANCE_FREQ}  window={WINDOW}  max_w={MAX_WEIGHT}  "
        f"cost={TRANSACTION_COST}"
    )
    results = backtest_portfolio_multi(
        returns_df=returns,
        strategies=strategies,
        rebalance_freq=REBALANCE_FREQ,
        window=WINDOW,
        max_weight=MAX_WEIGHT,
        transaction_cost=TRANSACTION_COST,
        factors_df=factors_df,
    )

    # ── Step 4: Compute metrics for each strategy ─────────────────────────────
    logger.info("Step 4/6 — Computing performance metrics")
    for strategy, data in results.items():
        metrics = compute_strategy_metrics(data["returns"], data["weights"])
        data["metrics"] = metrics
        logger.info(
            f"  [{strategy:30s}]  "
            f"cum={metrics['cumulative_return']:+.3f}  "
            f"SR={metrics['sharpe_ratio']:.3f}  "
            f"MDD={metrics['max_drawdown']:.3f}  "
            f"TO={metrics['avg_turnover']:.3f}"
        )

    # ── Step 5: Save tables ───────────────────────────────────────────────────
    logger.info("Step 5/6 — Saving tables")

    comparison_df = build_strategy_comparison_table(results)
    save_table(comparison_df, OUTPUTS_TABLES / "strategy_comparison.csv")

    strategy_returns_df = pd.DataFrame(
        {name: data["returns"] for name, data in results.items()}
    )
    save_table(strategy_returns_df, OUTPUTS_TABLES / "strategy_returns.csv")

    # Factor alpha latest weights table (only when strategy ran successfully)
    if "factor_alpha_weighted" in results and factors_df is not None:
        faw_weights_df = results["factor_alpha_weighted"]["weights"]
        if not faw_weights_df.empty:
            last_rebal = faw_weights_df.index[-1]
            last_weights = faw_weights_df.iloc[-1]
            latest_df = _build_factor_alpha_weights_table(
                returns, factors_df, last_rebal, last_weights
            )
            if not latest_df.empty:
                save_table(latest_df, OUTPUTS_TABLES / "factor_alpha_weights_latest.csv")
                logger.info(
                    f"  Latest factor alpha weights saved "
                    f"(window ending {last_rebal.date()})"
                )

    # ── Step 6: Generate charts ───────────────────────────────────────────────
    logger.info("Step 6/6 — Generating charts")
    ref_index = next(iter(results.values()))["returns"].index
    bm_aligned = benchmark_returns.reindex(ref_index)

    fig = plot_strategy_comparison(
        strategy_returns={name: data["returns"] for name, data in results.items()},
        benchmark_returns=bm_aligned,
        comparison_df=comparison_df,
    )
    save_figure(fig, OUTPUTS_FIGURES / "strategy_comparison.png", dpi=150)

    # Factor alpha latest weights figure
    if "factor_alpha_weighted" in results and factors_df is not None:
        latest_path = OUTPUTS_TABLES / "factor_alpha_weights_latest.csv"
        if latest_path.exists():
            latest_df = pd.read_csv(latest_path, index_col=0)
            fig_faw = plot_factor_alpha_weights(latest_df)
            save_figure(
                fig_faw,
                OUTPUTS_FIGURES / "factor_alpha_weights_latest.png",
                dpi=150,
            )

    logger.info("Strategy comparison complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
