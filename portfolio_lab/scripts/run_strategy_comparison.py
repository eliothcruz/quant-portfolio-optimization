"""Multi-strategy comparison backtest (Phase 8).

Compares min_variance, max_sharpe, max_sharpe_shrinkage, and risk_parity
under identical conditions: same data, rebalancing schedule, weight limits,
and transaction costs.

Reads:   data/processed/returns.csv
Writes:  outputs/tables/strategy_comparison.csv
         outputs/tables/strategy_returns.csv
         outputs/figures/strategy_comparison.png

Usage (from portfolio_lab/ directory):
    python scripts/run_strategy_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analytics.performance import compute_strategy_metrics
from src.backtesting.engine import backtest_portfolio_multi
from src.data.loader import load_processed
from src.reporting.export import save_figure, save_table
from src.reporting.plots import plot_strategy_comparison
from src.reporting.tables import build_strategy_comparison_table
from src.utils.logger import get_logger
from src.utils.paths import OUTPUTS_FIGURES, OUTPUTS_TABLES

logger = get_logger("run_strategy_comparison")

# ── Parameters (identical for all strategies) ─────────────────────────────────
STRATEGIES = ["min_variance", "max_sharpe", "max_sharpe_shrinkage", "risk_parity"]
REBALANCE_FREQ = "M"
WINDOW = 252
MAX_WEIGHT = 0.4
TRANSACTION_COST = 0.001
BENCHMARK_TICKER = "SPY"
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load returns ──────────────────────────────────────────────────
    logger.info("Step 1/5 — Loading returns")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations")
    logger.info(f"  Date range: {returns.index[0].date()} — {returns.index[-1].date()}")

    if BENCHMARK_TICKER not in returns.columns:
        raise ValueError(f"Benchmark '{BENCHMARK_TICKER}' not found. Available: {tickers}")
    benchmark_returns = returns[BENCHMARK_TICKER]

    # ── Step 2: Run multi-strategy backtest ───────────────────────────────────
    logger.info(
        f"Step 2/5 — Running {len(STRATEGIES)} strategies  "
        f"freq={REBALANCE_FREQ}  window={WINDOW}  max_w={MAX_WEIGHT}  "
        f"cost={TRANSACTION_COST}"
    )
    results = backtest_portfolio_multi(
        returns_df=returns,
        strategies=STRATEGIES,
        rebalance_freq=REBALANCE_FREQ,
        window=WINDOW,
        max_weight=MAX_WEIGHT,
        transaction_cost=TRANSACTION_COST,
    )

    # ── Step 3: Compute metrics for each strategy ─────────────────────────────
    logger.info("Step 3/5 — Computing performance metrics")
    for strategy, data in results.items():
        metrics = compute_strategy_metrics(
            data["returns"], data["weights"]
        )
        data["metrics"] = metrics
        logger.info(
            f"  [{strategy:25s}]  "
            f"cum={metrics['cumulative_return']:+.3f}  "
            f"SR={metrics['sharpe_ratio']:.3f}  "
            f"MDD={metrics['max_drawdown']:.3f}  "
            f"TO={metrics['avg_turnover']:.3f}"
        )

    # ── Step 4: Save tables ───────────────────────────────────────────────────
    logger.info("Step 4/5 — Saving tables")

    comparison_df = build_strategy_comparison_table(results)
    save_table(comparison_df, OUTPUTS_TABLES / "strategy_comparison.csv")

    # Align all strategy returns on the same DatetimeIndex
    strategy_returns_df = pd.DataFrame(
        {name: data["returns"] for name, data in results.items()}
    )
    save_table(strategy_returns_df, OUTPUTS_TABLES / "strategy_returns.csv")

    # ── Step 5: Generate comparison chart ────────────────────────────────────
    logger.info("Step 5/5 — Generating comparison chart")
    ref_index = next(iter(results.values()))["returns"].index
    bm_aligned = benchmark_returns.reindex(ref_index)

    fig = plot_strategy_comparison(
        strategy_returns={name: data["returns"] for name, data in results.items()},
        benchmark_returns=bm_aligned,
        comparison_df=comparison_df,
    )
    save_figure(fig, OUTPUTS_FIGURES / "strategy_comparison.png", dpi=150)

    logger.info("Strategy comparison complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
