"""Rolling Markowitz backtest with rebalancing and transaction costs (Phase 7).

Reads:   data/processed/returns.csv
Writes:  outputs/tables/backtest_returns.csv
         outputs/tables/backtest_weights.csv
         outputs/tables/backtest_metrics.csv
         outputs/figures/backtest_results.png

Usage (from portfolio_lab/ directory):
    python scripts/run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analytics.performance import compute_performance_metrics
from src.backtesting.engine import backtest_portfolio
from src.data.loader import load_processed
from src.reporting.export import save_figure, save_table
from src.reporting.plots import plot_backtest_results
from src.utils.logger import get_logger
from src.utils.paths import OUTPUTS_FIGURES, OUTPUTS_TABLES

logger = get_logger("run_backtest")

# ── Backtest parameters ────────────────────────────────────────────────────────
REBALANCE_FREQ = "M"       # "D" daily | "M" monthly | "Q" quarterly
WINDOW = 252               # rolling estimation window (trading days)
STRATEGY = "max_sharpe"    # "max_sharpe" | "min_variance"
MAX_WEIGHT = 0.4           # maximum weight per asset (concentration limit)
MIN_WEIGHT = 0.0           # minimum weight per asset
TRANSACTION_COST = 0.001   # 10 bps per unit of turnover
BENCHMARK_TICKER = "SPY"   # column in returns.csv used as benchmark
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load returns ──────────────────────────────────────────────────
    logger.info("Step 1/5 — Loading returns")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations")
    logger.info(f"  Tickers: {tickers}")
    logger.info(f"  Date range: {returns.index[0].date()} — {returns.index[-1].date()}")

    if BENCHMARK_TICKER not in returns.columns:
        raise ValueError(
            f"Benchmark '{BENCHMARK_TICKER}' not found in returns.csv. "
            f"Available: {tickers}"
        )
    benchmark_returns = returns[BENCHMARK_TICKER]

    # ── Step 2: Run backtest ──────────────────────────────────────────────────
    logger.info(
        f"Step 2/5 — Running backtest  "
        f"strategy={STRATEGY}  freq={REBALANCE_FREQ}  window={WINDOW}  "
        f"max_w={MAX_WEIGHT}  min_w={MIN_WEIGHT}  cost={TRANSACTION_COST}"
    )
    portfolio_returns, weights_history = backtest_portfolio(
        returns_df=returns,
        rebalance_freq=REBALANCE_FREQ,
        window=WINDOW,
        strategy=STRATEGY,
        max_weight=MAX_WEIGHT,
        min_weight=MIN_WEIGHT,
        transaction_cost=TRANSACTION_COST,
    )
    logger.info(
        f"  Backtest period: {portfolio_returns.index[0].date()} "
        f"— {portfolio_returns.index[-1].date()}"
    )
    logger.info(f"  Trading days: {len(portfolio_returns)}")
    logger.info(f"  Rebalance dates: {len(weights_history)}")

    # ── Step 3: Performance metrics ───────────────────────────────────────────
    logger.info("Step 3/5 — Computing performance metrics")
    bm_aligned = benchmark_returns.reindex(portfolio_returns.index)

    port_metrics = compute_performance_metrics(portfolio_returns)
    bm_metrics = compute_performance_metrics(bm_aligned)

    logger.info("  Portfolio metrics:")
    for k, v in port_metrics.items():
        logger.info(f"    {k:30s}: {v}")
    logger.info("  Benchmark metrics:")
    for k, v in bm_metrics.items():
        logger.info(f"    {k:30s}: {v}")

    metrics_df = pd.DataFrame(
        {"portfolio": port_metrics, "benchmark": bm_metrics}
    )

    # ── Step 4: Save tables ───────────────────────────────────────────────────
    logger.info("Step 4/5 — Saving tables")

    returns_path = OUTPUTS_TABLES / "backtest_returns.csv"
    portfolio_returns.to_frame().to_csv(returns_path)
    logger.info(f"  Saved -> {returns_path.name}")

    weights_path = OUTPUTS_TABLES / "backtest_weights.csv"
    weights_history.to_csv(weights_path)
    logger.info(f"  Saved -> {weights_path.name}")

    metrics_path = OUTPUTS_TABLES / "backtest_metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"  Saved -> {metrics_path.name}")

    # ── Step 5: Generate chart ────────────────────────────────────────────────
    logger.info("Step 5/5 — Generating backtest chart")
    fig = plot_backtest_results(
        portfolio_returns=portfolio_returns,
        benchmark_returns=bm_aligned,
        weights_history=weights_history,
    )
    save_figure(fig, OUTPUTS_FIGURES / "backtest_results.png", dpi=150)

    logger.info("Backtest complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
