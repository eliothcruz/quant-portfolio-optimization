"""Optimize portfolio weights and compute portfolio-level metrics.

Reads:   data/processed/returns.csv
Writes:  outputs/tables/portfolio_weights.csv
         outputs/tables/portfolio_metrics.csv
         outputs/figures/portfolio_weights.png

Usage (from portfolio_lab/ directory):
    python scripts/run_portfolio.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analytics.covariance import compute_covariance_matrix
from src.analytics.statistics import compute_mean_returns, summarize_asset_statistics
from src.data.loader import load_processed
from src.portfolio.construction import build_weight_series, validate_weights
from src.portfolio.metrics import portfolio_return, portfolio_variance, portfolio_volatility
from src.portfolio.optimization import min_variance_portfolio
from src.reporting.export import save_figure
from src.reporting.plots import plot_portfolio_weights
from src.utils.config import load_settings
from src.utils.logger import get_logger
from src.utils.paths import OUTPUTS_FIGURES, OUTPUTS_TABLES

logger = get_logger("run_portfolio")


def main() -> None:
    settings = load_settings()
    allow_short: bool = settings.get("allow_short", False)

    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load processed returns ────────────────────────────────────────
    logger.info("Step 1/6 — Loading returns")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations")
    logger.info(f"  Tickers: {tickers}")

    # ── Step 2: Asset statistics summary ──────────────────────────────────────
    logger.info("Step 2/6 — Computing asset statistics")
    asset_stats = summarize_asset_statistics(returns, annualize=True)
    logger.info("\n" + asset_stats.round(4).to_string())

    # ── Step 3: Covariance matrix ─────────────────────────────────────────────
    logger.info("Step 3/6 — Computing covariance matrix")
    mean_returns = compute_mean_returns(returns, annualize=True)
    cov_matrix = compute_covariance_matrix(returns, annualize=True)

    # ── Step 4: Minimum variance optimization ─────────────────────────────────
    logger.info("Step 4/6 — Optimizing minimum variance portfolio")
    optimal_weights = min_variance_portfolio(mean_returns, cov_matrix)

    validate_weights(optimal_weights, tickers, allow_short=allow_short)

    weights_series = build_weight_series(optimal_weights, tickers)
    logger.info("  Optimal weights:")
    for ticker, w in weights_series.items():
        logger.info(f"    {ticker:8s} {w:.4f}  ({w * 100:.2f}%)")

    # ── Step 5: Portfolio metrics ─────────────────────────────────────────────
    logger.info("Step 5/6 — Computing portfolio metrics")
    port_ret = portfolio_return(optimal_weights, mean_returns.values)
    port_var = portfolio_variance(optimal_weights, cov_matrix.values)
    port_vol = portfolio_volatility(optimal_weights, cov_matrix.values)

    metrics = {
        "portfolio_return_ann": round(port_ret, 6),
        "portfolio_variance_ann": round(port_var, 6),
        "portfolio_volatility_ann": round(port_vol, 6),
    }
    logger.info(f"  Return (ann.)     : {port_ret:.4f}  ({port_ret * 100:.2f}%)")
    logger.info(f"  Volatility (ann.) : {port_vol:.4f}  ({port_vol * 100:.2f}%)")
    logger.info(f"  Variance (ann.)   : {port_var:.6f}")

    # ── Save tables ───────────────────────────────────────────────────────────
    weights_path = OUTPUTS_TABLES / "portfolio_weights.csv"
    weights_series.to_frame().to_csv(weights_path)
    logger.info(f"  Saved -> {weights_path.name}")

    metrics_path = OUTPUTS_TABLES / "portfolio_metrics.csv"
    pd.Series(metrics, name="value").to_csv(metrics_path, header=True)
    logger.info(f"  Saved -> {metrics_path.name}")

    # ── Step 6: Portfolio weights chart ───────────────────────────────────────
    logger.info("Step 6/6 — Generating portfolio weights chart")
    fig = plot_portfolio_weights(weights_series)
    save_figure(fig, OUTPUTS_FIGURES / "portfolio_weights.png", dpi=150)

    logger.info("Portfolio optimization complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
