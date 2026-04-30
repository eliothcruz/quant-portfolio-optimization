"""Optimize portfolio weights and compute portfolio-level metrics.

Phases 1-4 (Steps 1-6): minimum variance pipeline.
Phase 6   (Steps 7-9): max Sharpe + efficient frontier.

Reads:   data/processed/returns.csv
Writes:  outputs/tables/portfolio_weights.csv
         outputs/tables/portfolio_metrics.csv
         outputs/tables/max_sharpe_portfolio.csv
         outputs/tables/efficient_frontier.csv
         outputs/figures/portfolio_weights.png
         outputs/figures/efficient_frontier.png

Usage (from portfolio_lab/ directory):
    python scripts/run_portfolio.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.analytics.covariance import compute_covariance_matrix
from src.analytics.statistics import compute_mean_returns, summarize_asset_statistics
from src.data.loader import load_processed
from src.portfolio.construction import build_weight_series, validate_weights
from src.portfolio.metrics import portfolio_return, portfolio_variance, portfolio_volatility
from src.portfolio.optimization import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
)
from src.reporting.export import save_figure
from src.reporting.plots import plot_efficient_frontier, plot_portfolio_weights
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
    logger.info("Step 1/9 — Loading returns")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations")
    logger.info(f"  Tickers: {tickers}")

    # ── Step 2: Asset statistics summary ──────────────────────────────────────
    logger.info("Step 2/9 — Computing asset statistics")
    asset_stats = summarize_asset_statistics(returns, annualize=True)
    logger.info("\n" + asset_stats.round(4).to_string())

    # ── Step 3: Covariance matrix ─────────────────────────────────────────────
    logger.info("Step 3/9 — Computing covariance matrix")
    mean_returns = compute_mean_returns(returns, annualize=True)
    cov_matrix = compute_covariance_matrix(returns, annualize=True)

    # ── Step 4: Minimum variance optimization ─────────────────────────────────
    logger.info("Step 4/9 — Optimizing minimum variance portfolio")
    optimal_weights = min_variance_portfolio(mean_returns, cov_matrix)

    validate_weights(optimal_weights, tickers, allow_short=allow_short)

    weights_series = build_weight_series(optimal_weights, tickers)
    logger.info("  Optimal weights:")
    for ticker, w in weights_series.items():
        logger.info(f"    {ticker:8s} {w:.4f}  ({w * 100:.2f}%)")

    # ── Step 5: Portfolio metrics ─────────────────────────────────────────────
    logger.info("Step 5/9 — Computing portfolio metrics")
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
    logger.info("Step 6/9 — Generating portfolio weights chart")
    fig = plot_portfolio_weights(weights_series)
    save_figure(fig, OUTPUTS_FIGURES / "portfolio_weights.png", dpi=150)

    # ── Step 7: Maximum Sharpe ratio portfolio ────────────────────────────────
    logger.info("Step 7/9 — Optimizing maximum Sharpe ratio portfolio")
    ms = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0)

    logger.info("  Max Sharpe weights:")
    for ticker, w in ms["weights"].items():
        logger.info(f"    {ticker:8s} {w:.4f}  ({w * 100:.2f}%)")
    logger.info(f"  Return (ann.)     : {ms['return']:.4f}  ({ms['return'] * 100:.2f}%)")
    logger.info(f"  Volatility (ann.) : {ms['volatility']:.4f}  ({ms['volatility'] * 100:.2f}%)")
    logger.info(f"  Sharpe ratio      : {ms['sharpe']:.4f}")

    ms_weights_path = OUTPUTS_TABLES / "max_sharpe_portfolio.csv"
    ms_summary = pd.concat([
        ms["weights"].to_frame(name="value"),
        pd.Series(
            {
                "portfolio_return_ann": ms["return"],
                "portfolio_volatility_ann": ms["volatility"],
                "sharpe_ratio": ms["sharpe"],
            },
            name="value",
        ).to_frame(),
    ])
    ms_summary.index.name = "metric"
    ms_summary.to_csv(ms_weights_path)
    logger.info(f"  Saved -> {ms_weights_path.name}")

    # ── Step 8: Efficient frontier ────────────────────────────────────────────
    logger.info("Step 8/9 — Computing efficient frontier")
    frontier_df = efficient_frontier(mean_returns, cov_matrix, n_points=50)
    logger.info(f"  Frontier points: {len(frontier_df)}")
    logger.info(
        f"  Return range : {frontier_df['return'].min():.4f} "
        f"-> {frontier_df['return'].max():.4f}"
    )
    logger.info(
        f"  Vol range    : {frontier_df['volatility'].min():.4f} "
        f"-> {frontier_df['volatility'].max():.4f}"
    )

    frontier_path = OUTPUTS_TABLES / "efficient_frontier.csv"
    frontier_df.to_csv(frontier_path, index=False)
    logger.info(f"  Saved -> {frontier_path.name}")

    # ── Step 9: Efficient frontier chart ─────────────────────────────────────
    logger.info("Step 9/9 — Generating efficient frontier chart")

    asset_volatilities = pd.Series(
        np.sqrt(np.diag(cov_matrix.values)),
        index=mean_returns.index,
    )
    min_var_point = {"return": port_ret, "volatility": port_vol}
    max_sharpe_point = {
        "return": ms["return"],
        "volatility": ms["volatility"],
        "sharpe": ms["sharpe"],
    }

    fig_frontier = plot_efficient_frontier(
        frontier_df,
        mean_returns,
        asset_volatilities,
        min_var_point,
        max_sharpe_point,
    )
    save_figure(fig_frontier, OUTPUTS_FIGURES / "efficient_frontier.png", dpi=150)

    logger.info("Portfolio optimization complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
