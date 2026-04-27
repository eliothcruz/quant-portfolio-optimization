"""Compute risk metrics and generate risk reports for assets and portfolio.

Reads:
  config/settings.yaml                    (confidence_level)
  data/processed/returns.csv              (asset returns)
  outputs/tables/portfolio_weights.csv    (optimal weights from run_portfolio.py)

Writes:
  outputs/tables/asset_risk_table.csv
  outputs/tables/portfolio_risk_table.csv
  outputs/tables/portfolio_returns.csv
  outputs/tables/risk_comparison_summary.csv
  outputs/figures/portfolio_returns_histogram.png
  outputs/figures/correlation_matrix.png
  outputs/figures/returns_comparison.png
  outputs/figures/risk_return_scatter.png
  outputs/figures/cumulative_returns.png

Usage (from portfolio_lab/ directory):
    python scripts/run_risk_report.py

Pre-requisites:
    1. run_download.py
    2. run_prepare_data.py
    3. run_portfolio.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analytics.covariance import compute_correlation_matrix
from src.analytics.statistics import compute_mean_returns, compute_volatilities
from src.data.loader import load_processed
from src.portfolio.metrics import portfolio_returns
from src.reporting.export import save_figure, save_table
from src.reporting.plots import (
    plot_correlation_matrix,
    plot_cumulative_returns,
    plot_portfolio_returns_histogram,
    plot_returns_distribution_comparison,
    plot_risk_return_scatter,
)
from src.reporting.tables import (
    build_asset_risk_table,
    build_portfolio_risk_table,
    build_risk_comparison_table,
)
from src.risk.var import historical_var
from src.risk.tvar import historical_tvar
from src.utils.config import load_settings
from src.utils.logger import get_logger
from src.utils.paths import OUTPUTS_FIGURES, OUTPUTS_TABLES

logger = get_logger("run_risk_report")


def _load_weights(weights_path: Path) -> pd.Series:
    """Load portfolio weights from CSV and return as a Series indexed by ticker."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Portfolio weights not found: {weights_path}\n"
            "Run run_portfolio.py first to generate optimal weights."
        )
    df = pd.read_csv(weights_path, index_col=0)
    if "weight" not in df.columns:
        raise ValueError(
            f"Expected a 'weight' column in {weights_path}, "
            f"found: {list(df.columns)}"
        )
    weights = df["weight"]
    logger.info(f"Loaded {len(weights)} portfolio weights from {weights_path.name}")
    return weights


def main() -> None:
    # ── Configuration ─────────────────────────────────────────────────────────
    settings = load_settings()
    confidence_level: float = float(settings.get("confidence_level", 0.95))

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level in settings.yaml must be in (0, 1), "
            f"got {confidence_level}"
        )

    logger.info(f"Risk report | confidence_level={confidence_level:.0%}")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    logger.info("Step 1/11 — Loading data")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations | {tickers}")

    weights = _load_weights(OUTPUTS_TABLES / "portfolio_weights.csv")

    # ── Step 2: Portfolio return time series ───────────────────────────────────
    logger.info("Step 2/11 — Computing portfolio return series")
    port_ret = portfolio_returns(returns, weights)

    # ── Step 3: Asset risk table ───────────────────────────────────────────────
    logger.info("Step 3/11 — Building asset risk table")
    asset_table = build_asset_risk_table(returns, confidence_level)
    logger.info("\n" + asset_table.round(6).to_string())

    # ── Step 4: Portfolio risk table ───────────────────────────────────────────
    logger.info("Step 4/11 — Building portfolio risk table")
    port_table = build_portfolio_risk_table(port_ret, confidence_level)

    p = port_table.iloc[0]
    logger.info(
        f"  Portfolio daily mean     : {p['mean_return']:.6f}"
    )
    logger.info(
        f"  Portfolio daily vol      : {p['volatility']:.6f}"
    )
    logger.info(
        f"  Historical VaR  (return) : {p['historical_var_return']:.6f}  "
        f"(loss: {p['historical_var_loss']:.6f})"
    )
    logger.info(
        f"  Parametric VaR  (return) : {p['parametric_var_return']:.6f}  "
        f"(loss: {p['parametric_var_loss']:.6f})"
    )
    logger.info(
        f"  Historical TVaR (return) : {p['historical_tvar_return']:.6f}  "
        f"(loss: {p['historical_tvar_loss']:.6f})"
    )

    # ── Step 5: Risk comparison table ─────────────────────────────────────────
    logger.info("Step 5/11 — Building risk comparison table")
    comparison_table = build_risk_comparison_table(asset_table, port_table)
    logger.info("\n" + comparison_table.round(6).to_string())

    # ── Step 6: Save tables ────────────────────────────────────────────────────
    logger.info("Step 6/11 — Saving tables")
    save_table(asset_table, OUTPUTS_TABLES / "asset_risk_table.csv")
    save_table(port_table, OUTPUTS_TABLES / "portfolio_risk_table.csv")
    save_table(comparison_table, OUTPUTS_TABLES / "risk_comparison_summary.csv")
    save_table(
        port_ret.to_frame(),
        OUTPUTS_TABLES / "portfolio_returns.csv",
    )

    # ── Step 7: Generate and save histogram ───────────────────────────────────
    logger.info("Step 7/11 — Generating portfolio return histogram")
    hist_var = historical_var(port_ret, confidence_level)
    hist_tvar = historical_tvar(port_ret, confidence_level)

    fig = plot_portfolio_returns_histogram(
        port_ret,
        historical_var=hist_var,
        historical_tvar=hist_tvar,
        confidence_level=confidence_level,
    )
    save_figure(fig, OUTPUTS_FIGURES / "portfolio_returns_histogram.png", dpi=150)

    # ── Step 8: Correlation matrix heatmap ────────────────────────────────────
    logger.info("Step 8/11 — Generating correlation matrix heatmap")
    corr_matrix = compute_correlation_matrix(returns)
    fig = plot_correlation_matrix(corr_matrix)
    save_figure(fig, OUTPUTS_FIGURES / "correlation_matrix.png", dpi=150)

    # ── Step 9: Return distribution comparison ────────────────────────────────
    logger.info("Step 9/11 — Generating return distribution comparison")
    fig = plot_returns_distribution_comparison(returns, port_ret)
    save_figure(fig, OUTPUTS_FIGURES / "returns_comparison.png", dpi=150)

    # ── Step 10: Risk-return scatter ──────────────────────────────────────────
    logger.info("Step 10/11 — Generating risk-return scatter")
    ann_mean = compute_mean_returns(returns, annualize=True)
    ann_vol = compute_volatilities(returns, annualize=True)
    port_ann_ret = float((port_ret + 1).prod() ** (252 / len(port_ret)) - 1)
    port_ann_vol = float(port_ret.std() * (252 ** 0.5))
    fig = plot_risk_return_scatter(
        ann_mean,
        ann_vol,
        portfolio_point={"volatility": port_ann_vol, "return": port_ann_ret},
    )
    save_figure(fig, OUTPUTS_FIGURES / "risk_return_scatter.png", dpi=150)

    # ── Step 11: Cumulative returns ───────────────────────────────────────────
    logger.info("Step 11/11 — Generating cumulative returns chart")
    fig = plot_cumulative_returns(returns, port_ret)
    save_figure(fig, OUTPUTS_FIGURES / "cumulative_returns.png", dpi=150)

    logger.info("Risk report complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
