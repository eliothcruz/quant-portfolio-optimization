"""Black-Litterman posterior return estimation and portfolio optimization.

Combines equilibrium market returns (implied from covariance and equal weights)
with investor views defined in config/views.yaml to produce a posterior
expected return vector (mu_BL), then optimizes a Max Sharpe portfolio using
those posterior returns.

Reads:
  data/processed/returns.csv          (asset returns from run_prepare_data)
  config/views.yaml                   (investor views and BL parameters)

Writes:
  outputs/tables/black_litterman_equilibrium_returns.csv
  outputs/tables/black_litterman_posterior_returns.csv
  outputs/tables/black_litterman_weights.csv
  outputs/tables/black_litterman_summary.csv
  outputs/figures/black_litterman_returns_comparison.png
  outputs/figures/black_litterman_weights.png

Pre-requisites:
  1. run_download.py
  2. run_prepare_data.py

Note on market weights:
  Equal weights are used as a proxy for market-cap weights because market
  cap data is not available in this pipeline. This is a deliberate simplification.
  Equal-weight equilibrium returns still encode covariance structure; they
  simply imply the same prior belief for all assets before views are applied.

Usage (from portfolio_lab/ directory):
    python scripts/run_black_litterman.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.analytics.covariance import compute_covariance_matrix
from src.analytics.statistics import compute_mean_returns
from src.data.loader import load_processed
from src.models.black_litterman import (
    black_litterman_posterior_returns,
    build_absolute_views,
    build_relative_views,
    implied_equilibrium_returns,
)
from src.portfolio.optimization import black_litterman_max_sharpe_portfolio
from src.reporting.export import save_figure, save_table
from src.reporting.plots import (
    plot_black_litterman_returns_comparison,
    plot_black_litterman_weights,
)
from src.utils.logger import get_logger
from src.utils.paths import CONFIG_DIR, OUTPUTS_FIGURES, OUTPUTS_TABLES

logger = get_logger("run_black_litterman")


def _load_views(views_path: Path) -> dict:
    """Load and validate views.yaml."""
    if not views_path.exists():
        raise FileNotFoundError(
            f"Views config not found: {views_path}\n"
            "Create config/views.yaml before running this script."
        )
    with open(views_path, "r") as f:
        views = yaml.safe_load(f)
    logger.info(f"Views config loaded from {views_path.name}")
    return views


def _validate_views_assets(views: dict, tickers: list[str]) -> None:
    """Ensure all view assets exist in the universe."""
    abs_views = views.get("absolute_views") or {}
    for ticker in abs_views:
        if ticker not in tickers:
            raise ValueError(
                f"Absolute view on unknown ticker '{ticker}'. "
                f"Universe: {tickers}"
            )
    for rv in (views.get("relative_views") or []):
        for key in ("long", "short"):
            if rv[key] not in tickers:
                raise ValueError(
                    f"Relative view {key}='{rv[key]}' not in universe: {tickers}"
                )


def _combine_views(
    assets: list[str],
    abs_views: dict,
    rel_views: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine absolute and relative views into a single (P, Q) pair."""
    P_parts = []
    Q_parts = []

    if abs_views:
        P_abs, Q_abs = build_absolute_views(assets, abs_views)
        P_parts.append(P_abs)
        Q_parts.append(Q_abs)

    if rel_views:
        P_rel, Q_rel = build_relative_views(assets, rel_views)
        P_parts.append(P_rel)
        Q_parts.append(Q_rel)

    if not P_parts:
        raise ValueError(
            "No views found in views.yaml. "
            "Define at least one absolute_view or relative_view."
        )

    P = np.vstack(P_parts)
    Q = np.concatenate(Q_parts)
    logger.info(f"Combined views: P shape={P.shape}  Q shape={Q.shape}")
    return P, Q


def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load returns ───────────────────────────────────────────────────
    logger.info("Step 1/6 — Loading returns")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations | {tickers}")

    # ── Step 2: Compute covariance and historical mean returns ─────────────────
    logger.info("Step 2/6 — Computing covariance matrix and historical means")
    cov_matrix = compute_covariance_matrix(returns, annualize=True)
    historical_mu = compute_mean_returns(returns, annualize=True)
    logger.info("  Historical annualized mean returns:")
    for ticker, mu in historical_mu.items():
        logger.info(f"    {ticker:8s} {mu:.4f}  ({mu * 100:.2f}%)")

    # ── Step 3: Define market weights (equal-weight proxy) ────────────────────
    logger.info("Step 3/6 — Defining market weights (equal-weight proxy)")
    n = len(tickers)
    market_weights = pd.Series(
        np.ones(n) / n,
        index=tickers,
        name="market_weight",
    )
    logger.info(
        "  Using equal weights as proxy for market-cap weights "
        f"({1/n:.4f} per asset)."
    )

    # ── Step 4: Load views and compute BL returns ──────────────────────────────
    logger.info("Step 4/6 — Loading views and computing BL posterior returns")
    views = _load_views(CONFIG_DIR / "views.yaml")
    _validate_views_assets(views, tickers)

    bl_params = views.get("black_litterman", {})
    tau: float = float(bl_params.get("tau", 0.05))
    risk_aversion: float = float(bl_params.get("risk_aversion", 2.5))
    confidence: float = float(bl_params.get("confidence", 0.5))

    logger.info(
        f"  BL params: tau={tau}  risk_aversion={risk_aversion}  confidence={confidence}"
    )

    abs_views: dict = views.get("absolute_views") or {}
    rel_views: list = views.get("relative_views") or []
    P, Q = _combine_views(tickers, abs_views, rel_views)

    pi = implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion)
    mu_bl = black_litterman_posterior_returns(
        cov_matrix, market_weights, P, Q,
        tau=tau, risk_aversion=risk_aversion, confidence=confidence,
    )

    logger.info("  Return comparison (annualized):")
    logger.info(f"  {'Ticker':8s} {'Historical':>12s} {'Equilibrium':>12s} {'BL Posterior':>13s}")
    for ticker in tickers:
        logger.info(
            f"  {ticker:8s} {historical_mu[ticker]:>11.4f}  "
            f"{pi[ticker]:>11.4f}  {mu_bl[ticker]:>12.4f}"
        )

    # ── Step 5: BL Max Sharpe optimization ────────────────────────────────────
    logger.info("Step 5/6 — Optimizing BL Max Sharpe portfolio")
    bl_result = black_litterman_max_sharpe_portfolio(
        mu_bl, cov_matrix, risk_free_rate=0.0
    )

    logger.info("  BL Max Sharpe weights:")
    for ticker, w in bl_result["weights"].items():
        logger.info(f"    {ticker:8s} {w:.4f}  ({w * 100:.2f}%)")
    logger.info(
        f"  Return (ann.) : {bl_result['return']:.4f}  "
        f"({bl_result['return'] * 100:.2f}%)"
    )
    logger.info(
        f"  Volatility    : {bl_result['volatility']:.4f}  "
        f"({bl_result['volatility'] * 100:.2f}%)"
    )
    logger.info(f"  Sharpe ratio  : {bl_result['sharpe']:.4f}")

    # ── Step 6: Save outputs and generate figures ──────────────────────────────
    logger.info("Step 6/6 — Saving outputs and generating figures")

    # Equilibrium returns
    save_table(
        pi.to_frame("equilibrium_return"),
        OUTPUTS_TABLES / "black_litterman_equilibrium_returns.csv",
    )

    # Posterior returns: combine historical, pi, and mu_bl for reference
    returns_comparison = pd.DataFrame(
        {
            "historical_mean": historical_mu,
            "equilibrium_pi": pi,
            "bl_posterior": mu_bl,
        }
    )
    save_table(
        returns_comparison,
        OUTPUTS_TABLES / "black_litterman_posterior_returns.csv",
    )

    # BL optimal weights
    bl_weights_df = bl_result["weights"].to_frame("weight")
    save_table(bl_weights_df, OUTPUTS_TABLES / "black_litterman_weights.csv")

    # BL summary metrics
    bl_summary = pd.Series(
        {
            "bl_return_ann": bl_result["return"],
            "bl_volatility_ann": bl_result["volatility"],
            "bl_sharpe_ratio": bl_result["sharpe"],
            "tau": tau,
            "risk_aversion": risk_aversion,
            "confidence": confidence,
            "n_views": len(Q),
        },
        name="value",
    )
    save_table(
        bl_summary.to_frame(),
        OUTPUTS_TABLES / "black_litterman_summary.csv",
    )

    # Returns comparison chart
    fig_ret = plot_black_litterman_returns_comparison(pi, mu_bl, historical_mu)
    save_figure(
        fig_ret,
        OUTPUTS_FIGURES / "black_litterman_returns_comparison.png",
        dpi=150,
    )

    # BL weights chart
    fig_w = plot_black_litterman_weights(bl_result["weights"])
    save_figure(
        fig_w,
        OUTPUTS_FIGURES / "black_litterman_weights.png",
        dpi=150,
    )

    logger.info("Black-Litterman pipeline complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
