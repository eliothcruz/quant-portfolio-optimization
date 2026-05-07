"""Factor exposure analysis: CAPM and Fama-French 3-Factor regressions.

Decomposes asset and strategy returns into systematic factor exposures
(beta_mkt, beta_smb, beta_hml) and abnormal return (alpha). Answers:
  - What fraction of return is explained by market, size, and value factors?
  - Do any assets or strategies generate statistically significant alpha?
  - Are strategies rewarded for systematic exposure or genuine skill?

Reads:
  data/processed/returns.csv                        (asset daily returns)
  data/factors/fama_french_3_factors.csv            (FF3 factors, local file)
  config/factors.yaml                               (factor file path, model, scale)
  outputs/tables/strategy_returns.csv               (optional: strategy returns)

Writes:
  outputs/tables/factor_exposures_assets_capm.csv
  outputs/tables/factor_exposures_assets_ff3.csv
  outputs/tables/factor_exposures_strategies_ff3.csv   (if strategy_returns.csv exists)
  outputs/figures/factor_betas_assets.png
  outputs/figures/factor_alpha_assets.png
  outputs/figures/factor_betas_strategies.png          (if strategy_returns.csv exists)
  outputs/figures/factor_alpha_strategies.png          (if strategy_returns.csv exists)

Pre-requisites:
  1. run_download.py
  2. run_prepare_data.py
  3. Place Fama-French 3 Factor daily data in data/factors/fama_french_3_factors.csv
     (see config/factors.yaml for format instructions)
  4. run_strategy_comparison.py (optional — needed for strategy-level analysis)

Usage (from portfolio_lab/ directory):
    python scripts/run_factor_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd

from src.data.loader import load_processed
from src.factors.loader import align_returns_with_factors, load_factor_data
from src.factors.metrics import (
    run_factor_analysis_for_assets,
    run_factor_analysis_for_strategies,
)
from src.reporting.export import save_figure, save_table
from src.reporting.plots import plot_alpha_comparison, plot_factor_betas
from src.reporting.tables import build_factor_summary_table
from src.utils.logger import get_logger
from src.utils.paths import CONFIG_DIR, OUTPUTS_FIGURES, OUTPUTS_TABLES

logger = get_logger("run_factor_analysis")


def _load_factor_config() -> dict:
    config_path = CONFIG_DIR / "factors.yaml"
    if not config_path.exists():
        logger.warning(
            "config/factors.yaml not found — using default settings. "
            "Create config/factors.yaml for full control."
        )
        return {
            "factor_file": "data/factors/fama_french_3_factors.csv",
            "default_model": "ff3",
            "min_observations": 252,
            "convert_percent_to_decimal": True,
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _try_load_strategy_returns() -> pd.DataFrame | None:
    """Load strategy_returns.csv if it exists; return None with a warning otherwise."""
    path = OUTPUTS_TABLES / "strategy_returns.csv"
    if not path.exists():
        logger.warning(
            "outputs/tables/strategy_returns.csv not found. "
            "Strategy-level factor analysis will be skipped. "
            "Run run_strategy_comparison.py first to generate this file."
        )
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    logger.info(
        f"Strategy returns loaded: {len(df)} observations  "
        f"strategies={list(df.columns)}"
    )
    return df


def main() -> None:
    OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load configuration ─────────────────────────────────────────────
    logger.info("Step 1/5 — Loading factor configuration")
    config = _load_factor_config()
    factor_path = Path(config["factor_file"])
    model: str = config.get("default_model", "ff3")
    convert: bool = bool(config.get("convert_percent_to_decimal", True))

    logger.info(f"  Factor file : {factor_path}")
    logger.info(f"  Model       : {model.upper()}")
    logger.info(f"  Convert %   : {convert}")

    # ── Step 2: Load data ──────────────────────────────────────────────────────
    logger.info("Step 2/5 — Loading returns and factor data")
    returns = load_processed("returns.csv")
    tickers = list(returns.columns)
    logger.info(f"  {len(tickers)} assets | {len(returns)} observations")

    factors = load_factor_data(factor_path, convert_percent_to_decimal=convert)

    # ── Step 3: Align ──────────────────────────────────────────────────────────
    logger.info("Step 3/5 — Aligning returns with factors")
    aligned_returns, aligned_factors = align_returns_with_factors(returns, factors)

    # ── Step 4: Asset-level factor analysis ───────────────────────────────────
    logger.info("Step 4/5 — Running factor regressions for assets")

    logger.info("  CAPM regressions:")
    capm_results = run_factor_analysis_for_assets(
        aligned_returns, aligned_factors, model="capm"
    )
    capm_summary = build_factor_summary_table(capm_results)
    save_table(capm_summary, OUTPUTS_TABLES / "factor_exposures_assets_capm.csv")

    logger.info("  FF3 regressions:")
    ff3_results = run_factor_analysis_for_assets(
        aligned_returns, aligned_factors, model="ff3"
    )
    ff3_summary = build_factor_summary_table(ff3_results)
    save_table(ff3_summary, OUTPUTS_TABLES / "factor_exposures_assets_ff3.csv")

    logger.info("  FF3 results:")
    logger.info(
        f"  {'Asset':8s} {'Alpha':>8s} {'β_mkt':>7s} {'β_smb':>7s} "
        f"{'β_hml':>7s} {'R²':>6s} {'p_α':>6s}"
    )
    for asset, row in ff3_summary.iterrows():
        logger.info(
            f"  {asset:8s} "
            f"{row['alpha']:>8.4f} "
            f"{row['beta_mkt']:>7.4f} "
            f"{row.get('beta_smb', float('nan')):>7.4f} "
            f"{row.get('beta_hml', float('nan')):>7.4f} "
            f"{row['r_squared']:>6.4f} "
            f"{row['p_alpha']:>6.4f}"
        )

    fig_betas = plot_factor_betas(
        ff3_summary,
        title="Factor Loadings by Asset — Fama-French 3-Factor",
    )
    save_figure(fig_betas, OUTPUTS_FIGURES / "factor_betas_assets.png", dpi=150)

    fig_alpha = plot_alpha_comparison(
        ff3_summary,
        title="Annualized Alpha by Asset — Fama-French 3-Factor",
    )
    save_figure(fig_alpha, OUTPUTS_FIGURES / "factor_alpha_assets.png", dpi=150)

    # ── Step 5: Strategy-level factor analysis (optional) ─────────────────────
    logger.info("Step 5/5 — Strategy-level factor analysis")
    strategy_returns = _try_load_strategy_returns()

    if strategy_returns is not None:
        aligned_strat, aligned_factors_strat = align_returns_with_factors(
            strategy_returns, factors
        )
        strat_ff3 = run_factor_analysis_for_strategies(
            aligned_strat, aligned_factors_strat, model="ff3"
        )
        strat_summary = build_factor_summary_table(strat_ff3)
        save_table(
            strat_summary,
            OUTPUTS_TABLES / "factor_exposures_strategies_ff3.csv",
        )

        fig_strat_betas = plot_factor_betas(
            strat_summary,
            title="Factor Loadings by Strategy — Fama-French 3-Factor",
        )
        save_figure(
            fig_strat_betas,
            OUTPUTS_FIGURES / "factor_betas_strategies.png",
            dpi=150,
        )

        fig_strat_alpha = plot_alpha_comparison(
            strat_summary,
            title="Annualized Alpha by Strategy — Fama-French 3-Factor",
        )
        save_figure(
            fig_strat_alpha,
            OUTPUTS_FIGURES / "factor_alpha_strategies.png",
            dpi=150,
        )
    else:
        logger.info(
            "  Strategy analysis skipped (strategy_returns.csv not found). "
            "Run run_strategy_comparison.py to generate it."
        )

    logger.info("Factor analysis complete.")
    logger.info(f"  Tables  -> {OUTPUTS_TABLES}")
    logger.info(f"  Figures -> {OUTPUTS_FIGURES}")


if __name__ == "__main__":
    main()
