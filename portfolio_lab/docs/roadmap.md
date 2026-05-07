# Roadmap

## Current State

The pipeline is feature-complete for a first-generation equity research system. The following capabilities are fully implemented and validated:

| Capability | Module | Script |
|---|---|---|
| Data ingestion and cleaning | `src/data/` | `run_download.py`, `run_prepare_data.py` |
| Descriptive statistics | `src/analytics/statistics.py` | `run_portfolio.py` |
| Covariance estimation (sample + shrinkage) | `src/analytics/covariance.py` | — |
| MVO: min variance, max Sharpe | `src/portfolio/optimization.py` | `run_portfolio.py` |
| Efficient frontier | `src/portfolio/optimization.py` | `run_portfolio.py` |
| Risk parity | `src/portfolio/optimization.py` | `run_strategy_comparison.py` |
| Historical and parametric VaR / CVaR | `src/risk/` | `run_risk_report.py` |
| Stress scenarios (return shock, vol shock) | `src/risk/scenarios.py` | — |
| Rolling walk-forward backtest | `src/backtesting/engine.py` | `run_backtest.py` |
| Multi-strategy comparison | `src/backtesting/engine.py` | `run_strategy_comparison.py` |
| Black-Litterman posterior returns | `src/models/black_litterman.py` | `run_black_litterman.py` |
| Visualization suite (11 chart types) | `src/reporting/plots.py` | all scripts |

---

## Known Limitations

### Critical (affect validity of results)

**Survivorship bias.** The ticker set (KO, GOOG, AAPL, MSFT, SPY) was selected in 2025 knowing that all five had strong 2019–2024 performance. A live implementation would face uncertainty about which tickers to include, and some fraction of any broader universe would have been delisted or merged during the period.

**Short estimation history for means.** With a 252-day rolling window and 5 assets, `T/n ≈ 50`, which is sufficient for covariance estimation but far too short for stable mean estimation. Academic consensus requires `T/n ≈ 100–1000` for reliable `μ̂`. Black-Litterman and Risk Parity partially address this, but the fundamental data limitation remains.

**Equity-only universe.** The portfolio holds only US equities (and one US equity index). There is no fixed income, real asset, commodity, or international exposure. Reported diversification benefits and drawdown statistics reflect intra-equity correlation, not genuine multi-asset diversification.

**Equal-weight market proxy in Black-Litterman.** The equilibrium prior `π = δ·Σ·w_market` should use market-cap weights to be theoretically consistent. Equal weights alter the prior in ways that may not reflect actual market equilibrium, especially for a universe that mixes large-cap tech with a defensive consumer staple and a broad ETF.

### Moderate (affect robustness of conclusions)

**No nested cross-validation for hyperparameter selection.** The backtest window (252 days), max weight (40%), and transaction cost (10 bps) were chosen once and applied universally. These parameters were not tuned on the backtest period, but they were not validated out-of-sample either. In a production system, these would be selected via walk-forward validation on a separate holdout period.

**Flat-rate transaction cost model.** The model `cost = c × Σ|Δw_i|` does not capture bid-ask spread variation, market impact, or intraday execution quality. For a small portfolio this is a reasonable approximation; for institutional-scale allocation it is not.

**Monthly rebalancing only.** The engine supports daily, monthly, and quarterly frequencies, but results are only reported for monthly. Higher-frequency rebalancing would significantly affect turnover and cost figures, particularly for Max Sharpe.

**No regime detection.** The covariance matrix and mean vector are estimated with equal weighting across the entire rolling window. During regime changes (e.g., the 2022 rate shock), historical covariances estimated on 2021 data do not reflect the correlation structure of the new environment. Exponential weighting or regime-switching models are not implemented.

---

## Planned Extensions

### Near-term: improving return estimation

**Market-cap weights in Black-Litterman.**  
Currently `w_market` is approximated with equal weights. Integrating `yfinance.Ticker.info["marketCap"]` at download time would allow proper market-cap weighting of the prior, bringing the BL implementation closer to theoretical specification.

**BL integration into the rolling backtest.**  
`run_black_litterman.py` currently runs as a standalone static analysis. The natural next step is to integrate `black_litterman_max_sharpe_portfolio` into `backtest_portfolio_multi()` as a fifth strategy (`"black_litterman"`) with views refreshed at each rebalance date. This requires a mechanism for specifying time-varying views.

**Exponentially weighted covariance.**  
Replace the equal-weighted rolling window with `pandas.DataFrame.ewm(span=...).cov()`. Exponential weighting assigns more mass to recent observations, allowing faster adaptation to changing volatility regimes without shortening the effective sample size.

### Medium-term: expanding the universe

**Fixed income exposure.**  
Adding TLT (20Y Treasury ETF) or AGG (Aggregate Bond ETF) would introduce genuine flight-to-quality dynamics. The 2022 period in particular would look different: both equities and long-duration bonds declined simultaneously, which would have penalized a naive multi-asset risk parity allocation.

**International equity.**  
Adding EFA (developed ex-US) or EEM (emerging markets) introduces currency and political risk alongside genuine diversification. The inner-join alignment would drop dates where international markets are closed but US markets are open (or vice versa) — a meaningful design constraint.

**Factor ETFs.**  
Quality (QUAL), momentum (MTUM), and low-volatility (USMV) factor ETFs would allow testing of factor-based allocation alongside the current strategies.

### Medium-term: dynamic risk management

**Volatility targeting.**  
Scale the portfolio to target a constant ex-ante volatility (e.g., 15% annualized):
```
w_scaled = w_opt × (σ_target / σ_forecast)
```
This is particularly useful for Max Sharpe, which inherits the time-varying volatility of its concentrated equity holdings.

**Walk-forward hyperparameter selection.**  
For each parameter of interest (window, max_weight), run a grid search over a "development" backtest period (e.g., 2019–2021) and apply the selected values to a "test" period (2022–2024). This requires a careful nesting structure to avoid look-ahead bias in the parameter selection itself.

**Drawdown-based position sizing.**  
Reduce allocation proportionally when the realized drawdown exceeds a threshold (e.g., 20%). This is a simple form of risk control that limits left-tail exposure without requiring a full regime model.

### Long-term: advanced models

**DCC-GARCH covariance.**  
The Dynamic Conditional Correlation GARCH model allows the covariance matrix to vary day-by-day, reflecting volatility clustering. This is particularly relevant for risk metrics: historical VaR estimated on a calm period underestimates tail risk during a volatile regime.

**Robust optimization (Worst-Case MVO).**  
Instead of treating `μ̂` as exact, define an ellipsoidal uncertainty set around the estimate and solve the minimax problem:
```
max_{μ ∈ U(μ̂)} min_w  −(w'μ − r_f) / √(w'Σw)
```
This directly penalizes sensitivity to estimation error in the objective.

**Bayesian updating of Black-Litterman views.**  
Currently views are static (set once in `config/views.yaml`). A systematic extension would estimate view confidence from analyst forecast dispersion or option-implied volatility, updating `Ω` dynamically at each rebalance.

---

## What Not to Add

The following are deliberately out of scope:

- **Machine learning return prediction.** Return prediction via ML introduces a qualitatively different type of model risk and requires a much larger universe and longer history to validate. The existing framework is designed to be auditable and interpretable — ML-based forecasts would sacrifice that property.
- **Derivatives and leverage.** Options overlays, futures hedging, and leveraged positions require margin modeling, Greeks management, and a different regulatory framework. They are not compatible with the current long-only, fully-invested constraint structure.
- **Monte Carlo simulation.** Scenario generation via simulation is useful for fat-tail risk estimation but adds substantial implementation complexity for marginal benefit given that the historical dataset is long enough for empirical tail estimation.
- **Real-time data and live trading.** The pipeline is designed for research and backtesting. Extending it to live execution would require a broker API, order management, and latency-sensitive infrastructure that are outside the scope of a research system.
