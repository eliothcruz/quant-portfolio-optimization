# Portfolio Lab

A modular quantitative portfolio construction and backtesting system built in Python. Covers the full pipeline from raw price data to strategy comparison, including risk analytics, rolling optimization, and transaction cost modeling.

**Universe:** KO · GOOG · AAPL · MSFT · SPY &nbsp;|&nbsp; **Period:** 2019–2024 &nbsp;|&nbsp; **Frequency:** Daily

---

## Overview

This project implements the mean-variance framework (Markowitz, 1952) alongside risk-based alternatives, with a focus on practical deployment constraints: rolling estimation windows, bounded weights, and realistic transaction costs. The architecture is intentionally modular — each analytical layer is isolated into its own module and tested independently before integration.

The central finding is that **return estimates (μ) are the main source of instability** in mean-variance optimization. Risk-based strategies that avoid estimating expected returns — particularly Risk Parity — outperform on a risk-adjusted basis while requiring a fraction of the rebalancing activity.

---

## Pipeline

```
run_download.py          → fetch adjusted close prices via yfinance
run_prepare_data.py      → clean, align (inner join), compute simple returns
run_portfolio.py         → static optimization + efficient frontier
run_risk_report.py       → VaR, CVaR, correlation, drawdown reports
run_backtest.py          → rolling single-strategy backtest
run_strategy_comparison.py → multi-strategy comparison under identical conditions
```

Each script is self-contained and reads from `data/processed/` and writes to `outputs/`.

---

## Methodology

### Data

- Source: Yahoo Finance (`yfinance`), adjusted close prices
- Alignment: inner join across all tickers — no imputation, no forward-fill
- Returns: simple daily returns `r_t = (P_t / P_{t-1}) - 1`
- Annualization factor: 252 trading days

### Covariance Estimation

Two estimators are used and compared:

| Estimator | Description |
|---|---|
| Sample | Standard MLE: `Σ = (1 / T-1) Σ (r_t - μ)(r_t - μ)'` |
| Ledoit-Wolf | Analytic shrinkage toward scaled identity; minimizes expected Frobenius loss |

Shrinkage is particularly relevant in rolling windows where `T / n` is small.

### Optimization

All optimizations use `scipy.optimize.minimize` with the SLSQP method and a fully-invested, long-only feasible region. Weight bounds `[w_min, w_max]` are applied uniformly.

| Strategy | Objective | Inputs required |
|---|---|---|
| Min Variance | min `w'Σw` | Σ |
| Max Sharpe | max `(w'μ - r_f) / sqrt(w'Σw)` | μ, Σ |
| Max Sharpe + Shrinkage | Same, with Ledoit-Wolf Σ | μ, Σ_shrink |
| Risk Parity | min `Var(RC_i)` where `RC_i = w_i · (Σw)_i` | Σ |

### Backtesting

- **Rolling window:** 252 trading days used to estimate μ and Σ at each rebalance
- **No look-ahead bias:** parameters estimated on `[t - 252, t]`, applied from `t+1`
- **Rebalancing:** monthly (last trading day of each month)
- **Transaction costs:** `cost = c · Σ|w_t - w_{t-1}|`, deducted from the first day of each new period
- **Constraints:** `w_i ∈ [0.0, 0.40]`, `Σw_i = 1`

### Risk Metrics

Computed at the daily horizon on realized portfolio returns:

- **Historical VaR** at 95% confidence
- **Parametric VaR** (Gaussian assumption)
- **CVaR / TVaR** (Expected Shortfall): mean return in the VaR tail
- **Maximum Drawdown:** `min_t [(V_t - max_{s≤t} V_s) / max_{s≤t} V_s]`

---

## Results

Backtest period: **Feb 2020 – Dec 2024** (1 236 trading days, 59 rebalances).  
All strategies use identical parameters: monthly rebalancing, 252-day window, `max_weight = 0.40`, `transaction_cost = 0.001`.

| Strategy | Cum. Return | Ann. Return | Volatility | Sharpe | Max DD | Avg. Turnover |
|---|---|---|---|---|---|---|
| **Risk Parity** | **+116%** | 17.0% | 21.6% | **0.787** | -32.5% | **0.024** |
| Max Sharpe | +120% | 17.5% | 24.4% | 0.716 | -31.7% | 0.268 |
| Max Sharpe + Shrinkage | +120% | 17.4% | 24.4% | 0.714 | -31.5% | 0.265 |
| Min Variance | +80% | 12.8% | 19.8% | 0.644 | -33.9% | 0.048 |
| SPY (benchmark) | +97% | 14.8% | 21.1% | 0.703 | -33.7% | — |

> Sharpe ratio computed with `r_f = 0`. Turnover is the average one-way turnover per rebalance period.

---

## Key Insights

**Risk Parity dominates on a risk-adjusted basis.**  
It achieves the highest Sharpe ratio (0.787) despite having lower absolute returns than Max Sharpe, because it accepts less volatility and incurs negligible transaction costs. With a turnover of 0.024 per rebalance versus 0.268 for Max Sharpe, the gap would widen further under realistic cost assumptions.

**Return estimates are the primary source of instability.**  
Max Sharpe rebalances 11× more than Risk Parity because each rolling window produces a different μ̂, causing the optimizer to rotate aggressively into whichever asset has recently outperformed. This behavior is well-documented in the literature — sample means require decades of data to estimate reliably, making Max Sharpe highly sensitive to estimation error.

**Shrinkage helps but does not solve the μ problem.**  
Ledoit-Wolf shrinkage improves covariance conditioning and produces a marginally better drawdown (-31.5% vs -31.7%), but it does not stabilize weight allocation because the instability comes from μ̂, not Σ̂. The two strategies are nearly indistinguishable across all metrics.

**Min Variance underperforms the benchmark.**  
With a Sharpe of 0.644, Min Variance is the only strategy that fails to beat SPY (0.703). The portfolio concentrated in KO and SPY, which suffered correlated drawdowns in 2022 and recovered more slowly than technology names. Minimizing ex-ante variance does not guarantee reduced realized drawdown.

**Practical recommendation:** Risk Parity or a shrinkage-based estimator with explicit return views (e.g., Black-Litterman) are the most defensible choices for live deployment. Max Sharpe is suitable as a research baseline but requires significant regularization before it can be traded.

---

## Project Structure

```
portfolio_lab/
│
├── config/
│   ├── assets.yaml            # Ticker list
│   └── settings.yaml          # Global parameters (dates, confidence level)
│
├── data/
│   ├── raw/                   # Downloaded CSVs, one file per ticker (gitignored)
│   └── processed/             # Aligned prices and daily returns (gitignored)
│
├── outputs/
│   ├── tables/                # CSV outputs from each pipeline stage
│   └── figures/               # PNG charts
│
├── scripts/                   # Pipeline entry points
│   ├── run_download.py
│   ├── run_prepare_data.py
│   ├── run_portfolio.py
│   ├── run_risk_report.py
│   ├── run_backtest.py
│   └── run_strategy_comparison.py
│
└── src/
    ├── analytics/             # returns, statistics, covariance, performance
    ├── backtesting/           # rolling engine, multi-strategy runner
    ├── data/                  # downloader, loader, cleaner, validator
    ├── portfolio/             # metrics, constraints, optimization, construction
    ├── risk/                  # VaR, CVaR/TVaR, stress scenarios
    ├── reporting/             # tables, plots, export
    └── utils/                 # logger, config loader, centralized paths
```

---

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Python:** 3.11+  
**Core dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`, `yfinance`, `scikit-learn`, `PyYAML`

---

## How to Run

All commands should be executed from the project root (`portfolio_lab/`).

```bash
# 1. Download historical prices
python scripts/run_download.py

# 2. Clean, align, and compute returns
python scripts/run_prepare_data.py

# 3. Static optimization and efficient frontier
python scripts/run_portfolio.py

# 4. Risk report (VaR, CVaR, drawdown, correlation)
python scripts/run_risk_report.py

# 5. Rolling backtest (single strategy)
python scripts/run_backtest.py

# 6. Multi-strategy comparison
python scripts/run_strategy_comparison.py
```

To change the asset universe, edit `config/assets.yaml`. To adjust the backtest parameters (window, rebalancing frequency, max weight), edit the constants block at the top of the relevant script.

---

## Limitations

- **Estimation window:** A 252-day window is short for stable mean estimation. In practice, shrinkage or Bayesian priors are recommended when T/n is below 5.
- **Universe size:** With only 5 assets, diversification benefits are limited and results are sensitive to individual stock behavior.
- **Transaction cost model:** The flat-rate turnover model does not capture bid-ask spread, market impact, or slippage. For a small portfolio these are negligible; for institutional sizing they are not.
- **No taxes, dividends reinvestment, or leverage constraints.**
- **Stationarity assumption:** All estimators assume i.i.d. returns. Volatility clustering (GARCH effects) is not modeled.
- **Survivorship bias:** Tickers were selected in hindsight. The results are not a valid estimate of live performance.

---

## Potential Extensions

- **Black-Litterman model** — combine market equilibrium priors with analyst views to stabilize μ̂
- **Expanding universe** — add fixed income, international equity, or commodity ETFs
- **Dynamic risk targeting** — scale position sizes to maintain a constant ex-ante volatility
- **Walk-forward optimization** — nested cross-validation to select hyperparameters (window, max_weight) without look-ahead bias
- **GARCH-based covariance** — use DCC-GARCH for time-varying Σ estimation
