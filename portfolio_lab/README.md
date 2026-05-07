# Portfolio_lab

A modular quantitative portfolio research system built in Python. The pipeline covers the complete lifecycle from raw market data ingestion to multi-strategy performance attribution, including static mean-variance optimization, rolling backtests with transaction costs, risk decomposition, and a Black-Litterman Bayesian return estimation layer.

**Universe:** KO · GOOG · AAPL · MSFT · SPY &nbsp;|&nbsp; **Period:** 2019–2024 &nbsp;|&nbsp; **Frequency:** Daily &nbsp;|&nbsp; **Source:** Yahoo Finance (adjusted close)

---

## Central Findings

The dominant empirical result is straightforward: **expected return estimates are the primary source of instability in mean-variance portfolios**, and strategies that sidestep return estimation produce better risk-adjusted outcomes in this universe.

| Strategy | Cum. Return | Ann. Return | Volatility | Sharpe | Max DD | Avg. Turnover |
|---|---|---|---|---|---|---|
| **Risk Parity** | **+116%** | 17.0% | 21.6% | **0.787** | -32.5% | **0.024** |
| Max Sharpe | +120% | 17.5% | 24.4% | 0.716 | -31.7% | 0.268 |
| Max Sharpe + Shrinkage | +120% | 17.4% | 24.4% | 0.714 | -31.5% | 0.265 |
| Min Variance | +80% | 12.8% | 19.8% | 0.644 | -33.9% | 0.048 |
| SPY (benchmark) | +97% | 14.8% | 21.1% | 0.703 | -33.7% | — |

> Backtest period: Feb 2020 – Dec 2024 · 59 monthly rebalances · window = 252 days · max weight = 0.40 · transaction cost = 10 bps/unit of turnover · r_f = 0.

Risk Parity achieves the highest Sharpe ratio with a turnover 11× lower than Max Sharpe. The Black-Litterman layer (Phase 9) partially addresses the return estimation problem by anchoring μ̂ to a covariance-consistent equilibrium prior before applying investor views.

---

## Pipeline

```
run_download.py              → fetch adjusted close prices via yfinance
run_prepare_data.py          → clean, inner-join align, compute simple returns
run_portfolio.py             → static MVO: min variance + max Sharpe + efficient frontier
run_risk_report.py           → VaR, CVaR, correlation heatmap, distribution analysis
run_backtest.py              → rolling single-strategy backtest with SPY benchmark
run_strategy_comparison.py   → multi-strategy comparison under identical constraints
run_black_litterman.py       → BL equilibrium prior + views → posterior optimization
```

Each script is self-contained: reads from `data/processed/` and `outputs/tables/`, writes to `outputs/`. No script modifies upstream artifacts.

---

## Methodology Overview

### Data

- Adjusted close prices downloaded via `yfinance`; one CSV per ticker in `data/raw/`
- Alignment uses an **inner join** across all tickers — no forward-fill, no imputation
- Returns: simple daily `r_t = P_t / P_{t-1} − 1`; first row (NaN) dropped
- Annualization factor: 252 trading days

### Optimization

All solvers use `scipy.optimize.minimize` (SLSQP), long-only constraint `w_i ≥ 0`, and full investment `Σw_i = 1`.

| Strategy | Objective | Required inputs |
|---|---|---|
| Min Variance | min `w'Σw` | Σ |
| Max Sharpe | max `(w'μ − r_f) / √(w'Σw)` | μ, Σ |
| Max Sharpe + Shrinkage | Same with Ledoit-Wolf Σ | μ, Σ_LW |
| Risk Parity | min `Var(RC_i)`, `RC_i = w_i (Σw)_i` | Σ |
| Black-Litterman Max Sharpe | Max Sharpe with μ_BL replacing μ_hist | μ_BL, Σ |

### Risk Analytics

Computed at the daily horizon on realized portfolio returns:

- **Historical VaR (95%):** empirical 5th-percentile return
- **Parametric VaR (95%):** `μ + z_{0.05} · σ` under Gaussian assumption
- **CVaR / TVaR:** `E[r | r ≤ VaR]` — expected loss in the tail
- **Maximum Drawdown:** `min_t [(V_t − max_{s≤t} V_s) / max_{s≤t} V_s]`

### Backtesting

- **No look-ahead:** at each rebalance date `t`, estimation uses `[t − 252, t)` strictly
- **Rebalancing frequency:** monthly (last trading day of the month)
- **Transaction cost model:** `cost = c · Σ|w_new − w_old|`, applied at rebalance
- **Weight bounds:** `w_i ∈ [0.0, 0.40]` uniformly across all strategies in comparison

### Black-Litterman

Market equilibrium returns computed as `π = δ · Σ · w_market` (with equal weights as proxy for market caps). Investor views (absolute and relative) specified in `config/views.yaml`. Posterior:

```
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]
```

See [docs/theory.md](docs/theory.md) for complete derivation.

---

## Project Structure

```
portfolio_lab/
│
├── config/
│   ├── assets.yaml            # Ticker universe
│   ├── settings.yaml          # Dates, confidence level, price field
│   └── views.yaml             # Black-Litterman investor views and parameters
│
├── data/
│   ├── raw/                   # Per-ticker CSVs from yfinance (gitignored)
│   └── processed/             # Aligned price matrix and daily returns (gitignored)
│
├── docs/
│   ├── architecture.md        # Module structure, data flow, design decisions
│   ├── theory.md              # Mathematical foundations for all models
│   ├── results.md             # Detailed empirical results and interpretation
│   └── roadmap.md             # Extension roadmap and known limitations
│
├── outputs/
│   ├── tables/                # CSV outputs from each pipeline stage
│   └── figures/               # PNG charts (histograms, frontier, heatmaps, etc.)
│
├── scripts/                   # Pipeline entry points (one per phase)
│   ├── run_download.py
│   ├── run_prepare_data.py
│   ├── run_portfolio.py
│   ├── run_risk_report.py
│   ├── run_backtest.py
│   ├── run_strategy_comparison.py
│   └── run_black_litterman.py
│
└── src/
    ├── analytics/             # returns, statistics, covariance, performance
    ├── backtesting/           # rolling engine, multi-strategy orchestration
    ├── data/                  # downloader, loader, cleaner, validator
    ├── models/                # Black-Litterman model
    ├── portfolio/             # metrics, constraints, optimization, construction
    ├── risk/                  # VaR, CVaR/TVaR, stress scenarios
    ├── reporting/             # tables, plots, export utilities
    └── utils/                 # logger, config loader, centralized path registry
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Python:** 3.11+  
**Core dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`, `yfinance`, `scikit-learn`, `PyYAML`

---

## Usage

All commands from the project root (`portfolio_lab/`):

```bash
python scripts/run_download.py            # fetch prices
python scripts/run_prepare_data.py        # clean, align, compute returns
python scripts/run_portfolio.py           # static optimization + frontier
python scripts/run_risk_report.py         # risk decomposition report
python scripts/run_backtest.py            # rolling backtest (single strategy)
python scripts/run_strategy_comparison.py # multi-strategy comparison
python scripts/run_black_litterman.py     # Black-Litterman pipeline
```

Configuration:
- **Asset universe:** `config/assets.yaml`
- **Date range, confidence level:** `config/settings.yaml`
- **BL views and parameters:** `config/views.yaml`
- **Backtest parameters** (window, rebalancing, max weight, cost): constants block at the top of `run_backtest.py` and `run_strategy_comparison.py`

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module dependency graph, data flow, design decisions |
| [docs/theory.md](docs/theory.md) | Mathematical derivations: MVO, risk parity, shrinkage, Black-Litterman |
| [docs/results.md](docs/results.md) | Full backtest results, per-strategy analysis, key empirical findings |
| [docs/roadmap.md](docs/roadmap.md) | Known limitations, planned extensions, research directions |

---

## Limitations

- **Estimation window:** 252 days is insufficient for stable mean estimation when `n = 5`. Shrinkage and Black-Litterman partially mitigate this.
- **Universe size:** 5 assets limit genuine diversification; results are sensitive to individual security behavior, particularly AAPL.
- **Transaction cost model:** Flat-rate turnover model ignores bid-ask spread, market impact, and slippage.
- **Survivorship bias:** The ticker set was selected ex-post. Live performance would differ.
- **Equity-only:** The universe contains only US equities and one broad-market ETF. No fixed income, real assets, or international exposure.
- **Stationarity:** All estimators assume i.i.d. returns. Volatility clustering is not modeled.
- **Black-Litterman market weights:** Equal weights are used as a proxy for market-cap weights, which affects the equilibrium prior.
