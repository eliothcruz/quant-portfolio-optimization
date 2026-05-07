# Architecture

## Design Philosophy

The project is organized around a strict separation of concerns: analytical layers do not import from each other horizontally, scripts orchestrate but do not compute, and all path management is centralized. Each module can be unit-tested in isolation using synthetic data without running the full pipeline.

Three invariants are enforced throughout:

1. **No look-ahead:** estimation always uses data strictly prior to the application date
2. **No imputation:** missing prices are dropped, never filled; alignment uses inner join only
3. **No circular imports:** the dependency graph is a DAG — `utils` → `analytics/data/risk` → `portfolio/reporting` → `scripts`

---

## Module Map

```
src/
├── utils/
│   ├── paths.py          Central path registry (ROOT derived from __file__)
│   ├── logger.py         Structured logger (StreamHandler, ISO timestamps)
│   └── config.py         YAML loaders: load_settings(), load_assets()
│
├── data/
│   ├── downloader.py     yfinance wrapper; handles MultiIndex column flattening
│   ├── loader.py         load_raw_asset(), load_all_raw_assets(), load_processed()
│   ├── cleaner.py        clean_asset_series(), align_to_common_period() [inner join]
│   └── validator.py      validate_all_raw(), validate_temporal_alignment()
│
├── analytics/
│   ├── returns.py        compute_simple_returns() [pct_change + dropna]
│   ├── statistics.py     mean, variance, volatility (annualized, ×252 or ×√252)
│   ├── covariance.py     sample covariance, Ledoit-Wolf shrinkage, correlation matrix
│   ├── performance.py    compute_performance_metrics(): cum_ret, ann_ret, Sharpe, MDD, turnover
│   └── diagnostics.py   Supplementary diagnostics
│
├── portfolio/
│   ├── constraints.py    long_only_bounds(), full_investment_constraint(), weight_bounds()
│   ├── construction.py   validate_weights(), build_weight_series()
│   ├── metrics.py        portfolio_return(), portfolio_variance(), portfolio_returns() [time series]
│   └── optimization.py   min_variance_portfolio(), max_sharpe_portfolio(),
│                         risk_parity_portfolio(), efficient_frontier(),
│                         min_variance_target_return(),
│                         black_litterman_max_sharpe_portfolio()
│
├── risk/
│   ├── var.py            historical_var(), parametric_var(), var_loss()
│   ├── tvar.py           historical_tvar(), tvar_loss()
│   └── scenarios.py      apply_return_shock(), apply_volatility_shock()
│
├── models/
│   └── black_litterman.py  implied_equilibrium_returns(), build_absolute_views(),
│                           build_relative_views(), build_omega(),
│                           black_litterman_posterior_returns()
│
├── backtesting/
│   └── engine.py         backtest_portfolio(), backtest_portfolio_multi()
│
└── reporting/
    ├── tables.py          build_asset_risk_table(), build_portfolio_risk_table(),
    │                      build_risk_comparison_table(), build_strategy_comparison_table()
    ├── plots.py           All visualization functions (matplotlib only, no seaborn)
    └── export.py          save_table(), save_figure()
```

---

## Data Flow

```
                        config/assets.yaml
                        config/settings.yaml
                              │
                              ▼
                      run_download.py
                              │
                    [data/raw/<ticker>.csv]
                              │
                              ▼
                    run_prepare_data.py
                              │
                    [data/processed/
                      prices_aligned.csv
                      returns.csv]
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
      run_portfolio.py  run_risk_report.py  run_black_litterman.py
              │               │                   │
    [outputs/tables/    [outputs/tables/    [outputs/tables/
     portfolio_*.csv     *_risk_*.csv        bl_*.csv
     frontier.csv]       portfolio_ret.csv]  outputs/figures/
    [outputs/figures/   [outputs/figures/    bl_*.png]
     weights.png         histogram.png
     frontier.png]       correlation.png
                         scatter.png
                         cumulative.png]
              │
              ▼
        run_backtest.py
              │
    [outputs/tables/
     backtest_*.csv]
    [outputs/figures/
     backtest_results.png]
              │
              ▼
    run_strategy_comparison.py
              │
    [outputs/tables/
     strategy_comparison.csv]
    [outputs/figures/
     strategy_comparison.png]
```

---

## Key Design Decisions

### Path centralization (`src/utils/paths.py`)

All file paths are computed once relative to `ROOT = Path(__file__).resolve().parents[2]`. Scripts never hardcode paths; they import `OUTPUTS_TABLES`, `OUTPUTS_FIGURES`, `DATA_PROCESSED`, etc. Moving the repository requires changing one line.

### `align_to_common_period` — inner join, no imputation

```python
series_list = [df.iloc[:, 0].rename(ticker) for ticker, df in clean_dict.items()]
aligned = pd.concat(series_list, axis=1, join="inner").sort_index()
```

`squeeze()` was rejected (ambiguous in pandas ≥ 2.0). `join="inner"` means that adding a ticker with gaps silently shrinks the analysis window — by design, since imputed prices would introduce phantom observations into the covariance estimator.

### Weight alignment by name in `portfolio_returns()`

```python
w_aligned = weights.reindex(returns_df.columns)
```

When loading weights from CSV, column order is not guaranteed. Aligning by ticker name prevents silent mismatch errors — a hard constraint given that any order-dependent bug would corrupt the entire backtest without raising an exception.

### No look-ahead in the backtest engine

```python
train = returns.iloc[pos - window : pos]   # strictly [t-window, t)
# weights applied from position pos onward
```

The `pos` index is the first day of the new period. Training data ends strictly before application, so no future information enters the estimation.

### Solver configuration

All optimizations use `method="SLSQP"`, `ftol=1e-12`, `maxiter=1000`. After convergence, tiny negative weights from floating-point residuals are clipped with `np.clip(result.x, 0.0, None)` and re-normalized. This is mathematically correct because the feasible region boundary `w_i = 0` is exact — residuals of order `1e-15` are not economically meaningful.

### Figure lifecycle

All plot functions return a `matplotlib.Figure` without calling `plt.show()`. `save_figure()` calls `plt.close(fig)` after saving. This prevents memory accumulation across the pipeline and ensures that figures do not interfere with each other's state in a sequential run.

---

## Output Inventory

### Tables (`outputs/tables/`)

| File | Produced by | Contents |
|---|---|---|
| `portfolio_weights.csv` | `run_portfolio.py` | Min-variance weights (ticker → weight) |
| `portfolio_metrics.csv` | `run_portfolio.py` | Ann. return, variance, volatility |
| `max_sharpe_portfolio.csv` | `run_portfolio.py` | Max Sharpe weights + metrics |
| `efficient_frontier.csv` | `run_portfolio.py` | (return, volatility) frontier points |
| `asset_risk_table.csv` | `run_risk_report.py` | Per-asset VaR, CVaR, vol (8 cols × n_assets) |
| `portfolio_risk_table.csv` | `run_risk_report.py` | Portfolio-level risk metrics |
| `portfolio_returns.csv` | `run_risk_report.py` | Daily portfolio return time series |
| `risk_comparison_summary.csv` | `run_risk_report.py` | Assets + portfolio side-by-side (5 cols) |
| `black_litterman_equilibrium_returns.csv` | `run_black_litterman.py` | π per asset |
| `black_litterman_posterior_returns.csv` | `run_black_litterman.py` | μ_hist, π, μ_BL comparison |
| `black_litterman_weights.csv` | `run_black_litterman.py` | BL Max Sharpe weights |
| `black_litterman_summary.csv` | `run_black_litterman.py` | BL metrics + model parameters |
| `strategy_comparison.csv` | `run_strategy_comparison.py` | Strategies × performance metrics |

### Figures (`outputs/figures/`)

| File | Chart type |
|---|---|
| `portfolio_weights.png` | Horizontal bar — min variance allocation |
| `efficient_frontier.png` | Risk-return scatter + frontier curve |
| `portfolio_returns_histogram.png` | Return distribution with VaR / CVaR markers |
| `correlation_matrix.png` | Heatmap (RdYlGn, annotated) |
| `returns_comparison.png` | Overlapping density histograms |
| `risk_return_scatter.png` | Assets + portfolio on vol-return axes |
| `cumulative_returns.png` | Cumulative return time series (% axis) |
| `backtest_results.png` | 3-panel: cumulative / drawdown / weight evolution |
| `strategy_comparison.png` | 3-panel: cumulative / drawdown / Sharpe bars |
| `black_litterman_returns_comparison.png` | Grouped bars: historical vs π vs μ_BL |
| `black_litterman_weights.png` | Horizontal bar — BL Max Sharpe allocation |
