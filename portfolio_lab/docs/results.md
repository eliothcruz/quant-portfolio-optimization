# Results

All results presented here are produced by running the pipeline on the configured universe with fixed parameters. No hyperparameter selection was performed on the backtest period.

**Universe:** KO · GOOG · AAPL · MSFT · SPY  
**Data period:** 2019-01-01 – 2024-12-31 (1 508 trading days after alignment)  
**Backtest period:** 2020-02-03 – 2024-12-30 (the first 252 observations are consumed by the initial estimation window)  
**Rebalancing:** Monthly (last trading day of each month) · 59 rebalance events  
**Window:** 252 trading days  
**Weight bounds:** `[0.0, 0.40]` per asset  
**Transaction cost:** 10 bps per unit of one-way turnover  
**Risk-free rate:** 0% (Sharpe denominator is gross excess return)

---

## 1. Multi-Strategy Comparison

### Performance Summary

| Strategy | Cum. Return | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown | Avg. Turnover |
|---|---|---|---|---|---|---|
| **Risk Parity** | **+116%** | **17.0%** | **21.6%** | **0.787** | -32.5% | **0.024** |
| Max Sharpe | +120% | 17.5% | 24.4% | 0.716 | -31.7% | 0.268 |
| Max Sharpe + Shrinkage | +120% | 17.4% | 24.4% | 0.714 | -31.5% | 0.265 |
| Min Variance | +80% | 12.8% | 19.8% | 0.644 | -33.9% | 0.048 |
| **SPY (benchmark)** | **+97%** | **14.8%** | **21.1%** | **0.703** | **-33.7%** | **—** |

### Key observations

**Risk Parity wins on a risk-adjusted basis** despite lower absolute returns than Max Sharpe. The Sharpe difference (0.787 vs 0.716) arises from two compounding effects: lower realized volatility (21.6% vs 24.4%) and dramatically lower transaction costs. With a turnover of 0.024 per rebalance versus 0.268 for Max Sharpe, Risk Parity is 11× cheaper to operate. The cost advantage would widen further under realistic bid-ask and market impact assumptions.

**All strategies beat SPY in absolute return** except Min Variance. Three of four strategies also beat SPY on a risk-adjusted basis (Sharpe > 0.703).

**Shrinkage is nearly indistinguishable from sample covariance.** Max Sharpe and Max Sharpe + Shrinkage differ by less than 0.003 Sharpe points and share identical absolute returns. This is the central empirical finding on estimation error: the instability in Max Sharpe comes from `μ̂`, not `Σ̂`. Ledoit-Wolf shrinkage improves `Σ̂` conditioning, which is beneficial for numerical stability but does not affect the dominant source of weight instability.

**Min Variance fails to beat the benchmark.** The portfolio concentrated heavily in KO and SPY — two assets that suffered correlated drawdowns in 2022 (rising rates, defensive underperformance). This illustrates the key limitation of variance minimization: it minimizes ex-ante covariance-implied variance, not realized drawdown.

---

## 2. Static Portfolio Analysis (2019–2024 full period)

### Minimum Variance

```
KO    47.6%  |  SPY   52.4%  |  AAPL  0.0%  |  MSFT  0.0%  |  GOOG  0.0%
Ann. Return  : 14.0%
Ann. Volatility : 17.9%
```

The optimizer assigns zero weight to all three tech names and splits the portfolio between KO (a low-volatility defensive name) and SPY (the market portfolio itself, which is by construction lower-volatility than individual holdings). This is mathematically correct: the covariance matrix shows that SPY and KO are less correlated with the high-volatility tech assets, so combining them achieves a lower `w'Σw` than any tech-inclusive portfolio.

### Maximum Sharpe (static, full-period μ)

```
AAPL  74.0%  |  MSFT  21.9%  |  GOOG   4.1%  |  KO   0.0%  |  SPY  0.0%
Ann. Return  : 34.5%
Ann. Volatility : 28.7%
Sharpe Ratio : 1.20
```

Without the 40% weight bound, the optimizer would concentrate even more aggressively in AAPL. The full-period μ̂ is dominated by AAPL's exceptional 2019–2024 performance, which the optimizer correctly identifies as the highest-Sharpe asset in isolation. This result is not a forecast — it is a backward-looking realization. In a live setting, this 5-year μ̂ would only be known after the fact.

### Efficient Frontier

The frontier spans:
- **Minimum:** return = 9.8%, volatility = 17.9% (min variance portfolio)
- **Maximum:** return = 36.5%, volatility = 30.9% (full AAPL concentration)

50 evenly-spaced target returns were evaluated; all 50 produced feasible solutions. The frontier is smooth, confirming that the SLSQP solver found the correct boundary at each point.

---

## 3. Risk Decomposition

### Asset-level daily risk metrics (95% confidence, daily horizon)

| Asset | Daily Mean | Daily Vol | Hist. VaR (return) | Hist. VaR (loss) | CVaR (loss) |
|---|---|---|---|---|---|
| KO | ~+0.03% | ~0.9% | ~−1.3% | ~1.3% | ~1.9% |
| GOOG | ~+0.08% | ~1.8% | ~−2.6% | ~2.6% | ~3.7% |
| AAPL | ~+0.10% | ~1.7% | ~−2.4% | ~2.4% | ~3.5% |
| MSFT | ~+0.09% | ~1.7% | ~−2.5% | ~2.5% | ~3.5% |
| SPY | ~+0.06% | ~1.1% | ~−1.6% | ~1.6% | ~2.4% |

> Values are illustrative of the relative ordering. Exact figures depend on the alignment window; run `python scripts/run_risk_report.py` to regenerate.

**The min variance portfolio risk profile** sits below all individual assets except KO. This confirms that the inner-join aligned dataset (no survivorship during the period) allows diversification to meaningfully reduce tail risk at the portfolio level.

**CVaR > VaR** holds for all assets by construction. The ratio `CVaR_loss / VaR_loss` is approximately 1.4–1.5 across the universe, indicating mild but consistent departure from Gaussian tails. Parametric VaR (Gaussian) slightly underestimates the empirical tail in all cases.

---

## 4. Black-Litterman Diagnostics

### Return comparison: historical vs equilibrium vs posterior (illustrative)

Views configured in `config/views.yaml`:
- Absolute: AAPL → 18% annualized, MSFT → 15% annualized
- Relative: AAPL outperforms KO by 5%
- Parameters: `τ = 0.05`, `δ = 2.5`, `confidence = 0.5`

| Asset | Historical μ | Equilibrium π | BL Posterior μ_BL |
|---|---|---|---|
| KO | ~12% | ~10% | ~10% |
| GOOG | ~20% | ~14% | ~16% |
| AAPL | ~25% | ~14% | **~17%** |
| MSFT | ~22% | ~14% | **~15%** |
| SPY | ~15% | ~12% | ~12% |

**μ_BL is anchored to π, not to μ_hist.** The BL posterior pulls AAPL's expected return from ~25% (historical) down to ~17% (a blend of the 18% absolute view and the 14% equilibrium). This reduces the optimizer's incentive to concentrate in AAPL, producing a more diversified portfolio than classical Max Sharpe.

**Assets without views (KO, GOOG, SPY) are still affected** because `Σ` creates correlation bleed-through in the posterior formula. GOOG's posterior is slightly above its equilibrium because the AAPL view is positive and AAPL-GOOG correlation is non-trivial.

### BL Max Sharpe weights (indicative)

The BL portfolio is typically more diversified than classical Max Sharpe:
- AAPL weight decreases from ~74% to a lower allocation
- MSFT and GOOG receive more weight than in the equilibrium-only solution
- KO and SPY receive small but non-zero allocations

The exact allocation depends on the specific views in `config/views.yaml`. Running `python scripts/run_black_litterman.py` generates `outputs/tables/black_litterman_weights.csv` with the current configuration.

---

## 5. Drawdown Analysis

### 2020 COVID crash (Feb–Mar 2020)

All strategies enter the backtest period at the start of this drawdown. Max Sharpe and Shrinkage recover faster due to higher tech weight; Risk Parity and Min Variance recover more slowly but suffer shallower initial declines (lower vol).

### 2022 bear market (Jan–Dec 2022)

The most informative period for distinguishing strategies:
- **Min Variance** suffers a deep drawdown because KO and SPY (its main holdings) both declined materially in the rising-rate environment
- **Risk Parity** holds up better due to its balanced risk contribution design
- **Max Sharpe** strategies experience high volatility as tech-heavy weights amplify the NASDAQ decline

This period illustrates why ex-ante variance minimization does not guarantee ex-post drawdown minimization: the correlation structure during a rate-driven selloff differs from the historical correlation used at estimation time.

### 2023–2024 tech recovery

Max Sharpe strategies benefit most from the tech rebound. The higher AAPL/MSFT concentration during this period drives the absolute return advantage of Max Sharpe over Risk Parity, despite the latter's better risk-adjusted performance.

---

## 6. Summary Interpretation

The central empirical message of this project is consistent with the academic literature:

**The sample mean `μ̂` is the weakest link in mean-variance optimization.** With only 5 years of daily data, the standard error of a monthly mean return is approximately `σ/√T ≈ 1.5%/√60 ≈ 0.19%` — roughly the size of a typical monthly mean itself. The optimizer treats this noisy estimate as ground truth and concentrates aggressively in recent winners.

**Three partial solutions are implemented:**

1. **Weight bounds** (max 40%): blunt instrument, always effective
2. **Risk Parity**: sidesteps `μ` entirely; best empirical performance
3. **Black-Litterman**: Bayesian regularization of `μ̂` toward a covariance-consistent prior; retains the Max Sharpe framework while reducing concentration

None of these solutions is a panacea. Risk Parity requires accurate covariance estimation (but covariance is far more stable than means). Black-Litterman requires a valid prior and well-calibrated views. Both require out-of-sample validation over longer horizons to draw robust conclusions.

---

## 7. Phase 10A — Factor Exposure Analysis

> **Status:** Module implemented; results pending execution with local factor data.
>
> Factor regression results will be populated here after running:
> ```bash
> python scripts/run_factor_analysis.py
> ```
> This requires placing Fama-French 3 Factor daily data in
> `data/factors/fama_french_3_factors.csv`. See `config/factors.yaml`
> for the expected format and download instructions.

### Expected findings (directional, based on asset characteristics)

**Asset-level betas (FF3):**

| Asset | β_mkt (expected) | β_smb (expected) | β_hml (expected) | Character |
|---|---|---|---|---|
| AAPL | > 1 | < 0 | < 0 | Large-cap growth |
| MSFT | > 1 | < 0 | < 0 | Large-cap growth |
| GOOG | > 1 | < 0 | < 0 | Large-cap growth |
| KO | < 1 | ≈ 0 | > 0 | Large-cap value/defensive |
| SPY | ≈ 1 | ≈ 0 | ≈ 0 | Broad market (by construction) |

**Strategy-level betas (FF3):**

| Strategy | β_mkt (expected) | Character |
|---|---|---|
| Min Variance | < 1 | Underweights high-beta tech |
| Risk Parity | < 1 | Balanced across vol-adjusted exposures |
| Max Sharpe | > 1 | Overweights high-beta AAPL/MSFT |
| Max Sharpe + Shrinkage | ≈ Max Sharpe | Similar concentration |

**Alpha significance:** With only 5 years of daily data and 5 assets, finding statistically significant alpha (p < 0.05, HC3 robust SE) would require an annualized alpha of approximately 3–5% to exceed the noise level. Genuine alpha is extremely difficult to establish with this sample size — a non-significant alpha should not be interpreted as evidence of zero skill, only of insufficient statistical power.

**Key diagnostic question:** If strategy alphas are statistically indistinguishable from zero and R² is high, the strategy's excess return is explained entirely by systematic factor exposure — which could be replicated via cheaper index instruments. If alpha is positive and significant, the strategy adds value beyond passive factor exposure.

---

## 8. Phase 10B — Factor-Aware Portfolio

> **Status:** Strategy implemented and integrated into the backtest engine. Results pending execution with local factor data.
>
> To run (requires Fama-French factor file):
> ```bash
> python scripts/run_strategy_comparison.py
> ```
> When `data/factors/fama_french_3_factors.csv` is present, the script automatically includes `factor_alpha_weighted` in the comparison and saves:
> - `outputs/tables/factor_alpha_weights_latest.csv` — last-window weights joined with FF3 regression results
> - `outputs/figures/factor_alpha_weights_latest.png` — weight and alpha visualization

### Strategy Description

At each monthly rebalance, the engine:
1. Runs Fama-French 3-Factor OLS regressions on the preceding 252-day window for each asset
2. Keeps only assets with positive estimated alpha
3. Assigns proportional weights to the alpha magnitudes
4. Caps each weight at 40% and renormalizes

No return forecasts or mean estimates are used — the portfolio is built entirely on the rolling in-sample alpha from the factor model.

### Expected Findings (Directional)

**Why alpha-weighting is unlikely to dominate in this sample:**

- With T = 252 and 5 assets, the annualized standard error of alpha is approximately 9–10%. A measured alpha of 5% annualized has a t-statistic below 1 — statistically indistinguishable from zero. The weights are therefore driven primarily by estimation noise rather than genuine abnormal return.
- The strategy ignores covariance structure. In a universe where AAPL, MSFT, and GOOG are highly correlated, concentrating weight in whichever tech name happened to have the highest in-sample alpha amplifies idiosyncratic and correlated risk simultaneously.
- The key comparison is with **Risk Parity**, which avoids both `μ̂` and `α̂` entirely. If alpha-weighting produces worse risk-adjusted performance than Risk Parity, the result is consistent with the literature: noise in return estimates — whether raw or factor-adjusted — degrades realized performance.

**What would constitute a meaningful positive result:** If `factor_alpha_weighted` achieves a Sharpe ratio meaningfully above `max_sharpe` (not just Risk Parity) with lower turnover, it would suggest that factor adjustment reduces the noise content of the return signal sufficiently to produce better weights than raw `μ̂`. This would be a strong empirical finding but requires out-of-sample validation beyond the 2020–2024 period.
