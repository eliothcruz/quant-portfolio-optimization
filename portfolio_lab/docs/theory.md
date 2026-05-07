# Theory

Mathematical foundations for the models implemented in `portfolio_lab`. The treatment assumes familiarity with linear algebra and basic probability. Derivations are kept concise but complete enough to verify the implementation.

---

## 1. Returns

Daily simple returns are used throughout:

```
r_t = P_t / P_{t-1} − 1
```

**Why simple and not log?** Log returns are additive over time (`r_{t:T} = Σ r_t`), but portfolio returns are additive across assets at a fixed point in time (`r_p = Σ w_i r_i`). Since cross-sectional aggregation is the primary operation in this project, simple returns are the natural choice.

**Annualization:**
- Mean return: `μ_ann = μ_daily × 252`
- Volatility: `σ_ann = σ_daily × √252`
- Variance: `σ²_ann = σ²_daily × 252`

The factor 252 is the conventional number of US equity trading days per year.

---

## 2. Mean-Variance Optimization (Markowitz, 1952)

### Problem statement

A portfolio is a vector `w ∈ ℝⁿ` such that `1'w = 1`. Given a return vector `μ` and covariance matrix `Σ`, the portfolio's expected return and variance are:

```
μ_p = w'μ
σ²_p = w'Σw
```

The **efficient frontier** is the set of portfolios that minimize variance for a given level of expected return:

```
min_w   w'Σw
s.t.    w'μ = target_return
        1'w = 1
        w ≥ 0   (long-only)
```

### Minimum variance portfolio

Special case of the frontier problem where no return target is imposed:

```
min_w   w'Σw
s.t.    1'w = 1
        w ≥ 0
```

The analytic solution without the non-negativity constraint is `w* = Σ⁻¹1 / (1'Σ⁻¹1)`. With long-only constraints, a numerical solver (SLSQP) is required.

### Maximum Sharpe ratio portfolio

The Sharpe ratio measures excess return per unit of risk:

```
SR = (μ_p − r_f) / σ_p = (w'μ − r_f) / √(w'Σw)
```

Maximizing SR is equivalent to finding the portfolio on the Capital Market Line tangent to the efficient frontier. Implemented as minimization of `−SR`:

```
min_w   −(w'μ − r_f) / √(w'Σw)
s.t.    1'w = 1,  w ≥ 0
```

The solver objective evaluates to 0 when `w'Σw ≤ 0` (degenerate case), preventing division by zero.

### Known pathologies

The classical MVO solution is highly sensitive to `μ`. Chopra & Ziemba (1993) showed that errors in `μ` affect the optimal weights roughly 10× more than equivalent errors in `Σ`. This manifests empirically as extreme concentration in recent winners: in the 2019–2024 period, an unconstrained Max Sharpe solution assigns nearly all weight to AAPL and MSFT because their trailing means are highest.

The `max_weight = 0.40` constraint is the primary practical remedy used in the backtest.

---

## 3. Covariance Estimation

### Sample estimator

The standard unbiased estimator:

```
Σ̂ = 1/(T−1) · X'X     where X is the T×n demeaned return matrix
```

The sample estimator is consistent but has poor finite-sample properties when `T/n` is small. For a rolling window of 252 days and 5 assets, `T/n = 50.4`, which is adequate for Σ but inadequate for μ.

### Ledoit-Wolf shrinkage

The Ledoit-Wolf (2004) estimator shrinks the sample covariance toward a structured target (scaled identity in the `scikit-learn` implementation):

```
Σ̂_LW = (1 − α) · Σ̂_sample + α · μ_F · I
```

where `α ∈ [0, 1]` is the analytically optimal shrinkage intensity and `μ_F` is the mean eigenvalue. The shrinkage intensity is derived by minimizing the expected Frobenius loss under a Gaussian assumption — no cross-validation is required.

**Effect:** Shrinkage pulls extreme eigenvalues toward the center, reducing condition number and preventing a few dominant factors from capturing all the variance. The result is a more stable inverse `Σ⁻¹`, which directly improves the numerical behavior of the optimizer.

**Empirical observation:** In this project, Max Sharpe with Ledoit-Wolf shrinkage produces virtually identical performance to the sample version (+120% vs +120%, SR 0.714 vs 0.716). This confirms that the instability source is `μ̂`, not `Σ̂` — shrinkage cannot fix a problem that lives in a different object.

---

## 4. Risk Parity

### Risk contribution decomposition

Define the marginal risk contribution of asset `i` as:

```
MRC_i = ∂σ_p/∂w_i = (Σw)_i / σ_p
```

The total risk contribution is:

```
RC_i = w_i · MRC_i = w_i · (Σw)_i / σ_p
```

Note that `Σ RC_i = σ_p` (Euler's homogeneous function theorem). Risk Parity requires `RC_i = σ_p / n` for all `i`.

### Objective function

Rather than imposing equality constraints on each `RC_i` (which leads to a non-convex program), the implemented objective minimizes the variance of the risk contributions:

```
min_w   Σᵢ (RC_i − RC̄)²
s.t.    1'w = 1,  w ≥ 0
```

where `RC_i = w_i · (Σw)_i` (without dividing by `σ_p`, which is equivalent since `σ_p` is constant w.r.t. the objective). This is a smooth, differentiable objective that SLSQP handles well.

### Why Risk Parity works empirically

Risk Parity makes no return estimate. This is its key advantage: by ignoring `μ̂`, it eliminates the dominant source of estimation error. The portfolio is constructed to be "balanced" in the sense that no single asset dominates the risk budget — diversification is enforced directly rather than emerging as a byproduct of optimization.

In this universe, Risk Parity tends to hold less tech (AAPL, MSFT are high-volatility and highly correlated) and more defensive names (KO) relative to Max Sharpe. This reduces total volatility, and since the numerator of the Sharpe ratio is similar across strategies on a long enough horizon, the lower denominator produces a higher ratio.

---

## 5. Value at Risk and Expected Shortfall

### Historical VaR

The empirical quantile of the realized return distribution:

```
VaR_α = Quantile(r, 1 − α)
```

For `α = 0.95`, this is the 5th percentile of the return series. The interpretation: with 95% probability, the daily loss will not exceed `|VaR_0.95|`.

**Sign convention used throughout:** `VaR_return < 0` (in loss territory); `VaR_loss = −VaR_return > 0` (positive magnitude). This distinction is enforced by naming in all output tables.

### Parametric VaR

Assumes returns are normally distributed:

```
VaR_α^param = μ + z_α · σ     where z_α = Φ⁻¹(1 − α)
```

For `α = 0.95`, `z_{0.05} = −1.6449`. The Gaussian assumption understates tail risk for financial returns (which exhibit excess kurtosis), so parametric VaR tends to be less conservative than historical VaR.

### CVaR / TVaR (Expected Shortfall)

The average return conditional on being in the tail:

```
CVaR_α = E[r | r ≤ VaR_α]
```

Computed as the mean of all realized returns at or below the VaR threshold. CVaR is a **coherent** risk measure (Artzner et al., 1999) — it satisfies subadditivity, meaning the risk of a combined portfolio is never greater than the sum of individual risks. VaR does not satisfy this property.

The invariant `CVaR_loss ≥ VaR_loss` holds by construction since CVaR averages returns that are weakly below the VaR threshold.

---

## 6. Black-Litterman Model

The Black-Litterman (BL) model (Black & Litterman, 1992) provides a Bayesian framework for blending market equilibrium return estimates with subjective investor views.

### Step 1: Implied equilibrium returns

Starting from the CAPM market equilibrium condition, the expected return vector that "explains" the observed market portfolio weights `w_market` is:

```
π = δ · Σ · w_market
```

where `δ` is the risk aversion coefficient (typically 2–4 for equity portfolios). This is derived by inverting the first-order condition of the market portfolio's mean-variance problem: at equilibrium, `μ_market = δ · Σ · w_market`.

**Intuition:** `π` encodes how much return each asset must offer to justify its weight in the market portfolio given the current covariance structure. An asset that is highly correlated with everything else requires a higher implied return to be worth holding.

**Proxy used in this implementation:** Market-cap weights are approximated by equal weights, since market capitalization data is not available in the pipeline. This is a known simplification — see [docs/roadmap.md](roadmap.md).

### Step 2: Investor views

Views are expressed as linear combinations of asset returns:

```
P · r = Q + ε,   ε ~ N(0, Ω)
```

- `P` is a `(k × n)` view-picking matrix (k views, n assets)
- `Q` is a `(k,)` vector of expected view returns
- `Ω` is a `(k × k)` diagonal uncertainty matrix

**Absolute view** on asset `i` (`E[r_i] = q`): row of P is `e_i` (standard basis vector).  
**Relative view** (`E[r_i − r_j] = q`): row of P has `+1` at `i` and `−1` at `j`.

### Step 3: View uncertainty (Ω)

The diagonal elements of `Ω` quantify how uncertain the investor is about each view:

```
Ω_kk = (P · τΣ · P')_kk / confidence
```

The numerator (`τ × P_k · Σ · P_k'`) is the view variance implied by the prior covariance structure. Dividing by `confidence ∈ (0, 1]` scales up uncertainty when the investor has less than full confidence. At `confidence = 0.5`, uncertainty doubles relative to the proportional Ω standard.

### Step 4: Posterior return vector

The BL posterior combines the prior `π` with the views `(P, Q)` via Bayes' theorem under Gaussian assumptions:

```
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]
```

This is a precision-weighted average of the prior and the views:
- The precision of the prior is `(τΣ)⁻¹`
- The precision of the views is `P'Ω⁻¹P`
- The posterior is the inverse of total precision times the precision-weighted mean

**Key properties:**
- If no views are expressed (`P = ∅`), then `μ_BL = π` (prior dominates)
- If views have zero uncertainty (`Ω → 0`), then `μ_BL → Ω⁻¹Q` (views dominate)
- Assets not directly covered by any view are still affected through `Σ` (correlation bleed-through)
- `τ` controls how tightly the prior is held; smaller `τ` → stronger prior

### Step 5: Portfolio optimization

`μ_BL` is passed directly to `max_sharpe_portfolio()` in place of `μ_hist`. The covariance matrix `Σ` is unchanged — BL is a return estimation technique, not a covariance technique.

The key behavioral difference: `μ_BL` is anchored to `π`, which reflects covariance structure rather than recent return history. This produces portfolio weights that are less sensitive to whichever asset happened to outperform in the trailing window.

---

## 7. Rolling Backtest Mechanics

The backtest engine implements a **walk-forward** procedure with no nested validation:

```
for each rebalance date t:
    train  = returns[t − window : t]      # strictly prior to t
    weights = strategy(train)              # solve on training data
    apply weights from t to t_next − 1   # hold until next rebalance
    portfolio_return = (returns[t:t_next] × weights).sum(axis=1)
    transaction_cost = c × Σ|w_new − w_prev|  (deducted at t)
```

**No look-ahead violation** is possible because `train` ends at `t − 1` (Python slicing `[t-w:t]` excludes index `t`).

**Fallback behavior:** If the optimizer fails to converge for a given training window (rare with SLSQP and a well-conditioned Σ), the engine falls back to equal weights and logs a warning. This prevents NaN propagation in the return series.

**Benchmark construction:** SPY returns are loaded from the same `returns.csv` and aligned to the backtest period by index intersection. The benchmark is buy-and-hold (no transaction costs applied).

---

## 8. Factor Models

Factor models decompose asset returns into a component explained by systematic risk factors and an unexplained residual (alpha). The core question they answer is: *does a strategy generate returns because it takes systematic exposures that any investor could replicate cheaply, or because it captures genuine skill?*

### CAPM (Single-Factor Model)

The Capital Asset Pricing Model (Sharpe, 1964; Lintner, 1965) states that the expected excess return of an asset is proportional to its exposure to the market portfolio:

```
r_i − RF = α + β_mkt · MKT_RF + ε
```

Where:
- `r_i − RF` — asset excess return (asset return minus risk-free rate)
- `MKT_RF` — market excess return (market return minus risk-free rate)
- `β_mkt` — market beta: sensitivity of the asset to market movements
- `α` — intercept; the return unexplained by the market (Jensen's alpha)
- `ε` — idiosyncratic residual

**Beta interpretation:**
- `β > 1`: asset amplifies market moves (high-beta, e.g., growth tech)
- `β = 1`: asset moves in line with the market
- `β < 1`: asset is defensive (dampens market moves, e.g., utilities, consumer staples)
- `β < 0`: asset moves opposite to the market (rare in long-only equity portfolios)

In this universe, AAPL and MSFT are expected to have `β > 1` (tech amplifies market risk), while KO is expected to have `β < 1` (defensive). SPY, by construction, should have `β ≈ 1`.

### Fama-French 3-Factor Model (FF3)

Fama and French (1993) documented that CAPM alpha is systematically related to two additional characteristics: firm size and book-to-market ratio. The FF3 model adds two factors:

```
r_i − RF = α + β_mkt · MKT_RF + β_smb · SMB + β_hml · HML + ε
```

Where:
- `SMB` (Small Minus Big) — return spread between small-cap and large-cap stocks. Positive `β_smb` indicates the asset behaves like a small-cap stock (higher return in environments where small caps outperform).
- `HML` (High Minus Low) — return spread between high book-to-market (value) and low book-to-market (growth) stocks. Positive `β_hml` indicates value-like behavior; negative indicates growth-like behavior.

**Beta_smb interpretation** for this universe:
- AAPL, MSFT, GOOG are large-cap growth stocks → expected `β_smb < 0` (they behave like the "Big" side of SMB)
- KO is a large-cap defensive → expected `β_smb ≈ 0` or slightly negative

**Beta_hml interpretation**:
- AAPL, MSFT, GOOG are growth stocks (low book-to-market) → expected `β_hml < 0`
- KO has more value characteristics → expected `β_hml > 0`

### Alpha (Jensen's Alpha)

The intercept `α` in the OLS regression represents the average return of the asset (or strategy) that is **not explained by the factor exposures**. A positive and statistically significant alpha indicates outperformance relative to what the factor loadings predict.

**Statistical test:** The null hypothesis is `H₀: α = 0`. Rejection at the 5% level (p-value < 0.05, using HC3 heteroskedasticity-consistent standard errors) provides evidence of genuine abnormal return. Statistical significance requires both:
- A t-statistic of magnitude > ~2 in absolute value
- A sufficiently long sample (≥ 252 observations)

**Annualization:** The daily alpha from the regression is multiplied by 252 to produce an annualized figure. This is approximate (the compounding effect is ignored) but standard in the literature.

**Caution:** A significant alpha over a backtest period does not imply future alpha. With only 5 assets and 5 years of data, the multiple testing problem is severe — one of five assets would be expected to show `p < 0.05` by chance alone even if all true alphas are zero.

### R² (Coefficient of Determination)

R² measures the fraction of return variance explained by the factor model:

```
R² = 1 − Var(ε) / Var(r_i − RF)
```

- `R² ≈ 0.9` for SPY: almost entirely explained by market beta (by construction)
- `R² ≈ 0.7–0.8` for large-cap stocks like AAPL, MSFT: mostly systematic
- `R² ≈ 0.5` for KO: more idiosyncratic variation

For strategies: Risk Parity and Min Variance typically have higher R² than Max Sharpe because they hold more diversified portfolios with less idiosyncratic exposure.

### Unit Consistency

Kenneth French distributes factor data in **percent** (0.31 = 0.31%). The pipeline computes returns in **decimal** (0.0031 = 0.31%). Setting `convert_percent_to_decimal: true` in `config/factors.yaml` divides all factor values by 100 before regression, ensuring both sides of the equation are in the same units. Running a regression with mixed units produces betas that are off by a factor of 100 — a silent but catastrophic error.

---

## 9. Factor-Aware Portfolio Construction

### Motivation

Standard mean-variance optimization weights assets by their full expected return `μ`. The factor-aware approach instead weights assets by their **abnormal return** — the component of expected return not explained by systematic risk factors. Factor exposure can be replicated cheaply via index instruments; only the unexplained component (alpha) justifies an active allocation tilt.

### Alpha-Weighted Portfolio

Given a rolling FF3 regression for each asset `i`:

```
r_i − RF = α_i + β_mkt,i · MKT_RF + β_smb,i · SMB + β_hml,i · HML + ε_i
```

Eligibility criteria:
1. Positive alpha only: `α_i > 0`
2. Optionally significant: additionally require `p_α,i < 0.05`

Among eligible assets, raw proportional weights:

```
w_i_raw = α_i / Σ_j α_j     (sum over eligible j)
```

Cap and renormalize:

```
w_i = clip(w_i_raw, 0, max_weight) / Σ_j clip(w_j_raw, 0, max_weight)
```

Ineligible assets receive zero weight. If no assets are eligible, the portfolio falls back to equal weights with a logged warning.

### Rolling Implementation

The factor-aware strategy is embedded in the rolling backtest engine identically to other strategies — no look-ahead violation is possible because:

1. The training window `returns[t-window:t]` ends at rebalance date `t` (exclusive)
2. Factor data is aligned to the same window by date intersection
3. The FF3 regression runs on the aligned `(T', n)` panel — `T'` is the count of common dates
4. Alpha estimates from the in-sample regression determine weights for the **subsequent** period `[t, t_next)`

### Limitations

**Alpha estimation noise is as severe as return estimation noise.** With `T = 252` and a residual volatility of ~10% annualized, the standard error of a daily alpha estimate is approximately `0.6% / √252 ≈ 0.038%` daily, or ~9.5% annualized. A 5% annualized alpha estimate has a t-statistic well below 2 — statistically indistinguishable from zero. The `require_significant_alpha=False` default is appropriate for exploration but weights produced in this regime are noise-driven.

**The strategy ignores covariance.** Unlike MVO, alpha-weighted portfolios do not account for `Σ`. Two highly correlated assets with similar alpha both receive positive weight, concentrating risk in a correlated cluster without penalty.
