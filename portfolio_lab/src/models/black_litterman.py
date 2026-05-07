"""Black-Litterman model for posterior expected return estimation.

The Black-Litterman model combines equilibrium market returns (implied by
market-cap weights and the covariance matrix) with investor views to produce
a posterior expected return vector that is more stable than raw historical means.

Reference formulation:
  pi   = delta * Sigma * w_market          (equilibrium returns)
  mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
           * [(tau*Sigma)^-1 * pi + P'*Omega^-1 * Q]

Notation used throughout:
  Sigma  — (n x n) annualized covariance matrix
  w      — (n,) market or portfolio weights
  pi     — (n,) implied equilibrium returns
  P      — (k x n) view-picking matrix (k views, n assets)
  Q      — (k,) view returns
  Omega  — (k x k) view uncertainty (diagonal)
  tau    — scalar scaling factor for prior uncertainty (typically 0.01–0.10)
  delta  — risk aversion coefficient (typically 2–4)
"""

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_cov_matrix(cov_matrix: pd.DataFrame) -> None:
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError(
            "cov_matrix must be a pd.DataFrame with tickers as index and columns"
        )
    if cov_matrix.empty:
        raise ValueError("cov_matrix is empty")
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(
            f"cov_matrix must be square, got shape {cov_matrix.shape}"
        )


def _validate_weights_alignment(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> None:
    extra = set(market_weights.index) - set(cov_matrix.columns)
    missing = set(cov_matrix.columns) - set(market_weights.index)
    if extra or missing:
        raise ValueError(
            f"market_weights and cov_matrix columns are misaligned. "
            f"Extra in weights: {sorted(extra)}  Missing: {sorted(missing)}"
        )
    w_sum = float(market_weights.sum())
    if abs(w_sum - 1.0) > 1e-4:
        raise ValueError(
            f"market_weights must sum to 1.0, got {w_sum:.6f}"
        )


def _validate_assets_in_universe(
    assets: list[str],
    tickers: list[str],
    context: str,
) -> None:
    unknown = [a for a in assets if a not in tickers]
    if unknown:
        raise ValueError(
            f"{context}: assets not found in universe {tickers}: {unknown}"
        )


# ── Core functions ─────────────────────────────────────────────────────────────

def implied_equilibrium_returns(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Compute implied equilibrium returns: pi = delta * Sigma * w_market.

    These are the returns that would be expected if the market were in
    equilibrium given the current covariance structure and market weights.
    They serve as the prior mean vector in the Black-Litterman model.

    Args:
        cov_matrix: (n x n) annualized covariance matrix as a DataFrame.
            Index and columns must be ticker names.
        market_weights: (n,) portfolio weights indexed by ticker.
            Must align with cov_matrix columns. Must sum to approximately 1.
        risk_aversion: Risk aversion coefficient delta > 0 (default 2.5).
            Higher values imply investors demand more return per unit of risk.

    Returns:
        pd.Series of implied equilibrium returns, indexed by ticker.

    Raises:
        TypeError: If cov_matrix is not a pd.DataFrame.
        ValueError: If inputs are empty, misaligned, or risk_aversion <= 0.
    """
    _validate_cov_matrix(cov_matrix)
    _validate_weights_alignment(market_weights, cov_matrix)
    if risk_aversion <= 0.0:
        raise ValueError(
            f"risk_aversion must be > 0, got {risk_aversion}"
        )

    w = market_weights.reindex(cov_matrix.columns).to_numpy(dtype=float)
    sigma = cov_matrix.to_numpy(dtype=float)

    pi = risk_aversion * sigma @ w
    pi_series = pd.Series(pi, index=cov_matrix.columns, name="pi")

    logger.info(
        f"Equilibrium returns computed (delta={risk_aversion}): "
        + "  ".join(f"{k}={v:.4f}" for k, v in pi_series.items())
    )
    return pi_series


def build_absolute_views(
    assets: list[str],
    views_dict: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build P and Q matrices for absolute views.

    An absolute view on asset i states: E[r_i] = q_i.
    The corresponding P row has a 1 at position i and 0 elsewhere.

    Args:
        assets: Ordered list of all asset tickers in the universe.
        views_dict: Mapping of ticker -> expected return for that asset.
            Example: {"AAPL": 0.18, "MSFT": 0.15}

    Returns:
        P: np.ndarray of shape (k, n) — view-picking matrix.
        Q: np.ndarray of shape (k,) — view expected returns.

    Raises:
        ValueError: If views_dict is empty or contains unknown tickers.
    """
    if not views_dict:
        raise ValueError("build_absolute_views: views_dict is empty")
    _validate_assets_in_universe(list(views_dict.keys()), assets, "build_absolute_views")

    n = len(assets)
    asset_idx = {ticker: i for i, ticker in enumerate(assets)}

    P_rows = []
    Q_vals = []
    for ticker, expected_ret in views_dict.items():
        row = np.zeros(n)
        row[asset_idx[ticker]] = 1.0
        P_rows.append(row)
        Q_vals.append(float(expected_ret))

    P = np.array(P_rows)
    Q = np.array(Q_vals)

    logger.info(
        f"Absolute views built: {len(Q)} view(s) | "
        + "  ".join(f"{k}={v:.4f}" for k, v in views_dict.items())
    )
    return P, Q


def build_relative_views(
    assets: list[str],
    relative_views: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build P and Q matrices for relative views (outperformance).

    A relative view of the form "long A, short B by q" states:
    E[r_A - r_B] = q. The corresponding P row has +1 at A and -1 at B.

    Args:
        assets: Ordered list of all asset tickers in the universe.
        relative_views: List of dicts, each with keys:
            - 'long'        : ticker of the outperforming asset
            - 'short'       : ticker of the underperforming asset
            - 'view_return' : expected outperformance (float)
            Example: [{"long": "AAPL", "short": "MSFT", "view_return": 0.03}]

    Returns:
        P: np.ndarray of shape (k, n).
        Q: np.ndarray of shape (k,).

    Raises:
        ValueError: If relative_views is empty or contains unknown tickers.
    """
    if not relative_views:
        raise ValueError("build_relative_views: relative_views list is empty")

    n = len(assets)
    asset_idx = {ticker: i for i, ticker in enumerate(assets)}

    P_rows = []
    Q_vals = []
    for v in relative_views:
        long_asset = v["long"]
        short_asset = v["short"]
        view_return = float(v["view_return"])

        _validate_assets_in_universe(
            [long_asset, short_asset], assets, "build_relative_views"
        )

        row = np.zeros(n)
        row[asset_idx[long_asset]] = +1.0
        row[asset_idx[short_asset]] = -1.0
        P_rows.append(row)
        Q_vals.append(view_return)
        logger.info(
            f"Relative view: {long_asset} outperforms {short_asset} "
            f"by {view_return:.4f}"
        )

    P = np.array(P_rows)
    Q = np.array(Q_vals)
    return P, Q


def build_omega(
    P: np.ndarray,
    cov_matrix: pd.DataFrame,
    tau: float = 0.05,
    confidence: float = 0.5,
) -> np.ndarray:
    """Build the diagonal view-uncertainty matrix Omega.

    Omega represents the uncertainty of each investor view. A larger Omega
    means less confidence in the view — the model will stay closer to the
    equilibrium prior.

    Formula: Omega = diag(P * tau * Sigma * P.T) / confidence

    Where:
    - P * tau * Sigma * P.T gives the variance of each view implied by
      the prior covariance structure.
    - Dividing by confidence > 0 scales up uncertainty when confidence < 1.
      At confidence=1.0, Omega equals the standard proportional Omega.
      At confidence=0.5, uncertainty doubles (views are half as trusted).

    Args:
        P: (k x n) view-picking matrix.
        cov_matrix: (n x n) annualized covariance DataFrame.
        tau: Scalar prior uncertainty scaling factor (default 0.05).
            Typically in [0.01, 0.10].
        confidence: Investor confidence in the views in (0, 1] (default 0.5).
            Lower confidence -> larger Omega -> views have less weight.

    Returns:
        np.ndarray of shape (k, k) — diagonal Omega matrix.

    Raises:
        ValueError: If tau <= 0 or confidence not in (0, 1].
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if not (0.0 < confidence <= 1.0):
        raise ValueError(
            f"confidence must be in (0, 1], got {confidence}"
        )

    sigma = cov_matrix.to_numpy(dtype=float)
    raw_variance = np.diag(P @ (tau * sigma) @ P.T)
    omega_diag = raw_variance / confidence
    omega = np.diag(omega_diag)

    logger.info(
        f"Omega built: tau={tau}  confidence={confidence}  "
        f"diag={np.round(omega_diag, 6).tolist()}"
    )
    return omega


def black_litterman_posterior_returns(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    omega: np.ndarray | None = None,
    confidence: float = 0.5,
) -> pd.Series:
    """Compute the Black-Litterman posterior expected return vector.

    Combines the equilibrium prior (pi) with investor views (P, Q) using
    Bayesian updating:

        mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
                * [(tau*Sigma)^-1 * pi + P'*Omega^-1 * Q]

    When no views are expressed, mu_BL equals pi (the prior dominates).
    When views are very precise (small Omega), mu_BL is pulled toward Q.

    Args:
        cov_matrix: (n x n) annualized covariance DataFrame.
        market_weights: (n,) market weights, indexed by ticker.
        P: (k x n) view-picking matrix.
        Q: (k,) view returns.
        tau: Prior uncertainty scaling factor (default 0.05).
        risk_aversion: Risk aversion coefficient for pi (default 2.5).
        omega: (k x k) view uncertainty matrix. If None, built automatically
            via build_omega(P, cov_matrix, tau, confidence).
        confidence: Used only when omega is None (default 0.5).

    Returns:
        pd.Series of posterior expected returns, indexed by ticker.

    Raises:
        ValueError: If P and Q are dimensionally inconsistent or tau <= 0.
    """
    _validate_cov_matrix(cov_matrix)
    _validate_weights_alignment(market_weights, cov_matrix)

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if P.ndim != 2:
        raise ValueError(f"P must be 2-dimensional, got shape {P.shape}")
    k, n = P.shape
    if n != len(cov_matrix.columns):
        raise ValueError(
            f"P has {n} columns but cov_matrix has {len(cov_matrix.columns)} assets"
        )
    if Q.shape != (k,):
        raise ValueError(
            f"Q shape {Q.shape} is inconsistent with P shape {P.shape} "
            f"(expected ({k},))"
        )
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")

    sigma = cov_matrix.to_numpy(dtype=float)
    tickers = list(cov_matrix.columns)

    # Prior: equilibrium returns
    pi = implied_equilibrium_returns(
        cov_matrix, market_weights, risk_aversion
    ).to_numpy(dtype=float)

    # View uncertainty
    if omega is None:
        omega = build_omega(P, cov_matrix, tau=tau, confidence=confidence)

    # BL posterior formula (numerically stable via np.linalg.solve)
    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(omega)

    lhs = tau_sigma_inv + P.T @ omega_inv @ P                  # (n x n)
    rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q             # (n,)

    mu_bl = np.linalg.solve(lhs, rhs)

    mu_bl_series = pd.Series(mu_bl, index=tickers, name="mu_bl")

    logger.info(
        "Black-Litterman posterior returns: "
        + "  ".join(f"{k}={v:.4f}" for k, v in mu_bl_series.items())
    )
    return mu_bl_series
