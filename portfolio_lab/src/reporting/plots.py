"""Generate plots with matplotlib. No seaborn.

All functions return a matplotlib Figure object. The caller decides
whether to display or save — no plt.show() is called here.
Use export.save_figure() to persist a figure to disk.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def plot_returns_histogram(
    returns: pd.Series,
    title: str | None = None,
    bins: int = 50,
) -> plt.Figure:
    """Plot a simple histogram of a return series.

    Draws a vertical line at zero for reference. Intended for individual
    asset return distributions.

    Args:
        returns: pd.Series of returns. Name attribute is used in the title.
        title: Optional chart title. Defaults to 'Return Distribution - <name>'.
        bins: Number of histogram bins (default 50).

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If returns is empty after dropping NaN.
    """
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("plot_returns_histogram: returns is empty after dropping NaN")

    label = returns.name if returns.name else "Asset"
    chart_title = title or f"Return Distribution - {label}"

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(clean, bins=bins, edgecolor="black", linewidth=0.3, color="steelblue")
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--", label="Zero")

    ax.set_title(chart_title, fontsize=13)
    ax.set_xlabel("Daily Return", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    logger.info(f"Returns histogram created: '{chart_title}'  ({len(clean)} obs)")
    return fig


def plot_portfolio_returns_histogram(
    portfolio_returns: pd.Series,
    historical_var: float | None = None,
    historical_tvar: float | None = None,
    confidence_level: float | None = None,
    bins: int = 50,
) -> plt.Figure:
    """Plot portfolio daily return distribution with optional VaR and TVaR markers.

    VaR and TVaR are drawn as vertical dashed lines in the left tail,
    allowing visual inspection of the risk threshold and expected shortfall.

    Args:
        portfolio_returns: pd.Series of daily portfolio returns.
        historical_var: VaR as a return value (negative). If provided,
            a vertical line is drawn at this level.
        historical_tvar: TVaR as a return value (negative, <= historical_var).
            If provided, a vertical line is drawn at this level.
        confidence_level: Used in labels (e.g. 0.95 -> '95%'). Optional.
        bins: Number of histogram bins (default 50).

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If portfolio_returns is empty after dropping NaN.
    """
    clean = portfolio_returns.dropna()
    if clean.empty:
        raise ValueError(
            "plot_portfolio_returns_histogram: portfolio_returns is empty after dropping NaN"
        )

    cl_label = f" ({confidence_level:.0%})" if confidence_level is not None else ""

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(clean, bins=bins, edgecolor="black", linewidth=0.3, color="steelblue")
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--", label="Zero")

    if historical_var is not None:
        ax.axvline(
            historical_var,
            color="red",
            linewidth=1.8,
            linestyle="--",
            label=f"VaR{cl_label} = {historical_var:.4f}",
        )

    if historical_tvar is not None:
        ax.axvline(
            historical_tvar,
            color="darkred",
            linewidth=1.8,
            linestyle="-",
            label=f"TVaR{cl_label} = {historical_tvar:.4f}",
        )

    ax.set_title("Portfolio Daily Return Distribution", fontsize=13)
    ax.set_xlabel("Daily Return", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    logger.info(
        f"Portfolio histogram created: {len(clean)} obs"
        + (f"  VaR={historical_var:.4f}" if historical_var is not None else "")
        + (f"  TVaR={historical_tvar:.4f}" if historical_tvar is not None else "")
    )
    return fig


def plot_portfolio_weights(weights: pd.Series) -> plt.Figure:
    """Plot portfolio weight allocation as a horizontal bar chart.

    Args:
        weights: pd.Series of portfolio weights indexed by ticker.
            Values must be non-negative and sum to approximately 1.

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If weights is empty or contains NaN values.
    """
    if weights.empty:
        raise ValueError("plot_portfolio_weights: weights is empty")
    if weights.isna().any():
        raise ValueError("plot_portfolio_weights: weights contains NaN values")

    fig, ax = plt.subplots(figsize=(9, max(3, len(weights) * 0.7)))

    tickers = weights.index.tolist()
    values = weights.values
    y_pos = np.arange(len(tickers))

    bars = ax.barh(y_pos, values, color="steelblue", edgecolor="black", linewidth=0.4)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers, fontsize=10)
    ax.set_xlabel("Weight", fontsize=11)
    ax.set_title("Minimum Variance Portfolio — Weight Allocation", fontsize=13)
    ax.set_xlim(0, max(values) * 1.18)
    ax.axvline(0.0, color="black", linewidth=0.6)

    plt.tight_layout()
    logger.info(
        f"Portfolio weights chart created: {len(weights)} assets, "
        f"total weight = {values.sum():.6f}"
    )
    return fig


def plot_correlation_matrix(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Plot a correlation matrix heatmap using matplotlib imshow.

    Args:
        corr_matrix: Square DataFrame of pairwise correlations,
            indexed and columned by ticker. Values in [-1, 1].

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If corr_matrix is empty or not square.
    """
    if corr_matrix.empty:
        raise ValueError("plot_correlation_matrix: corr_matrix is empty")
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(
            f"plot_correlation_matrix: expected square matrix, "
            f"got shape {corr_matrix.shape}"
        )

    n = len(corr_matrix)
    tickers = list(corr_matrix.columns)
    data = corr_matrix.values

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))

    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tickers, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(tickers, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            text_color = "black" if abs(val) < 0.75 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    ax.set_title("Asset Correlation Matrix", fontsize=13)
    plt.tight_layout()
    logger.info(f"Correlation matrix heatmap created: {n}x{n}")
    return fig


def plot_returns_distribution_comparison(
    returns_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    bins: int = 50,
) -> plt.Figure:
    """Plot overlapping return distributions for all assets and the portfolio.

    Each asset and the portfolio are plotted as semi-transparent histograms
    on the same axes, allowing visual comparison of distribution widths
    and centers. The portfolio is highlighted with a distinct color and
    a thicker outline.

    Args:
        returns_df: DataFrame of asset daily returns (DatetimeIndex, columns=tickers).
        portfolio_returns: pd.Series of portfolio daily returns.
        bins: Number of histogram bins (default 50).

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If returns_df or portfolio_returns is empty.
    """
    if returns_df.empty:
        raise ValueError(
            "plot_returns_distribution_comparison: returns_df is empty"
        )
    if portfolio_returns.empty:
        raise ValueError(
            "plot_returns_distribution_comparison: portfolio_returns is empty"
        )

    fig, ax = plt.subplots(figsize=(12, 5))

    asset_colors = plt.cm.tab10.colors
    for idx, ticker in enumerate(returns_df.columns):
        clean = returns_df[ticker].dropna()
        color = asset_colors[idx % len(asset_colors)]
        ax.hist(
            clean,
            bins=bins,
            alpha=0.35,
            color=color,
            edgecolor="none",
            label=ticker,
            density=True,
        )

    clean_port = portfolio_returns.dropna()
    ax.hist(
        clean_port,
        bins=bins,
        alpha=0.85,
        color="black",
        edgecolor="black",
        label="Portfolio",
        density=True,
        histtype="step",
        linewidth=2.0,
    )

    ax.axvline(0.0, color="grey", linewidth=0.9, linestyle="--")
    ax.set_title("Return Distribution Comparison — Assets vs Portfolio", fontsize=13)
    ax.set_xlabel("Daily Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    logger.info(
        f"Return distribution comparison created: "
        f"{len(returns_df.columns)} assets + portfolio"
    )
    return fig


def plot_risk_return_scatter(
    mean_returns: pd.Series,
    volatilities: pd.Series,
    portfolio_point: tuple | dict,
) -> plt.Figure:
    """Plot annualized risk-return scatter for assets with portfolio highlighted.

    Each asset is plotted as a dot with its ticker label. The portfolio is
    plotted as a larger star marker in a contrasting color so it can be
    visually located relative to the individual assets.

    Args:
        mean_returns: pd.Series of annualized mean returns, indexed by ticker.
        volatilities: pd.Series of annualized volatilities, indexed by ticker.
        portfolio_point: Either a tuple (vol, ret) or a dict with keys
            'volatility' and 'return'. Both values must be annualized floats.

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If mean_returns or volatilities is empty, or if
            portfolio_point does not contain the expected keys/values.
    """
    if mean_returns.empty:
        raise ValueError("plot_risk_return_scatter: mean_returns is empty")
    if volatilities.empty:
        raise ValueError("plot_risk_return_scatter: volatilities is empty")

    if isinstance(portfolio_point, dict):
        try:
            port_vol = float(portfolio_point["volatility"])
            port_ret = float(portfolio_point["return"])
        except KeyError as e:
            raise ValueError(
                f"plot_risk_return_scatter: portfolio_point dict missing key {e}. "
                "Expected keys: 'volatility', 'return'"
            ) from e
    else:
        try:
            port_vol, port_ret = float(portfolio_point[0]), float(portfolio_point[1])
        except (IndexError, TypeError) as e:
            raise ValueError(
                "plot_risk_return_scatter: portfolio_point tuple must have at least 2 elements"
            ) from e

    tickers = mean_returns.index.tolist()
    vols = volatilities.reindex(mean_returns.index).values
    rets = mean_returns.values

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(vols, rets, s=80, color="steelblue", edgecolors="black",
               linewidths=0.5, zorder=3, label="Assets")

    for ticker, v, r in zip(tickers, vols, rets):
        ax.annotate(
            ticker,
            xy=(v, r),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.scatter(
        port_vol, port_ret,
        s=200,
        marker="*",
        color="red",
        edgecolors="darkred",
        linewidths=0.8,
        zorder=4,
        label="Portfolio (min var)",
    )

    ax.axhline(0.0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_title("Risk-Return Scatter (Annualized)", fontsize=13)
    ax.set_xlabel("Volatility (Ann.)", fontsize=11)
    ax.set_ylabel("Mean Return (Ann.)", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    logger.info(
        f"Risk-return scatter created: {len(tickers)} assets, "
        f"portfolio at vol={port_vol:.4f} ret={port_ret:.4f}"
    )
    return fig


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> plt.Figure:
    """Plot cumulative returns over time for all assets and the portfolio.

    Cumulative return is computed as (1 + r).cumprod() - 1, starting from 0.
    Assets are plotted as thin semi-transparent lines; the portfolio is
    highlighted with a thicker black line.

    Args:
        returns_df: DataFrame of asset daily returns (DatetimeIndex, columns=tickers).
        portfolio_returns: pd.Series of portfolio daily returns (same DatetimeIndex).

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If returns_df or portfolio_returns is empty.
    """
    if returns_df.empty:
        raise ValueError("plot_cumulative_returns: returns_df is empty")
    if portfolio_returns.empty:
        raise ValueError("plot_cumulative_returns: portfolio_returns is empty")

    cum_assets = (1 + returns_df).cumprod() - 1
    cum_port = (1 + portfolio_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(13, 5))

    asset_colors = plt.cm.tab10.colors
    for idx, ticker in enumerate(cum_assets.columns):
        color = asset_colors[idx % len(asset_colors)]
        ax.plot(
            cum_assets.index,
            cum_assets[ticker],
            color=color,
            linewidth=1.0,
            alpha=0.6,
            label=ticker,
        )

    ax.plot(
        cum_port.index,
        cum_port,
        color="black",
        linewidth=2.2,
        label="Portfolio",
        zorder=5,
    )

    ax.axhline(0.0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_title("Cumulative Returns — Assets and Portfolio", fontsize=13)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.0%}")
    )

    plt.tight_layout()
    logger.info(
        f"Cumulative returns chart created: "
        f"{len(returns_df.columns)} assets + portfolio, "
        f"{len(returns_df)} observations"
    )
    return fig


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    mean_returns: pd.Series,
    volatilities: pd.Series,
    min_var_point: dict,
    max_sharpe_point: dict,
) -> plt.Figure:
    """Plot the efficient frontier with individual assets and key portfolios.

    Args:
        frontier_df: DataFrame with columns 'return' and 'volatility'.
        mean_returns: Annualized asset mean returns, indexed by ticker.
        volatilities: Annualized asset volatilities, indexed by ticker.
        min_var_point: Dict with keys 'return' and 'volatility'.
        max_sharpe_point: Dict with keys 'return', 'volatility', and 'sharpe'.

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If frontier_df is empty.
    """
    if frontier_df.empty:
        raise ValueError("plot_efficient_frontier: frontier_df is empty")

    tickers = mean_returns.index.tolist()
    asset_vols = volatilities.reindex(mean_returns.index).values
    asset_rets = mean_returns.values

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        frontier_df["volatility"],
        frontier_df["return"],
        color="navy",
        linewidth=2.0,
        zorder=2,
        label="Efficient Frontier",
    )

    ax.scatter(
        asset_vols, asset_rets,
        s=80,
        color="steelblue",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
        label="Assets",
    )

    for ticker, v, r in zip(tickers, asset_vols, asset_rets):
        ax.annotate(
            ticker,
            xy=(v, r),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )

    mv = min_var_point
    ax.scatter(
        mv["volatility"], mv["return"],
        s=180,
        marker="D",
        color="green",
        edgecolors="darkgreen",
        linewidths=0.8,
        zorder=4,
        label=f"Min Variance  ret={mv['return']:.2%}  vol={mv['volatility']:.2%}",
    )

    ms = max_sharpe_point
    ax.scatter(
        ms["volatility"], ms["return"],
        s=220,
        marker="*",
        color="red",
        edgecolors="darkred",
        linewidths=0.8,
        zorder=4,
        label=(
            f"Max Sharpe  ret={ms['return']:.2%}  "
            f"vol={ms['volatility']:.2%}  SR={ms.get('sharpe', 0.0):.2f}"
        ),
    )

    ax.axhline(0.0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_title("Efficient Frontier — Markowitz Mean-Variance", fontsize=13)
    ax.set_xlabel("Volatility (Annualized)", fontsize=11)
    ax.set_ylabel("Expected Return (Annualized)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    logger.info(
        f"Efficient frontier chart created: {len(frontier_df)} frontier points, "
        f"{len(tickers)} assets"
    )
    return fig


def plot_backtest_results(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights_history: pd.DataFrame,
) -> plt.Figure:
    """Plot backtest performance: cumulative returns, drawdown, and weight evolution.

    Three stacked panels:
        1. Cumulative returns — portfolio vs benchmark
        2. Portfolio drawdown
        3. Weight allocation at each rebalance date (stacked area)

    Args:
        portfolio_returns: Daily portfolio return series (DatetimeIndex).
        benchmark_returns: Daily benchmark return series, aligned to the
            same DatetimeIndex as portfolio_returns.
        weights_history: DataFrame of portfolio weights at each rebalance
            date (DatetimeIndex, columns=tickers).

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If portfolio_returns or benchmark_returns is empty.
    """
    if portfolio_returns.empty:
        raise ValueError("plot_backtest_results: portfolio_returns is empty")
    if benchmark_returns.empty:
        raise ValueError("plot_backtest_results: benchmark_returns is empty")

    bm_aligned = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0)

    cum_port = (1 + portfolio_returns).cumprod() - 1
    cum_bm = (1 + bm_aligned).cumprod() - 1

    cum_val = (1 + portfolio_returns).cumprod()
    rolling_max = cum_val.expanding().max()
    drawdown = (cum_val - rolling_max) / rolling_max

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.45)

    # Panel 1: cumulative returns
    ax1 = axes[0]
    ax1.plot(
        cum_port.index, cum_port.values,
        color="steelblue", linewidth=2.0, label="Portfolio",
    )
    ax1.plot(
        cum_bm.index, cum_bm.values,
        color="darkorange", linewidth=1.5, linestyle="--", label="Benchmark",
    )
    ax1.axhline(0.0, color="grey", linewidth=0.7, linestyle=":")
    ax1.set_title("Cumulative Returns — Portfolio vs Benchmark", fontsize=12)
    ax1.set_ylabel("Cumulative Return", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Panel 2: drawdown
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown.values, 0.0, color="crimson", alpha=0.4)
    ax2.plot(drawdown.index, drawdown.values, color="darkred", linewidth=1.0)
    ax2.axhline(0.0, color="grey", linewidth=0.7, linestyle=":")
    ax2.set_title("Portfolio Drawdown", fontsize=12)
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Panel 3: weight evolution (stacked area)
    ax3 = axes[2]
    if not weights_history.empty:
        tickers = list(weights_history.columns)
        colors = plt.cm.tab10.colors
        ax3.stackplot(
            weights_history.index,
            [weights_history[t].fillna(0.0).values for t in tickers],
            labels=tickers,
            colors=[colors[i % len(colors)] for i in range(len(tickers))],
            alpha=0.85,
        )
        ax3.set_ylim(0, 1.02)
        ax3.legend(fontsize=9, loc="upper left")
    ax3.set_title("Weight Allocation at Each Rebalance Date", fontsize=12)
    ax3.set_ylabel("Weight", fontsize=10)

    plt.tight_layout()
    logger.info(
        f"Backtest chart created: {len(portfolio_returns)} obs, "
        f"{len(weights_history)} rebalance dates"
    )
    return fig


def plot_strategy_comparison(
    strategy_returns: dict,
    benchmark_returns: pd.Series,
    comparison_df: pd.DataFrame,
) -> plt.Figure:
    """Compare multiple portfolio strategies: cumulative returns, drawdown, Sharpe.

    Three stacked panels:
        1. Cumulative returns for all strategies plus the benchmark.
        2. Drawdown for all strategies.
        3. Horizontal bar chart of Sharpe ratios.

    Args:
        strategy_returns: Dict mapping strategy_name -> pd.Series of daily returns.
        benchmark_returns: Benchmark return series (aligned to strategy period).
        comparison_df: Output of build_strategy_comparison_table (strategies × metrics).

    Returns:
        matplotlib Figure object.

    Raises:
        ValueError: If strategy_returns is empty.
    """
    if not strategy_returns:
        raise ValueError("plot_strategy_comparison: strategy_returns is empty")

    ref_index = next(iter(strategy_returns.values())).index
    bm_aligned = benchmark_returns.reindex(ref_index).fillna(0.0)

    colors = plt.cm.tab10.colors
    strategy_names = list(strategy_returns.keys())

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.subplots_adjust(hspace=0.45)

    # Panel 1: cumulative returns
    ax1 = axes[0]
    for i, name in enumerate(strategy_names):
        cum = (1 + strategy_returns[name]).cumprod() - 1
        ax1.plot(cum.index, cum.values, color=colors[i], linewidth=1.8, label=name)
    cum_bm = (1 + bm_aligned).cumprod() - 1
    ax1.plot(
        cum_bm.index, cum_bm.values,
        color="black", linewidth=1.4, linestyle="--", label="SPY",
    )
    ax1.axhline(0.0, color="grey", linewidth=0.7, linestyle=":")
    ax1.set_title("Cumulative Returns by Strategy", fontsize=12)
    ax1.set_ylabel("Cumulative Return", fontsize=10)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Panel 2: drawdown
    ax2 = axes[1]
    for i, name in enumerate(strategy_names):
        cum_val = (1 + strategy_returns[name]).cumprod()
        dd = (cum_val - cum_val.expanding().max()) / cum_val.expanding().max()
        ax2.plot(dd.index, dd.values, color=colors[i], linewidth=1.5, alpha=0.85, label=name)
    ax2.axhline(0.0, color="grey", linewidth=0.7, linestyle=":")
    ax2.set_title("Drawdown by Strategy", fontsize=12)
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Panel 3: Sharpe ratio bar chart
    ax3 = axes[2]
    if "sharpe_ratio" in comparison_df.columns:
        sharpe = comparison_df["sharpe_ratio"].astype(float)
        y_pos = np.arange(len(sharpe))
        bar_colors = [colors[strategy_names.index(s) % len(colors)] for s in sharpe.index]
        bars = ax3.barh(
            y_pos, sharpe.values,
            color=bar_colors, edgecolor="black", linewidth=0.4,
        )
        for bar, val in zip(bars, sharpe.values):
            ax3.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center", ha="left", fontsize=9,
            )
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(sharpe.index, fontsize=10)
        ax3.set_xlabel("Sharpe Ratio", fontsize=10)
        ax3.set_title("Sharpe Ratio by Strategy", fontsize=12)
        ax3.set_xlim(0, max(sharpe.values) * 1.2)
    ax3.axvline(0.0, color="grey", linewidth=0.6)

    plt.tight_layout()
    logger.info(
        f"Strategy comparison chart created: {len(strategy_names)} strategies"
    )
    return fig


def plot_price_series(prices: pd.DataFrame) -> plt.Figure:
    """Plot normalized (base-100) price series for all assets.

    Placeholder — Phase 4.
    """
    raise NotImplementedError("plots.plot_price_series: Phase 4")
