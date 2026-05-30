"""Visualization helpers for forecast distributions and trajectories.

Public API
----------
``plot_survival_curve(distribution, current_price, ...)``
    Render a survival curve showing P(price > X) for each price level.
    Includes vertical markers for the current price and CI bounds.

``plot_trajectory(trajectory, ...)``
    Plot the expected price path over the forecast horizon with shaded
    confidence bands (p5–p95).

``plot_regime_sequence(regimes, ...)``
    Display the regime classification sequence over a historical window
    with color-coded bull/bear/sideways bands.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_survival_curve(
    distribution: dict[str, float],
    current_price: float,
    title: str = "Survival Curve",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot a survival curve P(price > X) vs price levels.

    Parameters
    ----------
    distribution : dict[str, float]
        Map from price level to P(>price).
    current_price : float
        Current price for reference line.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    prices = sorted([float(k) for k in distribution.keys()])
    probs = [distribution[str(p)] for p in prices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prices, probs, marker="o", markersize=3, linewidth=2)
    ax.axvline(current_price, color="red", linestyle="--", label=f"Current: ${current_price:,.2f}")
    ax.set_xlabel("Price")
    ax.set_ylabel("P(>price)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    return fig


def plot_trajectory(
    trajectory: pd.DataFrame,
    current_price: float,
    title: str = "Price Trajectory Forecast",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot day-by-day expected price with confidence interval ribbons.

    Parameters
    ----------
    trajectory : pd.DataFrame
        Columns: day (int), expected (float), lower (float), upper (float), p_up (float).
    current_price : float
        Current price for reference.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    days = trajectory["day"].values
    expected = trajectory["expected"].values
    lower = trajectory["lower"].values
    upper = trajectory["upper"].values

    ax.fill_between(days, lower, upper, alpha=0.2, label="90% CI")
    ax.plot(days, expected, linewidth=2, label="Expected")
    ax.axhline(current_price, color="red", linestyle="--", label=f"Current: ${current_price:,.2f}")

    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_regime_sequence(
    prices: pd.DataFrame,
    regimes: list[str],
    title: str = "Regime Sequence",
    figsize: tuple[int, int] = (14, 4),
) -> plt.Figure:
    """Plot price series coloured by regime state.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns: date, close.
    regimes : list[str]
        Regime states (bull/bear/sideways) aligned with prices.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {"bull": "green", "bear": "red", "sideways": "gray"}
    dates = prices["date"].values
    closes = prices["close"].values

    # Plot segments
    for i in range(len(regimes)):
        color = colors.get(regimes[i], "blue")
        if i < len(dates) - 1:
            ax.plot(dates[i : i + 2], closes[i : i + 2], color=color, linewidth=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig
