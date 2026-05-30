"""Python research mirror for the Cramer-Short forecasting engines.

Sub-packages
------------
``research.data``
    Historical price fetching with fallback chain (Financial Datasets →
    Binance → Yahoo), Polymarket Gamma API client, sentiment data.

``research.models``
    Forecasting engines: Markov regime model, trajectory Monte Carlo,
    Polymarket ensemble blending, Hidden Markov Models, GARCH volatility,
    conformal calibration, jump diffusion, and signal extraction.

``research.utils``
    Calibration helpers (YES-bias correction, anchor trust, Kalshi
    volatility signals), forecast-lab runtime defaults, regime calibrator.

``research.viz``
    Visualization helpers for survival curves, trajectories, and regime
    sequences.

``research.backtest``
    Walk-forward backtest harness, metrics (Brier, directional accuracy,
    CI coverage, CRPS, Murphy-Winkler), baseline guards, CLI entry point.

TypeScript parity
-----------------
Every module maps directly to a TypeScript counterpart under
``src/tools/finance/``.  Functions use the same names and mathematics so
Python results are comparable to the production TS engine.
"""

__version__ = "0.1.0"
