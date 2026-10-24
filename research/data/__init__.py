"""Data fetching for financial research — historical prices, prediction
markets, and sentiment indicators.

Modules
-------
``prices.py``
    Historical close prices with fallback chain: Financial Datasets →
    Binance → Yahoo Finance.

``polymarket.py``
    Polymarket Gamma API client: market search, event history, tag-based
    filtering.

``polymarket_clob.py``
    Polymarket CLOB microstructure helpers (order-book depth, spread).

``sentiment.py``
    Fear & Greed Index and social sentiment data fetching.

``metaforecast.py``
    Metaforecast.org cross-platform prediction-market fusion.
"""

from research.data.prices import fetch_historical_prices
from research.data.polymarket import fetch_polymarket_markets
from research.data.sentiment import fetch_fear_greed, fetch_social_sentiment

__all__ = [
    "fetch_historical_prices",
    "fetch_polymarket_markets",
    "fetch_fear_greed",
    "fetch_social_sentiment",
]
