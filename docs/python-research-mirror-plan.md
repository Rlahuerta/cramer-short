# Python Research Mirror: Polymarket & Markov Forecast Engine

## Goal

Create a local Python virtual environment and a suite of Jupyter Notebooks that mirror the TypeScript forecasting engines (`markov-distribution.ts`, `polymarket-forecast.ts`, `ensemble.ts`). This enables interactive research, visual debugging, parameter tuning, and rapid experimentation without the LangChain tool-call overhead.

---

## 1. Conda Environment Setup

Create `environment-research.yml` at repo root:

```yaml
name: cramer-research
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - numpy
  - scipy
  - pandas
  - matplotlib
  - seaborn
  - jupyterlab
  - ipywidgets
  - requests
  - python-dotenv
  - pip:
      - plotly
      - statsmodels
      - arch
      - yfinance
      - hmmlearn
```

Bootstrap the environment and install the local package in editable mode:

```bash
# From repo root
conda env create -f environment-research.yml
conda activate cramer-research

# Install the research package in editable mode
cd research
pip install -e .
cd ..
```

After adding new packages, update the `.yml` and sync:

```bash
conda env update -f environment-research.yml --prune
```

Add to `.gitignore`:
```
.ipynb_checkpoints/
*.ipynb_checkpoints
```

---

## 2. Notebook Architecture

Each notebook isolates one subsystem. They share a `research/` package (see §3).

| Notebook | Purpose | Maps to TS File |
|----------|---------|-----------------|
| `01_Data_Pipeline.ipynb` | Fetch & cache historical prices, Polymarket Gamma API, sentiment. | `api.ts`, `binance.ts`, `polymarket.ts` |
| `02_Markov_Engine.ipynb` | Regime classification, transition matrix, HMM, structural break. | `markov-distribution.ts`, `hmm.ts` |
| `03_Polymarket_Signals.ipynb` | Search, quality-weighting, YES-bias correction, history flags. | `polymarket-forecast.ts`, `ensemble.ts` |
| `04_Ensemble_Blending.ipynb` | Combine PM + sentiment + fundamentals + options + Markov. | `ensemble.ts` |
| `05_Visualization_Dashboard.ipynb` | Distribution charts, trajectory plots, backtest comparison. | `price-distribution-chart.ts`, backtest harness |
| `06_Live_Forecast.ipynb` | End-to-end: ticker → forecast → action signal. | Full pipeline |
| `07_Parameter_Tuning.ipynb` | Grid-search calibration kappa, decay, threshold sensitivity. | Backtest experiments |

---

## 3. Shared `research/` Python Package

Create `research/` at repo root (or under `notebooks/research/`):

```
research/
  __init__.py
  data/
    prices.py          # Financial Datasets API, Binance, Yahoo fallbacks
    polymarket.py      # Gamma API client with caching
    sentiment.py       # Fear & Greed, social scrapers
  models/
    markov.py          # 3-state regime, transition matrix, Dirichlet smoothing
    hmm.py             # Gaussian HMM (Baum-Welch, Viterbi)
    ensemble.py        # Weighted signal blending, variance, CI, quality score
  viz/
    distributions.py   # Survival curves, ASCII bar charts → matplotlib
    trajectories.py    # Day-by-day expected price + CI ribbons
    backtests.py       # Walk-forward plots, metric tables
  utils/
    calibration.py     # Bayesian shrinkage, YES-bias correction
    impact_map.py      # Asset-class → deltaYes/deltaNo lookup
    signal_extractor.py # Ticker → search phrases (replicate TS logic)
```

### Key Porting Decisions

| TS Construct | Python Equivalent | Notes |
|--------------|-------------------|-------|
| `z.enum` inputs | `Literal` + pydantic `BaseModel` | Same validation, better IDE support |
| `fetch()` | `requests.Session()` with retries | Re-use TS retry logic (exponential backoff) |
| JSONL snapshots | `pandas.read_json(..., lines=True)` | Polymarket history |
| Matrix exponentiation | `numpy.linalg.matrix_power` or `scipy.linalg.expm` | For n-step regime forecasts |
| HMM Baum-Welch | `hmmlearn.GaussianHMM` or custom EM | `hmmlearn` is battle-tested; custom EM only if we need exact TS parity |
| Dirichlet smoothing | `numpy.random.dirichlet` or manual `α` addition | Match `α = max(0.01, 5/N)` from TS |
| Structural break | Frobenius norm of matrix difference | `numpy.linalg.norm(A-B, 'fro')` |
| Survival interpolation | `scipy.interpolate.interp1d` or `numpy.interp` | Log-spaced price levels |
| Walk-forward backtest | `pandas` rolling windows | Match 90/60/120 warmup windows |

---

## 4. Data Pipeline (`01_Data_Pipeline.ipynb`)

### 4.1 Historical Prices

Replicate the TS fallback chain:

```python
from research.data.prices import fetch_historical_prices

prices = fetch_historical_prices(
    ticker="BTC-USD",
    days=365,
    sources=["financial_datasets", "binance", "yahoo"],
)
# Returns: pd.DataFrame with columns [date, open, high, low, close, volume]
```

Cache to `research/.cache/prices/{ticker}_{start}_{end}.parquet`.

### 4.2 Polymarket Markets

```python
from research.data.polymarket import fetch_polymarket_markets

markets = fetch_polymarket_markets(
    query="bitcoin",
    tags=["bitcoin", "crypto-prices", "crypto"],
    min_volume_24h=1000,
    cache_ttl_minutes=5,
)
# Returns: pd.DataFrame with columns [market_id, question, probability, volume24h, age_days, end_date]
```

### 4.3 Sentiment

```python
from research.data.sentiment import fetch_fear_greed, fetch_social_sentiment

fear_greed = fetch_fear_greed()       # alternative.me API
social = fetch_social_sentiment("BTC", limit=25)
```

---

## 5. Markov Engine (`02_Markov_Engine.ipynb`)

### 5.1 Regime Classification

```python
from research.models.markov import classify_regime, estimate_transition_matrix

returns = prices["close"].pct_change().dropna()
regimes = classify_regime(returns, method="adaptive_threshold")
# adaptive_threshold = 0.5 * median(|returns|)

P = estimate_transition_matrix(
    regimes,
    decay=0.96,
    dirichlet_alpha=lambda N: max(0.01, 5 / N),
)
```

### 5.2 Structural Break Detection

```python
from research.models.markov import detect_structural_break

break_detected, divergence = detect_structural_break(
    regimes,
    window=60,
    threshold=0.15,
)
if break_detected:
    P = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
```

### 5.3 HMM Enhancement

```python
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=50, tol=1e-3)
model.fit(returns.values.reshape(-1, 1))
hmm_states = model.predict(returns.values.reshape(-1, 1))
```

### 5.4 Distribution & Calibration

```python
from research.models.markov import compute_distribution, calibrate_probabilities

raw_dist = compute_distribution(P, current_price=78000, horizon=7)
calibrated = calibrate_probabilities(
    raw_dist,
    kappa=0.35,
    center=0.5,
    asset_profile="crypto",  # or "etf", "commodity"
)
```

---

## 6. Polymarket Signals (`03_Polymarket_Signals.ipynb`)

### 6.1 YES-Bias Correction

```python
from research.utils.calibration import adjust_yes_bias

p_adj = adjust_yes_bias(p=0.72, additive_discount=0.035)
```

### 6.2 Quality Weighting

```python
from research.models.ensemble import compute_market_quality

w = compute_market_quality(
    age_days=14,
    volume_24h=450_000,
    tier="macro",          # macro | electoral | geopolitical
    price_spike_detected=False,
    transitory_move=False,
)
```

### 6.3 Conditional Return

```python
from research.utils.impact_map import lookup_impact

delta_yes, delta_no = lookup_import(category="crypto-prices", asset_class="crypto")
expected_return = p_adj * delta_yes + (1 - p_adj) * delta_no
```

### 6.4 History Flags

```python
from research.data.polymarket import evaluate_market_history

flags = evaluate_market_history(
    market_id="...",
    snapshots_path=".cramer-short/polymarket-snapshots.jsonl",
)
# Returns: {price_spike_detected, transitory_move, baseline_probability}
```

---

## 7. Ensemble Blending (`04_Ensemble_Blending.ipynb`)

```python
from research.models.ensemble import compute_ensemble_forecast

forecast = compute_ensemble_forecast(
    polymarket_signal=0.03,      # 3% expected return
    sentiment_signal=0.01,
    fundamental_signal=-0.005,
    options_signal=0.0,
    markov_signal=0.02,
    pm_avg_quality=0.85,
    pm_variance=0.001,
)

print(forecast.expected_return)
print(forecast.ci_95)
print(forecast.quality_score)   # 0-100
print(forecast.grade)           # A | B | C | D
```

---

## 8. Visualization (`05_Visualization_Dashboard.ipynb`)

### 8.1 Survival Curve (Distribution)

```python
import matplotlib.pyplot as plt
from research.viz.distributions import plot_survival_curve

plot_survival_curve(
    distribution=calibrated,
    current_price=78000,
    action_signal=action,
    title="BTC 7-Day Forecast",
)
```

### 8.2 Trajectory Ribbon

```python
from research.viz.trajectories import plot_trajectory

plot_trajectory(
    trajectory_df,  # columns: day, expected, lower, upper, p_up
    current_price=78000,
)
```

### 8.3 Backtest Heatmap

```python
from research.viz.backtests import plot_backtest_heatmap

plot_backtest_heatmap(
    results_df,  # rows: tickers, columns: horizons
    metric="directional_accuracy",
)
```

---

## 9. Live Forecast Notebook (`06_Live_Forecast.ipynb`)

End-to-end single cell:

```python
from research.pipeline import run_forecast

result = run_forecast(
    ticker="BTC",
    horizon=7,
    current_price=78000,
    include_polymarket=True,
    include_markov=True,
    include_sentiment=True,
    trajectory=True,
)

result.display()  # rich HTML/print output
```

---

## 10. Parameter Tuning (`07_Parameter_Tuning.ipynb`)

Interactive widgets for key hyperparameters:

```python
from ipywidgets import interact, FloatSlider

@interact(
    decay=FloatSlider(min=0.90, max=0.99, step=0.01, value=0.96),
    kappa=FloatSlider(min=0.10, max=0.80, step=0.05, value=0.35),
    threshold=FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0),
)
def tune(decay, kappa, threshold):
    P = estimate_transition_matrix(regimes, decay=decay)
    dist = compute_distribution(P, current_price=78000, horizon=7)
    cal = calibrate_probabilities(dist, kappa=kappa)
    plot_survival_curve(cal)
```

---

## 11. Verification & Parity Testing

### 11.1 Fixture Parity

The TS backtest harness downloads fixtures (`src/tools/finance/backtest/download-fixtures.ts`). Re-use those same CSV/JSON files in Python to verify bit-exact parity for:

- Regime sequences
- Transition matrices
- HMM state assignments
- Ensemble weights
- Final probability distributions

### 11.2 Snapshot Parity

Run both TS and Python engines against the same Polymarket snapshot JSONL and assert:

```python
assert abs(py_quality_score - ts_quality_score) < 1e-6
assert py_grade == ts_grade
```

### 11.3 Regression Tests

Add `pytest` in Python layer:

```bash
pytest research/tests/ -v
```

Cover:
- `test_markov_regime_classification` — SPY, BTC, TSLA fixtures
- `test_ensemble_yes_bias` — known bias cases from Reichenbach & Walther
- `test_polymarket_quality_weight` — whale flag → 50% penalty
- `test_structural_break_fallback` — divergence > threshold → default matrix

---

## 12. Milestones

| Phase | Deliverable | ETA |
|-------|-------------|-----|
| 0 | Venv + `requirements-research.txt` | 30 min |
| 1 | `research/data/` (prices, PM, sentiment) | 2 hrs |
| 2 | `research/models/markov.py` + `02_Markov_Engine.ipynb` | 4 hrs |
| 3 | `research/models/ensemble.py` + `03_` & `04_` notebooks | 3 hrs |
| 4 | `research/viz/` + `05_` & `06_` notebooks | 3 hrs |
| 5 | Parity tests + `07_Parameter_Tuning.ipynb` | 3 hrs |
| 6 | Documentation + README in `research/` | 1 hr |

---

## 13. Risks & Tradeoffs

| Risk | Mitigation |
|------|------------|
| `hmmlearn` behavior differs from custom TS EM | Add a `strict_mode` flag that uses a hand-ported EM loop for parity tests; `hmmlearn` for speed in research. |
| Polymarket API changes | Keep TS `polymarket.ts` as the source of truth; Python client is a thin wrapper that re-uses the same URL/param logic. |
| TS ↔ Python fixture drift | CI job that runs `download-fixtures.ts` before Python tests. |
| Notebook bloat | Each notebook < 200 cells. Move heavy computation to `research/` modules. |
| Environment reproducibility | `requirements-research.txt` + `runtime.txt` pinning Python version. |

---

## 14. Next Steps

1. **Bootstrap venv** (§1)
2. **Create `research/data/prices.py`** — port `api.ts` + `binance.ts` fallback chain
3. **Create `02_Markov_Engine.ipynb`** — load SPY fixture, classify regimes, compare against TS output
4. **Iterate** on parity until backtest metrics match within 1e-4
5. **Promote** to full notebook suite once single-ticker parity is proven
