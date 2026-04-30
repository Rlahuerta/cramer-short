# Cramer-Short Research (Python)

Python research mirror of the TypeScript forecasting engines for interactive analysis, visual debugging, and parameter tuning.

## Quick Start

### 1. Create the Conda Environment

```bash
# (Optional) Activate a base conda environment
source ~/anaconda3/bin/activate root

# From repo root
conda env create -f environment-research.yml
```

This creates a Python 3.12 environment named `cramer-research` with all scientific dependencies pre-installed.

### 2. Activate and Install the Package

```bash
conda activate cramer-research

cd research
pip install -e .
cd ..
```

The `-e .` flag installs the package in **editable mode**, so changes to `.py` files are immediately reflected in notebooks without re-installation.

### 3. Verify Installation

```bash
python -c "import research; print(research.__version__)"
```

Expected output: `0.1.0`

### 4. Launch JupyterLab

```bash
jupyter lab notebooks/
```

Open `01_Data_Pipeline.ipynb` as a smoke test.

---

## Package Structure

```
research/
├── data/                        # Data fetching
│   ├── prices.py                # Historical prices (Financial Datasets → Binance → Yahoo)
│   ├── polymarket.py            # Polymarket Gamma API client
│   └── sentiment.py             # Fear & Greed, social sentiment
├── models/                      # Forecasting engines
│   ├── markov.py                # 3-state regime model, transition matrix, structural break
│   └── ensemble.py              # Polymarket ensemble: YES-bias, quality weights, blending
├── utils/                       # Calibration helpers
│   └── calibration.py           # YES-bias correction utilities
└── viz/                         # Visualization
    └── distributions.py         # Survival curves, trajectory ribbons, regime plots
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_Data_Pipeline.ipynb` | Fetch prices, Polymarket markets, sentiment |
| `02_Markov_Engine.ipynb` | Regime classification, transition matrix, HMM, structural break |
| `03_Polymarket_Signals.ipynb` | Market search, quality weighting, history flags |
| `04_Ensemble_Blending.ipynb` | Signal combination, variance, CI, quality scoring |
| `05_Visualization_Dashboard.ipynb` | Distribution charts, backtest comparison |
| `06_Live_Forecast.ipynb` | End-to-end ticker → forecast → action signal |
| `07_Parameter_Tuning.ipynb` | Interactive sliders for kappa, decay, threshold |

---

## Updating Dependencies

After adding new packages to `environment-research.yml`:

```bash
conda env update -f environment-research.yml --prune
```

Or install one-off packages into the active environment:

```bash
conda install -c conda-forge some-package
# or
pip install some-package
```

Remember to sync `environment-research.yml` if the dependency is needed by the package.

---

## Environment Variables

The research package reads the same environment variables as the TypeScript app:

| Variable | Required By | Purpose |
|----------|-------------|---------|
| `FINANCIAL_DATASETS_API_KEY` | `prices.py` | Primary price source |
| `WHALE_ALERT_API_KEY` | `onchain-crypto.ts` (TS only) | Multi-chain whale tracking |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Optional | LLM-based features |

Copy `.env` from the repo root — the Python package automatically loads it via `python-dotenv`.

---

## Development

### Code Style

```bash
# Format
black research/

# Lint
ruff check research/

# Type check
mypy research/
```

### Running Tests

```bash
pytest research/tests/ -v
```

(Tests are planned for parity validation against TypeScript fixtures.)

---

## Design Philosophy

- **Mirror, don't invent**: Every function maps directly to a TypeScript counterpart. If the TS engine does X, the Python function does X with the same math.
- **Parity first**: We verify Python outputs against TS fixtures before declaring a module complete.
- **Notebooks for exploration, modules for reuse**: Heavy computation lives in `.py` files; notebooks orchestrate and visualize.
