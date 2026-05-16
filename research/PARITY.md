# TypeScript ↔ Python Parity Manifest

Auto-generated from codebase audit. Lists every TypeScript module in `src/`
with a known or intended Python mirror in `research/`, its parity-test status,
and known divergences.

**Last updated**: 2026-05-15
**Convention**: A TS file is "mirrored" when it carries a `Mirrors research/...`
JSDoc tag AND its math is verified against the Python implementation.

---

## Status Legend

| Icon | Meaning |
|------|---------|
| ✅ | TS docstring mirror tag + active `.parity.test.ts` + Python parity test |
| 🟡 | TS docstring mirror tag, but no TS runtime parity test (Python side only) |
| ❌ | Python parity test exists, but TS side missing docstring marker and TS parity test |
| 🔸 | Partial/asymmetric mirror — documented divergence |
| ⬜ | No Python mirror expected (tool wrapper, utility, or presentation-layer code) |

---

## Models Layer

| TS File | Python File | Status | TS Parity Test | Python Test |
|---------|-------------|--------|----------------|-------------|
| `src/tools/finance/jump-diffusion.ts` | `research/models/jump_diffusion.py` | ✅ | `jump-diffusion.parity.test.ts` | `test_jump_diffusion.py` |
| `src/tools/finance/rnd-integration.ts` | `research/models/rnd.py` | ✅ | `rnd-integration.parity.test.ts` | `test_rnd.py` |
| `src/tools/finance/logit-jump-diffusion.ts` | `research/models/logit_jump_diffusion.py` | 🟡 | — | `test_logit_jump_diffusion_parity.py` |
| `src/tools/finance/beta-hmm.ts` | `research/models/beta_hmm.py` | 🟡 | — | `test_beta_hmm_parity.py` |
| `src/tools/finance/hmm.ts` | `research/models/hmm.py` | 🟡 | — | `test_hmm_parity.py` |
| `src/tools/finance/conformal.ts` | `research/models/conformal.py` | 🟡 | — | `test_conformal_parity.py` |
| `src/tools/finance/hawkes.ts` | `research/models/hawkes.py` | 🟡 | — | `test_hawkes_parity.py` |
| `src/tools/finance/garch-scales.ts` | `research/models/garch_scales.py` | 🟡 | — | `test_garch_scales_parity.py` |
| `src/tools/finance/calibration-offsets.ts` | `research/models/calibration_offsets.py` | 🟡 | — | `test_calibration_offsets_parity.py` |
| `src/tools/finance/transition-entropy.ts` | `research/models/transition_entropy.py` | 🟡 | — | `test_transition_entropy_parity.py` |
| `src/tools/finance/markov-distribution/regime.ts` | `research/models/markov.py` | 🟡 | — | `test_markov_parity.py` |
| `src/tools/finance/markov-distribution/transition.ts` | `research/models/markov.py` | 🟡 | — | `test_markov_parity.py` |
| `src/tools/finance/markov-distribution/confidence-intervals.ts` | `research/models/markov.py` + `research/models/trajectory.py` | 🟡 | — | `test_markov_parity.py` + `test_trajectory_parity.py` |
| `src/utils/finance/ensemble.ts` | `research/models/ensemble.py` | 🟡 | — | `test_ensemble_parity.py` + `test_ensemble_p1_parity.py` |
| `src/utils/finance/vol-regime.ts` | `research/models/vol_regime.py` | 🟡 | — | `test_vol_regime_parity.py` |
| `src/utils/finance/adwin.ts` | `research/models/adwin.py` | 🟡 | — | `test_adwin_parity.py` |

### Divergent Mirrors

| TS File | Python File | Divergence |
|---------|-------------|------------|
| `src/utils/finance/garch.ts` | `research/models/garch.py` | 🔸 TS uses fixed-prior moment-matching shortcut; Python uses full MLE (golden-section). TS header explicitly says "not a mirror — for full MLE see garch.py". |
| `src/tools/finance/markov-distribution.ts` | `research/models/markov.py` | 🔸 TS (5,006 lines) has extensive forecast-lab parameter defaults, anchor inspection, live policies, and action signals that may not be mirrored. Python (775 lines) covers core regime logic, short-horizon live policies, and validation gates. |

## Utils & Runtime Layer

| TS File | Python File | Status | Notes |
|---------|-------------|--------|-------|
| `src/tools/finance/forecast-lab-runtime-defaults.ts` | `research/utils/forecast_lab_runtime_defaults.py` | 🟡 | Runtime default registry with asset-scoped resolution |
| `src/tools/finance/regime-calibrator.ts` | `research/utils/regime_calibrator.py` | 🟡 | Single-pass regime-conditional Platt recalibrator |
| `src/tools/finance/regime-calibrator-two-pass.ts` | `research/utils/regime_calibrator_two_pass.py` | 🟡 | Two-pass iterative Platt recalibrator |
| `src/tools/finance/kalshi-vol-signals.ts` | `research/utils/kalshi_vol_signals.py` | 🟡 | Kalshi macro event volatility signals |
| — | `research/utils/anchor_trust.py` | 🟡 | Anchor trust evaluation (no standalone TS file; likely inline in markov-distribution) |
| — | `research/utils/calibration.py` | 🟡 | YES-bias correction v1/v2 (mirrored inline in `ensemble.ts`) |

## Backtest Layer

| TS File | Python File | Status | Notes |
|---------|-------------|--------|-------|
| `src/tools/finance/backtest/baselines.ts` | `research/backtest/baselines.py` | 🟡 | Naive baselines (coin-flip, last-period) |
| `src/tools/finance/backtest/metrics.ts` | `research/backtest/metrics.py` | 🟡 | Brier, directional accuracy, CI coverage, calibration tables, CRPS |
| `src/tools/finance/backtest/walk-forward.ts` | `research/backtest/walk_forward.py` | 🟡 | Walk-forward harness with HMM/GARCH/entropy options |

## Data Layer (TS-only tool wrappers)

| TS File | Python File | Notes |
|---------|-------------|-------|
| `src/tools/finance/polymarket.ts` | `research/data/polymarket.py` | 🟡 Both implement Gamma API + tag-slug search. TS has additional LangChain tool wrapper. |
| `src/tools/finance/polymarket-clob.ts` | `research/data/polymarket_clob.py` | 🟡 CLOB price-history parsing and velocity |
| `src/tools/finance/metaforecast.ts` | `research/data/metaforecast.py` | 🟡 Metaforecast.org cross-platform validation |
| `src/tools/finance/get-market-data.ts` | `research/data/prices.py` | 🟡 TS wraps `fetch_historical_prices` in LangChain tool. Python is standalone fetcher. |

## Stubs / Unimplemented

| Python File | Status |
|-------------|--------|
| `research/models/crypto_native_peers.py` | ❌ STUB — raises `CryptoPeerLoaderUnavailable`. No TS equivalent found in `src/tools/finance/crypto-native-peers.ts`. |

## No Mirror Expected

These TS modules are data-fetching wrappers, presentation utilities, or tool-definition files with no Python equivalent. A Python mirror is neither intended nor expected.

| TS File | Category |
|---------|----------|
| `src/tools/finance/get-financials.ts` | LangChain tool wrapper |
| `src/tools/finance/read-filings.ts` | LangChain tool wrapper |
| `src/tools/finance/fixed-income.ts` | LangChain tool wrapper |
| `src/tools/finance/options.ts` | LangChain tool wrapper |
| `src/tools/finance/bitmex.ts` | LangChain tool wrapper |
| `src/tools/finance/robinhood.ts`, `robinhood-client.ts` | LangChain tool wrapper |
| `src/tools/finance/fundamentals.ts` | LangChain tool wrapper |
| `src/tools/finance/screen-stocks.ts` | LangChain tool wrapper |
| `src/tools/finance/portfolio-risk.ts` | LangChain tool wrapper |
| `src/tools/finance/wacc-inputs.ts` | LangChain tool wrapper |
| `src/tools/finance/stock-price.ts` | LangChain tool wrapper |
| `src/tools/finance/onchain-crypto.ts` | LangChain tool wrapper |
| `src/tools/finance/social-sentiment.ts` | LangChain tool wrapper |
| `src/tools/finance/earnings-transcripts.ts` | LangChain tool wrapper |
| `src/tools/finance/filings.ts` | LangChain tool wrapper |
| `src/tools/forecast-lab-run.ts` | Tool factory + runner |
| `src/utils/finance/portfolio-stats.ts` | Portfolio math utility |
| `src/utils/finance/number-format.ts` | Presentation formatting |
| `src/utils/finance/fmp-quota.ts` | Quota tracking utility |
| `src/utils/finance/wacc.ts` | WACC computation utility |
| `src/utils/finance/probability.ts` | Probability formatting |
| `src/utils/finance/cross-validate.ts` | Validation utility |
| `src/utils/finance/kswin.ts` | Drift detector (TS-only) |

---

## How to Add a New Mirror

1. Add `/** Mirrors \`research/models/your_module.py\`. */` to the TS file header.
2. Create `src/.../your_module.parity.test.ts` that imports `runPython` from `src/utils/finance/python-parity.ts` and asserts numerical equality.
3. Create `research/tests/test_your_module_parity.py` with the Python side.
4. Update this manifest.

## How to Promote from 🟡 to ✅

1. Create the `.parity.test.ts` file.
2. Add the CI step or test runner command that executes parity tests.
3. File a follow-up PR updating the status in this manifest.
