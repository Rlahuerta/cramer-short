# Design: Polymarket RND → Markov Pipeline Integration

## 1. Goal

Augment the existing Markov trajectory pipeline with forward-looking regime priors extracted from Polymarket strike markets. Extract a Risk-Neutral Density (RND) from a chain of "Will asset be above $K?" contracts, transform Q-measure probabilities to P-measure via Market Price of Risk, and inject them into both the terminal regime distribution and the transition matrix.

## 2. Theory

### 2.1 RND Extraction from Digital Options
Polymarket binary contracts are digital options paying $1 if $S_T > K$. A chain of strikes gives us survival probabilities $P^Q(S_T > K_i)$. The CDF is $F(K) = 1 - P^Q(S_T > K)$.

**Critical design choice:** Instead of differentiating a spline (which produces negative RND when Polymarket quotes violate no-arbitrage, e.g. strike $95k at 42% when $90k is at 40%), we fit a **parametric Log-Normal distribution** to the transformed physical survival curve via Least Squares. This guarantees a valid, positive probability density and gracefully smooths over noisy/disordered quotes.

For richer strike chains ($\ge 4$ strikes), a **2-Log-Normal mixture** can be fitted. For sparse chains (2–3 strikes), a single Log-Normal suffices.

The fitted Log-Normal parameters $(\mu_{\ln}, \sigma_{\ln})$ give us the closed-form physical CDF $F(K) = \Phi\left(\frac{\ln K - \mu_{\ln}}{\sigma_{\ln}}\right)$ and PDF $f(K) = \frac{1}{K \sigma_{\ln}} \phi\left(\frac{\ln K - \mu_{\ln}}{\sigma_{\ln}}\right)$.

### 2.2 Q → P Measure Transformation
Polymarket prices are risk-neutral ($\mathbb{Q}$) because market makers hedge with options. The physical probability is recovered via:

$$ \text{Prob}^\mathbb{P}(S_T > K) = \Phi\left( \Phi^{-1}\left( \text{Prob}^\mathbb{Q}(S_T > K) \right) + \frac{\mu - r_f}{\sigma} \sqrt{T} \right) $$

Where $\lambda = (\mu - r_f) / \sigma$ is the Market Price of Risk (Sharpe ratio).

## 3. Architecture

```
Polymarket API
  │ fetchPolymarketAnchorMarkets(ticker, horizon, strike_pattern="above")
  ▼
PriceThreshold[] (strikes + P^Q(S_T > K))
  │
  ├──→ Q→P Transformer ──→ Physical survival probabilities
  │                        [P^P(S_T > K_1), ..., P^P(S_T > K_n)]
  │                              │
  │                              ▼
  │                     Fit Log-Normal CDF via Least Squares
  │                     (single LN for 2–3 strikes, 2-LN mixture for ≥4)
  │                              │
  │                              ▼
  │                     Map fitted CDF/PDF to regime buckets
  │                     [bull, bear, sideways] probabilities
  │                              │
  │                              ├──→ start_mixture override
  │                              │    compute_horizon_drift_vol(start_mixture=...)
  │                              │
  │                              └──→ Transition matrix nudge
  │                                   nudge_transition_matrix(P, target_dist, horizon)
```

## 4. New Modules

### 4.1 Python: `research/models/rnd.py`

```python
# Core functions (TDD targets)
def fit_lognormal_from_strikes(
    strikes: list[float],
    yes_prices: list[float],
    current_price: float,
) -> tuple[float, float]:
    """Fit a Log-Normal distribution to physical survival probabilities via Least Squares.
    Returns (mu_ln, sigma_ln) such that P(S_T > K) = 1 - Φ((ln K - mu_ln) / sigma_ln).
    Falls back to single-point drift estimate if < 2 strikes.
    """

def transform_q_to_p(
    q_prob: float,
    historical_drift: float,
    risk_free_rate: float,
    volatility: float,
    days_to_expiry: int,
) -> float:
    """Convert risk-neutral probability to physical probability."""

def lognormal_to_regime_probabilities(
    mu_ln: float,
    sigma_ln: float,
    current_price: float,
    bull_threshold: float = 0.01,
    bear_threshold: float = -0.01,
) -> dict[str, float]:
    """Integrate fitted Log-Normal over regime return buckets to get P(bull), P(bear), P(sideways)."""

def nudge_transition_matrix(
    P: np.ndarray,
    current_regime: str,
    target_terminal_dist: dict[str, float],
    horizon: int,
    quality_score: float,
) -> np.ndarray:
    """Nudge transition matrix so P^horizon from current_regime approximates target_terminal_dist.
    Nudge strength = 0.5 * (quality_score / 100), bounded to [0, 0.5].
    """
```

### 4.2 TypeScript: `src/tools/finance/rnd-integration.ts`

```typescript
// Core functions (parity with Python)
export function fitLognormalFromStrikes(
  strikes: number[],
  yesPrices: number[],
  currentPrice: number,
): { muLn: number; sigmaLn: number };

export function transformQToP(
  qProb: number,
  historicalDrift: number,
  riskFreeRate: number,
  volatility: number,
  daysToExpiry: number,
): number;

export function lognormalToRegimeProbabilities(
  muLn: number,
  sigmaLn: number,
  currentPrice: number,
  bullThreshold?: number,
  bearThreshold?: number,
): Record<string, number>;

export function nudgeTransitionMatrix(
  P: number[][],
  currentRegime: string,
  targetTerminalDist: Record<string, number>,
  horizon: number,
  qualityScore: number,
): number[][];
```

### 4.3 Integration Points

**Python:**
- `research/models/trajectory.py` — `compute_horizon_drift_vol()` already accepts `start_mixture`. Wire RND output here.
- `research/backtest/walk_forward.py` — Optional `use_rnd_priors` flag per backtest window.

**TypeScript:**
- `src/tools/finance/markov-distribution.ts` — After anchor extraction and before `computeTrajectory()`, call `applyRndPriors()` if sufficient strike markets exist.
- `src/utils/ensemble.ts` — Optional: use RND-derived regime probabilities as additional signal in ensemble blending.

## 5. Data Flow (End-to-End)

1. **Agent query:** "BTC 7-day forecast"
2. **Fetch anchors:** `fetchCandidatePolymarketAnchors("BTC-USD", 7)` → 5 markets with strikes [$95k, $100k, $105k, $110k, $115k]
3. **Extract thresholds:** `extractPriceThresholds()` → `{ strike: 100000, probability: 0.42, ... }`
4. **Q→P transform:** For each strike, call `transform_q_to_p()` using `mu=0.40`, `sigma=0.50`, `r_f=0.045`, `T=7/365`
5. **Fit distribution:** `fit_lognormal_from_strikes()` → fit Log-Normal CDF to physical survival probabilities via Least Squares (single LN for 2–3 strikes, 2-LN mixture for ≥4)
6. **Regime mapping:** `lognormal_to_regime_probabilities()` with thresholds ±1% → `{ bull: 0.45, bear: 0.30, sideways: 0.25 }`
7. **Inject into trajectory:**
   - `start_mixture = regime_probs` → terminal distribution override
   - `P_nudged = nudge_transition_matrix(P, current_regime, regime_probs, 7)` → path dynamics aligned
8. **Compute trajectory** with both overrides → Student-t MC paths, CI, p_up

## 6. Testing Strategy (TDD Parity)

### Phase 1: Python tests (RED)
- `test_transform_q_to_p_identity_when_lambda_zero` — When μ = r_f, P^P = P^Q
- `test_transform_q_to_p_increases_for_bullish` — Positive λ increases P(S_T > K) for K > S_0
- `test_transform_q_to_p_decreases_for_bearish` — Positive λ decreases P(S_T < K) for K < S_0
- `test_transform_q_to_p_clip_extremes` — Input 0.0001 or 0.9999 handled safely
- `test_fit_lognormal_recovers_known_params` — Fit to synthetic log-normal data → recovers (μ, σ) within 5%
- `test_fit_lognormal_handles_arbitrage_violations` — Non-monotonic quotes (higher strike > lower strike prob) still produce valid CDF
- `test_fit_lognormal_falls_back_single_point` — 1 strike → returns drift-based estimate
- `test_lognormal_to_regime_probabilities_sum_one` — Output probabilities sum to 1
- `test_lognormal_to_regime_probabilities_single_regime` — LN concentrated in one bucket → that regime ≈ 1
- `test_nudge_transition_matrix_preserves_rows` — Row sums still = 1
- `test_nudge_transition_matrix_terminal_match` — P^nudged^horizon ≈ target_dist
- `test_nudge_transition_matrix_strength_scales_with_quality` — q=100 → stronger nudge than q=20
- `test_nudge_transition_matrix_identity_when_target_matches` — If target already matches historical, P unchanged

### Phase 2: TypeScript tests (RED)
Mirror every Python test case in `src/tools/finance/rnd-integration.test.ts`.

### Phase 3: Parity tests (GREEN)
- `test_ts_py_parity_transform_q_to_p` — Same inputs → same outputs to 6 decimal places
- `test_ts_py_parity_rnd_to_regime` — Same strike chain → same regime probabilities
- `test_ts_py_parity_nudge_matrix` — Same P + target → same nudged matrix

### Phase 4: Integration tests
- `test_markov_distribution_with_rnd_priors` — End-to-end: BTC 7d forecast with RND injection produces finite trajectory
- `test_backtest_walk_forward_with_rnd` — walk_forward with `use_rnd_priors=True` runs without error

## 7. Parameters & Defaults

| Parameter | Default | Source |
|-----------|---------|--------|
| `risk_free_rate` | 0.045 | Hardcoded (US 3-month T-bill proxy) |
| `historical_drift` | `mu_obs` from `compute_horizon_drift_vol` | Runtime from regime stats |
| `volatility` | `sigma_eff` from `compute_horizon_drift_vol` | Runtime from regime stats |
| `bull_threshold` | +1% return | `current_price * 1.01` |
| `bear_threshold` | -1% return | `current_price * 0.99` |
| `nudge_strength` | `0.5 * (q / 100)` | Dynamic based on Polymarket quality score (max 0.5) |
| `min_strikes_for_rnd` | 2 | 2 strikes → single Log-Normal; ≥4 → 2-LN mixture |

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Too few Polymarket strikes for meaningful RND | Require ≥2 strikes for single Log-Normal; ≥4 for mixture; fall back to pure Markov |
| Polymarket arbitrage violations (e.g. higher strike priced higher) | Parametric Log-Normal fit smooths over violations gracefully; LSQ finds best-fit curve |
| Measure transformation blows up at extremes | Clip Q-probabilities to [0.001, 0.999] before probit |
| Historical drift unreliable for short windows | Use HMM override drift if available, else 3yr annualized |
| TypeScript/Python parity drift | Parity test suite runs on CI; golden file fixtures shared |
| Transition matrix nudge produces invalid rows | Renormalize rows after nudge; assert row sums ≈ 1 |

## 9. Files to Create / Modify

**Create:**
- `research/models/rnd.py` (Log-Normal fit + Q→P + regime mapping + matrix nudge)
- `research/tests/test_rnd.py` (Python TDD tests)
- `src/tools/finance/rnd-integration.ts` (TypeScript mirror)
- `src/tools/finance/rnd-integration.test.ts` (TypeScript TDD tests)
- `src/tools/finance/rnd-integration.parity.test.ts` (TS/Python cross-validation)

**Modify:**
- `research/models/trajectory.py` — Wire `start_mixture` from RND
- `research/models/markov.py` — Add `nudge_transition_matrix()`
- `research/backtest/walk_forward.py` — Add `use_rnd_priors` parameter
- `src/tools/finance/markov-distribution.ts` — Call RND integration after anchor extraction
