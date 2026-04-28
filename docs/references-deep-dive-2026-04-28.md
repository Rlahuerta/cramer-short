# References Deep Dive — Forecast & Prediction Improvement Ideas

**Date:** 2026-04-28
**Reviewer:** GitHub Copilot CLI
**Scope:** Systematic mining of `references/` (47 PDFs across 8 topic folders) for ideas that could materially improve the project's forecasting/prediction behaviour. Prior reports (`forecast-improvement-review-2026-04-28.md`, `markov-forecast-review-2026-04-28.md`, `polymarket-prediction-improvements-research-2026-07.md`) covered Q→P, jump-diffusion, MSM, regime-switching basics, whale detection and Phase 1/2/3 polymarket improvements; this document only surfaces **ideas not already implemented or queued**.

> **Method.** Three parallel `pdftotext`-driven explore passes covered (a) `markov-probability/` (11 papers), (b) `financial-forecasting/` + `derivatives-risk/` + `sequence-models/` + `optimization/` + `agentic-workflows/` + top-level Risks/SSRN papers (10 papers), (c) `llm-forecasting/` + `sentiment-analysis/` + the BetaHMM Polymarket paper (9 papers). Textbook PDFs (4) skimmed by TOC only.

---

## 0. TL;DR — Top 10 ranked

| # | Idea | Source | Where it slots in | Effort | Expected uplift |
|---|------|--------|-------------------|--------|-----------------|
| 1 | **Beta-HMM emissions for Polymarket** (bounded [0,1] prices) | Voigt 2025, *BetaHMMpolymarket* | `src/tools/finance/hmm.ts`, new `beta-hmm.ts` | M | **High** — current Gaussian emissions can emit invalid prices on Polymarket; Beta is the natively-correct family |
| 2 | **Logit jump-diffusion w/ risk-neutral drift constraint** for Polymarket trajectories | *Toward Black–Scholes for Prediction Markets*, arXiv:2510.15205 | `src/tools/finance/markov-distribution.ts`, `rnd-integration.ts` | H | **High** — gives a true martingale-consistent dynamic on probabilities, not just a Q→P static shift |
| 3 | **Domain × horizon recalibration intercepts** (αᵢ + βᵢ × log T) on raw Polymarket prices | *Decomposing Crowd Wisdom*, arXiv:2602.19520 (292M trades) | new `src/tools/finance/calibration-offsets.ts` upstream of `transformQToP` | M | **High** — paper attributes 87.3% of calibration variance to these terms; politics ≈ +0.15 underconfidence intercept |
| 4 | **Convergence-time as a confidence multiplier** on Polymarket-derived signals | Voigt 2025, *BetaHMMpolymarket* | `src/tools/finance/polymarket-forecast.ts` (already exists) | L–M | **Med-High** — fast-converging markets ⇒ boost forecast confidence; slow ⇒ damp it; ~free signal |
| 5 | **Range-before-point prompt scaffolding** for every LLM-emitted forecast | Lu 2025, arXiv:2507.04562 | `src/agent/prompts.ts`, skill prompts | L | **Med-High** — paper reports ~5–7 pp Brier improvement; trivial to ship |
| 6 | **Scale-mixture (Gaussian-Inverse-Gamma) HMM emissions** for endogenous fat tails | Taleb & Cirillo 2025, *risks-13-00247* | `hmm.ts` (extend emission family) | H | **High** — replaces ad-hoc Student-t with a principled errors-on-errors model; resolves crypto tail underestimation |
| 7 | **Trade-size-stratified transition-matrix nudging** (whale matrix vs retail matrix) | *Decomposing Crowd Wisdom* + Voigt | `markov-distribution.ts::nudgeTransitionMatrix` | M | **Med-High** — we *detect* whales already but don't weight nudges by trade size |
| 8 | **Hurst exponent H as live momentum/mean-reversion regime dial** | Bloch 2016, *ssrn-2715517* | new observable in `markov-distribution.ts::computeRegimeUpRates` | M | **Med** — H>0.5 ⇒ bias toward momentum-state persistence; H<0.5 ⇒ shorten regime half-life |
| 9 | **Entity-aware sentiment with weighted ensemble + exponential decay** | FUNNEL (s42521-025-00162-3) + Davidovic (jrfm-18-00412) | `src/tools/finance/social-sentiment.ts` (or new `entity-sentiment.ts`) | L–M | **Med** — the current pipeline treats headlines as scalar sentiment; entity-level + half-life massively reduces noise |
| 10 | **Multi-criterion HMM model selection (AIC + BIC + HQC + CAIC)** with refit cadence | 2310.03775, ijfs-06-00036 | `src/tools/finance/hmm.ts` | L | **Low-Med** — guards against silently picking wrong K when crypto regimes split |

The remaining ideas (Sections 5+) are useful but lower priority — recorded so we don't re-discover them later.

---

## 1. Idea #1 — Beta-HMM for Polymarket (TOP PRIORITY)

**Source:** Voigt (2025), *Predicting Prediction Markets: A Beta-Hidden Markov Modeling Approach* (`references/prediction-markets/BetaHMMpolymarket-18-1.pdf`).

### Why it matters
Polymarket prices live in `[0, 1]`. Our HMM uses Gaussian (and optionally Student-t) emissions, which:
- Have unbounded support, so a sample can violate the probability constraint and silently corrupt downstream Bayes updates.
- Can't represent the U-shape or J-shape that prediction-market price distributions take on near resolution.

The Beta family is the natural conjugate for `[0,1]` data. Voigt's groupwise variant additionally models the constraint that, across outcomes of the same multi-outcome market, prices sum to ≈1.

### Reported numbers
- 89.3% classification accuracy on resolved Polymarket contracts.
- $0.101 expected per-trade profit at simulated execution.
- Asymmetric convergence times: NO contracts converge ~40% faster than YES — exploitable execution edge.

### Integration sketch (TS)
```ts
// src/tools/finance/beta-hmm.ts (new)
export interface BetaEmission { alpha: number; beta: number }

export function betaPdf(x: number, a: BetaEmission): number {
  if (x <= 0 || x >= 1) return 0;
  // log-space for stability; uses lgamma helper already in the repo
  const logB = lgamma(a.alpha) + lgamma(a.beta) - lgamma(a.alpha + a.beta);
  return Math.exp((a.alpha - 1) * Math.log(x) + (a.beta - 1) * Math.log(1 - x) - logB);
}

// M-step: method-of-moments fit α,β from per-state weighted samples
export function fitBetaMoM(weights: number[], samples: number[]): BetaEmission { /* … */ }
```
Then plumb `EmissionFamily = 'gaussian' | 'student-t' | 'beta'` through `hmm.ts` and switch automatically when the input series is bounded in `[0,1]`.

### TDD checklist
- [ ] `beta-hmm.test.ts`: pdf integrates to ~1 over `(0,1)`.
- [ ] MoM recovers known α=2, β=5 from a sampled series within tolerance.
- [ ] Baum-Welch on a synthetic 2-regime Beta sequence (α₁=2,β₁=8 vs α₂=8,β₂=2) recovers regime labels at >85% accuracy.
- [ ] Parity test against Python mirror in `research/models/`.

---

## 2. Idea #2 — Logit Jump-Diffusion with Risk-Neutral Drift

**Source:** *Toward Black–Scholes for Prediction Markets* (arXiv:2510.15205v1, Oct 2025).

### Why it matters
Idea #4 in our prior plan applied a static Girsanov shift `transformQToP`. That correctly translates *one snapshot* but the *evolution* of the Polymarket price under our trajectory MC is still ad hoc. The 2510.15205 paper derives the prediction-market analogue of Black–Scholes:

1. Work in log-odds space `x = log(p / (1-p))` so the support is unbounded.
2. Decompose `dx = μ(t,x)dt + σ(t,x)dW + J·dN` into diffusion + Poisson jumps.
3. Constrain the drift `μ` to be the unique value that makes `p` a martingale under the chosen measure (the prediction-market analogue of the Black–Scholes drift = r constraint).

This gives us a fully-internally-consistent dynamic for "what's the distribution of the Polymarket price over the next 7 days" — far stronger than today's "shift the snapshot once, then ignore time."

### How it composes with what we already have
- Reuse `transformQToP` for the *initial-condition* shift.
- Replace the diffusion in `computeTrajectory` (when the underlying is a Polymarket price) with a logit-space SDE and the martingale-constrained drift.
- Keep the Polymarket-informed jump intensity `λ` from our Phase B4 work, but feed it into a Poisson process *in logit space*.

### Heteroskedastic Kalman pre-filter
The same paper recommends a heteroskedastic Kalman pre-filter (regime-dependent measurement variance) before EM jump-diffusion separation. This gives a clean `μ̂(t)`, `σ̂(t)` decomposition and dramatically reduces false jump detections caused by microstructure noise — useful for Polymarket where bid/ask jitter at low liquidity is huge.

### Scheduled-jump calendar
Encode known event windows (FOMC, earnings, election dates, scheduled hard forks) as Gaussian kernel boosts to `λ(t)`. Today our `λ` is uniform across the trajectory horizon; a calendar makes news jumps land on the right day.

---

## 3. Idea #3 — Domain × Horizon Recalibration of Polymarket Prices

**Source:** *Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets* (arXiv:2602.19520, Feb 2026; 292M trades on Kalshi + Polymarket).

### What the paper found (and we currently ignore)
Bayesian hierarchical decomposition explains **87.3% of calibration variance** by three terms:

| Term | What it is | Typical magnitude |
|------|-----------|-------------------|
| αᵢ (domain intercept) | Politics: **+0.15 underconfidence**; Sports: ~0; Crypto: ~+0.05 | up to ±0.20 in logit |
| βᵢ × log(T) (horizon × domain interaction) | Calibration slope rises from 0.99 at 1h to 1.32 at 1mo+ | growing miscalibration with horizon |
| γ (trade-size effect) | Kalshi-specific: large trades carry stronger signal | ±0.10 in logit |

**Implication.** Our `transformQToP` does an *asset-only* shift via `(μ-r)/σ`. It misses the *domain* and *horizon* effects entirely. For political-event Polymarket markets at 30-day horizon, the calibration drift can be ≥3× the Sharpe-driven shift.

### Implementation
```ts
// src/tools/finance/calibration-offsets.ts (new)
export interface CalibrationOffset { alpha: number; betaPerLogT: number }

export const DOMAIN_OFFSETS: Record<Domain, CalibrationOffset> = {
  politics: { alpha: 0.15, betaPerLogT: 0.05 },
  sports:   { alpha: 0.00, betaPerLogT: 0.00 },
  crypto:   { alpha: 0.05, betaPerLogT: 0.03 },
  macro:    { alpha: 0.02, betaPerLogT: 0.04 },
  unknown:  { alpha: 0.00, betaPerLogT: 0.00 },
};

export function recalibratePolymarketPrice(
  qProb: number,
  domain: Domain,
  daysToExpiry: number,
): number {
  const { alpha, betaPerLogT } = DOMAIN_OFFSETS[domain];
  const T = Math.max(daysToExpiry, 1) / 365;
  const slope = 1 + betaPerLogT * Math.log1p(daysToExpiry);
  const z = norm.ppf(qProb);
  return norm.cdf(slope * z + alpha);
}
```
Apply *upstream* of `transformQToP`. The resulting pipeline is:

```
raw polymarket price
  → recalibratePolymarketPrice (domain × horizon)
  → transformQToP (Sharpe-driven Girsanov)
  → fitLognormalFromStrikes / nudgeTransitionMatrix
```

The domain tag is already inferable from the market title via cheap LLM classification, and we cache it in `.cramer-short/api-routing.json` style.

---

## 4. Idea #4 — Convergence-Time Confidence Signal

**Source:** Voigt 2025 (Polymarket Beta-HMM paper).

### Definition
For each Polymarket contract, compute the time elapsed from market open to the first crossing of `p > 1 − ε` or `p < ε` (suggested ε = 0.05). Call that `convTime`.

- `convTime ≤ 7d` ⇒ market converged fast ⇒ high consensus ⇒ **boost** forecast confidence by ~+15%.
- `convTime ≥ 30d` ⇒ persistent uncertainty ⇒ **damp** confidence by ~−10%.
- YES vs NO asymmetry: NO converges ~40% faster (Voigt) → tighter exit timing on NO positions.

### Where it slots in
The `polymarket_forecast` tool already returns confidence bands; multiply them by the convergence-derived factor before final emission. Implementation is ~30 LoC plus a unit test on a 14-day synthetic price path.

---

## 5. Idea #5 — Range-Before-Point Prompt Scaffolding

**Source:** Lu (2025), *Evaluating LLMs on Real-World Forecasting Against Expert Forecasters* (arXiv:2507.04562).

### Pattern
Two-stage prompting on every forecasting question:

1. **Range stage.** Force the model to emit a confidence interval first (e.g., "My P(YES) is somewhere in 30%–55%, mostly because X, Y, Z").
2. **Point stage.** Constrained to the prior interval, request a single point probability.

### Why
Lu shows ~5–7 pp Brier-score improvement vs direct point prompts on Metaculus, across o3 / Claude / DeepSeek-R1. The mechanism is well-understood: LLMs are systematically overconfident at point prompts; forcing a range first calibrates the prior.

### Where to add it
- `src/agent/prompts.ts` — augment the "answer" template for forecasting skills.
- `src/skills/probability_assessment.md` — the existing skill is the obvious place to make this mandatory.
- Re-use the existing `combinedProbability is in 1–99% range` E2E assertion as a guardrail.

This is the cheapest high-impact item in the document.

---

## 6. Idea #6 — Scale-Mixture HMM (Errors-on-Errors)

**Source:** Taleb & Cirillo (Risks 13(12):247, Dec 2025), *The Regress of Uncertainty and the Forecasting Paradox*.

### Theorem (informal)
Any thin-tailed baseline becomes heavy-tailed once you compound iterated parameter uncertainty (uncertainty about σ̂, uncertainty about the uncertainty about σ̂, …). This is the *forecasting paradox*: the future tails are necessarily fatter than the past tails.

### Consequence for our HMM
Our Gaussian emissions assume σ is known per state. A scale-mixture (Gaussian-Inverse-Gamma) emission marginalises over σ̂ and emits Student-t **without us picking ν by hand** — ν falls out of the posterior on σ̂. This:
- Removes the Student-t `df = 6` ad-hoc constant in `hmm.ts`.
- Makes tail thickness adapt to per-regime sample size: small-sample regimes auto-fatten.
- Naturally generalises to the multifractal MSM track in `forecast-improvement-review-2026-04-28.md`.

### Cost
Conjugate updates are closed-form, but Baum-Welch needs the M-step over `(μ, σ̂_shape, σ̂_scale)`. A research spike in `research/spikes/scale_mixture_hmm/` first, then a TS port if the spike beats Student-t on the BTC 14d backtest.

---

## 7. Idea #7 — Trade-Size-Stratified Transition Nudging

**Source:** *Decomposing Crowd Wisdom* + Voigt.

### Today
`nudgeTransitionMatrix` blends transition probabilities with Polymarket-derived bull/bear/sideways weights. Whale detection exists separately as a *flag* but does not weight the nudge.

### Improvement
Maintain *two* nudge matrices: `M_whale` and `M_retail`, weighted by per-bucket trade-share over the rolling 14-day window. Then:
```
M_total = w_whale · M_whale + (1 − w_whale) · M_retail
```
where `w_whale` is the cumulative dollar share of trades over the configurable whale threshold (already in `polymarket.ts`). The 2602.19520 paper finds large-trade prices encode stronger private info on Kalshi; this generalises for Polymarket where whale dominance is even higher.

---

## 8. Idea #8 — Hurst Exponent as a Live Regime Dial

**Source:** Bloch (2016), *Quantitative Volatility Trading* (`references/ssrn-2715517.pdf`).

Compute rolling Hurst exponent `H` (e.g., R/S analysis or DFA over a 252-period window). Use it as an extra observable in the HMM:
- `H > 0.55` ⇒ trend-persistent regime ⇒ extend dwell-time in current state by a soft prior.
- `H < 0.45` ⇒ mean-reverting regime ⇒ shorten expected dwell-time.
- `0.45 ≤ H ≤ 0.55` ⇒ near-Brownian ⇒ no adjustment.

This is cheap and complements the planned MSM work: H gives a *single-scalar* hint while MSM gives the *multi-scale decomposition*. They are not redundant.

---

## 9. Idea #9 — Entity-Aware Sentiment + Decay

**Sources:** Davidovic & McCleary (jrfm-18-00412), Nordansjö et al. (s42521-025-00162-3, *FUNNEL*).

### Two findings, one fix
- Sentiment alone is ≈50% (random) on directional prediction; sentiment + IV + volume reaches ≈70%. ⇒ **Gate sentiment by volatility**: only weight it when VIX or its crypto-equivalent is elevated.
- Predictive power is short-lived (hours to days). ⇒ Apply exponential decay `s_t · exp(−λ Δt)` with λ ≈ 0.5/day for equities, λ ≈ 2/day for crypto.
- FUNNEL: NER-based per-entity attribution lifts label accuracy from ≈88% to ≈97%. ⇒ Replace headline-level sentiment with entity-level via spaCy NER (or a `social_sentiment` LLM call we already have).

### Code changes
- New `src/tools/finance/entity-sentiment.ts` wrapping `social-sentiment.ts` with NER + decay + volatility gating.
- Memory schema bump to store `(entity, sentiment, ts)` triples instead of `(headline, sentiment)`.

---

## 10. Idea #10 — Multi-Criterion HMM Model Selection

**Sources:** 2310.03775, ijfs-06-00036.

Today the HMM K (number of regimes) is mostly fixed at 3. Both papers recommend selecting K each refit by a **majority vote across AIC, BIC, HQC, CAIC** rather than any single criterion. This is ~50 LoC in `hmm.ts` and prevents silently choosing the wrong K when crypto markets split into 4 regimes (e.g., bull, bear, chop, blow-off).

---

## 11. Lower-priority ideas (recorded for later)

These are useful but are either:
- Already covered by an in-flight initiative (MSM, Phase 1/2/3 polymarket improvements).
- Or specific to data sources we don't yet have (e.g., LOB depth, IV surface history).

| Idea | Source | Why deferred |
|------|--------|--------------|
| Hierarchical Bayesian multi-venue synthesis (Kolmogorov + WinBUGS) | welton2005 | High complexity; requires Kalshi data feed we don't have today |
| HJB-optimal market-making spread | mathematics-13-00778 | Out of scope (we're a research agent, not a market-maker) |
| Order-imbalance HMM | 87624-submission2017 | Requires LOB feed we don't ingest |
| Hitting-time matrices for risk budgeting | Kumar 2023 | Useful but requires a portfolio/sizing layer we don't surface |
| Stochastic funding-rate (r₁ − r₂) as 3rd HMM dimension | El Hassan/Maddah/Taleb 2026 | Crypto-perp specific; revisit when we add perpetuals coverage |
| Citation-graph traversal for multi-hop thesis discovery | FutureHouse (Skarlinski 2024) | Compelling but agent-architecture change; spike separately |
| Stochastic-dominance factor ranking | Chung et al. 2015 | Equities multi-factor; not core to the BTC/SPY short-horizon track |
| Greeks-as-regime-filter for option strategies | Li & Le Floch 2024 | Out of current product scope |
| LLM model-bias correction (US vs CN models) | Cao/Wang/Yi 2026 | Useful if we add CN-model support; not today |
| OHLC multi-observation emissions | 2310.03775 | Easy win; bundle into Beta-HMM work above |

---

## 12. Cross-cutting themes

1. **Polymarket prices need three corrections, not one.** Today we apply (a) Girsanov via `transformQToP`. The literature says we additionally need (b) domain × horizon recalibration (Idea #3) and (c) trade-size stratification (Idea #7). Together they explain >87% of measured calibration variance.
2. **Use the right family for the right support.** Gaussian on `[0,1]` is wrong; Beta is right (Idea #1). Gaussian with fixed σ on heavy-tailed assets is wrong; scale-mixture with posterior σ̂ is right (Idea #6). The `[0,1]` and the scale-mixture work are independent and can ship in parallel.
3. **Cheap LLM scaffolding > expensive model surgery.** Idea #5 (range-before-point) is the highest-ROI item in the doc by far: ~5–7 pp Brier improvement for ~50 LoC.
4. **Time matters in two ways.** (a) Polymarket signals decay (Idea #9); (b) calibration of those signals also depends on horizon (Idea #3). Both must be encoded.

---

## 13. Suggested execution plan

> Reflect into the `todos` table once this doc is approved.

**Wave 1 (low-friction, high-ROI; ~1 week each):**
- W1a: **Range-before-point prompt scaffolding** (Idea #5) — touches `prompts.ts` + `probability_assessment` skill. Verify against the existing `combined probability is in 1–99% range` E2E assertion.
- W1b: **Convergence-time confidence multiplier** (Idea #4) — pure post-processing on `polymarket_forecast`.
- W1c: **Multi-criterion HMM K selection** (Idea #10) — `hmm.ts` only.

**Wave 2 (medium effort, structural; ~2 weeks):**
- W2a: **Domain × horizon recalibration** (Idea #3) — new `calibration-offsets.ts`, mirror in `research/models/`.
- W2b: **Trade-size-stratified nudging** (Idea #7) — extend `nudgeTransitionMatrix`.
- W2c: **Entity-aware sentiment + decay** (Idea #9) — new `entity-sentiment.ts`.

**Wave 3 (architectural, TDD-first):**
- W3a: **Beta-HMM** (Idea #1) — research spike → TS+Python parity.
- W3b: **Logit jump-diffusion + scheduled-jump calendar** (Idea #2) — extends our existing jump-diffusion path.
- W3c: **Hurst exponent regime dial** (Idea #8).

**Wave 4 (R&D track):**
- W4: **Scale-mixture HMM** (Idea #6) — Python spike in `research/spikes/scale_mixture_hmm/` first; only port to TS if it beats Student-t on the BTC 14d + SPY 30d backtests.

Each wave should keep TDD discipline (per the user's standing instruction): failing test first, parity test against `research/`, then ship. Backtest-driven verification on the existing `phase4`, `phase5` and `markov-backtest` integration suites is mandatory before each wave merges.

---

## 14. Reference inventory used

```
references/prediction-markets/BetaHMMpolymarket-18-1.pdf          # Voigt 2025 → Idea #1, #4
references/markov-probability/2510.15205v1.pdf                    # Logit JD → Idea #2
references/markov-probability/2602.19520v1.pdf                    # Crowd-wisdom calibration → Idea #3, #7
references/llm-forecasting/2507.04562v3.pdf                       # Lu 2025 → Idea #5
references/risks-13-00247-v2.pdf                                  # Taleb & Cirillo → Idea #6
references/ssrn-2715517.pdf                                       # Bloch volatility → Idea #8
references/sentiment-analysis/jrfm-18-00412.pdf                   # Davidovic → Idea #9
references/sentiment-analysis/s42521-025-00162-3.pdf              # FUNNEL → Idea #9
references/markov-probability/2310.03775v2.pdf                    # 4-state HMM, model selection → Idea #10
references/markov-probability/ijfs-06-00036-v2.pdf                # HMM stock trading → Idea #10
references/markov-probability/Kumar_Amer_MEng_2023.pdf            # LSTM+MC hybrid (deferred)
references/markov-probability/welton2005.pdf                      # Bayesian multi-venue (deferred)
references/markov-probability/risks-09-00037.pdf                  # Calibration of intensities (deferred)
references/markov-probability/mathematics-13-00778-v2.pdf         # Birth-death LOB (deferred)
references/derivatives-risk/2602.14350v1.pdf                      # American options hidden risks (deferred)
references/financial-forecasting/2601.11958v1.pdf                 # Agentic nowcasting (deferred)
references/financial-forecasting/1-s2.0-S1566253524005335-main.pdf # NLP-finance survey (deferred)
references/llm-forecasting/2409.13740v2.pdf                       # FutureHouse (deferred)
references/llm-forecasting/Kong_et_al_2024_Large_language_models.pdf # LLM survey (deferred)
references/llm-forecasting/Financial-Statement-Analysis-with-Large-Language-Models.pdf # CoT fundamentals (deferred)
```

Textbooks and out-of-scope papers omitted.
