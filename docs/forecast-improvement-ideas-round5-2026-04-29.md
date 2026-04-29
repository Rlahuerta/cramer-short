# Forecast Improvement Ideas — Round 5 (2026-04-29)

**Date:** 2026-04-29  
**Predecessors:**
- `docs/forecast-improvement-ideas-round3-2026-04-28.md` (R3 brainstorm — 8 ideas, arXiv-sourced)
- `docs/forecast-improvement-ideas-round4-2026-04-28.md` (R4 brainstorm — 7 ideas, post W3R2)
- `docs/btc-multi-horizon-backtest-round4-2026-04-29.md` (R4 backtest results)

---

## 0. What the R4 backtest just taught us

Before listing new ideas, we must account for what the Round-4 walk-forward revealed.

### R4 results recap (Δ vs Round-2 improved+HA arm)

| Horizon | ΔdirAcc | ΔBrier | ΔmeanEdge | Result |
|--------:|--------:|-------:|----------:|--------|
| 1d | +0.005 | −0.001 | +0.0001 | ✅ small improvement |
| 2d | +0.005 | −0.002 | +0.0004 | ✅ small improvement |
| 3d | 0.000 | −0.001 | −0.0004 | — neutral |
| 7d | −0.005 | +0.002 | −0.0001 | ⚠️ slight regression |
| 14d | −0.006 | +0.006 | −0.0004 | ❌ regression |
| 30d | 0.000 | +0.007 | 0.0000 | ❌ Brier worse |

### Key lessons

1. **GARCH helps short horizons, hurts medium-long.** The uniform `[0.33, 3.0]` vol scalar works
   well at h=1–2d (recent vol matters) but over-widens CI at h=14–30d (GARCH reverts toward
   unconditional σ within ~5 days; beyond that it should be ignored, not amplified).

2. **Cross-asset Lasso showed virtually no effect.** Equity peers (SPY/GLD/QQQ) co-movement with
   BTC was near-zero during the 2024–2025 BTC bull run. The L1 regularisation correctly drove
   most coefficients to zero — but that means the feature is harmless, not helpful. Crypto-native
   peers (ETH, SOL, MSTR) would carry genuine co-movement signal.

3. **ADWIN + Hawkes remain byte-identical null on daily BTC.** Confirmed across both Round-2 and
   Round-4 walks. The wiring is correct; the daily-bar granularity is wrong for these detectors.

4. **Round-1 W2/W3 flags are still the dominant improvement** (+7.8pp dirAcc at h=14d,
   positive meanEdge at all horizons ≤ 14d). No R4 flag surpassed them.

5. **30d is still the bleeding edge.** Every arm still posts negative edge (−0.004) at h=30d.
   The information set (BTC alone + equity peers) is insufficient at this horizon.

---

## 1. TL;DR — top unimplemented features, ranked by EV × inverse cost

| # | Idea | Origin | Status | Effort | Expected EV |
|--:|------|--------|--------|--------|-------------|
| 1 | **Crypto-native cross-asset Lasso** (ETH/SOL/MSTR peers) | R4 Idea 2 (fix) | R4 done, wrong peers | S | 🔥🔥🔥 — fix one line; ETH has >0.8 rolling correlation with BTC at 30d |
| 2 | **Polymarket order-book depth weighting** | R4 Idea 6 | Not implemented | S | 🔥🔥🔥 — structural fix; removes thin-market noise from all horizon anchors |
| 3 | **Crypto unpredictability sanity baseline** | R3 Idea 8 | Not implemented | S | 🔥🔥 — mandatory guard before promoting any flag to default |
| 4 | **Wire conformalLow/High into forecast outputs** | todo `w3-2b` | Partial (PID exists, not surfaced) | S | 🔥🔥 — surfaces the existing PID conformal interval in the agent tool response |
| 5 | **Horizon-aware GARCH clamping / decay** | R4 backtest finding | Not implemented | S | 🔥🔥 — fix R4 regression at 14d+; GARCH should soft-decay toward 1.0 beyond h=5d |
| 6 | **Regime-Platt two-pass walk-forward** | R4 Idea 3 (fix) | Module done, backtest untested | M | 🔥🔥 — validates Platt recalibrator on real data; currently untested in any walk-forward |
| 7 | **Bayesian Model Averaging across forecast heads** | R4 Idea 7 | Not implemented | M | 🔥🔥 — self-tuning; removes manual flag selection; adapts per-horizon |
| 8 | **On-chain Polymarket trade direction** | R3 Idea 1 | Not implemented | M | 🔥🔥🔥 — fixes 59%-accuracy direction inference on public feed |
| 9 | **LASSO-PCA pooling across model variants** | R3 Idea 5 | Not implemented | M | 🔥 — different from cross-asset Lasso; pools {Markov, HMM, jump-diffusion} heads |
| 10 | **Intraday Hawkes evaluation** (research spike) | R4 Idea 5 | Not implemented | S | 🔥 — confirms or kills the daily-bar Hawkes path definitively |
| 11 | **Longshot premium sanity check for P3 calibration** | todo `w3-1b` | Not implemented | S | 🔥 — verifies the R3 P3 domain×horizon recalibration doesn't create longshot artifacts |
| 12 | **mfBm realised-vol forecast** | R3 Idea 6 | Not implemented | H | 🔥 — correct long-memory structure; Python spike only until it beats HAR |

---

## 2. Idea #1 — Crypto-native cross-asset Lasso ⭐ quick win

**What happened.** R4 implemented cross-asset Lasso with SPY/GLD/QQQ peers. The backtest
showed zero measurable effect. The root cause is structural: during BTC's 2024–2025 bull
run, equity co-movement was negligible (BTC decoupled from risk-on/risk-off). The Lasso
correctly killed these features — but that means no signal was injected.

**Fix.** Replace (or supplement) the equity peers with crypto-native ones: **ETH**, **SOL**,
**MSTR**. ETH/BTC rolling 30-day correlation is consistently 0.75–0.90. MSTR is a leveraged
BTC proxy that leads BTC on institutional flow. SOL captures altcoin rotation risk. These
peers should carry genuine drift bias at h=7–30d.

**Implementation.** The Lasso module is complete. The only change is the `crossAssetReturns`
map passed to `walkForward`. If ETH/SOL/MSTR closes are added to the fixture (or fetched
at query time), the wiring needs zero changes. Optionally add a `cryptoPeers` convenience
flag to the `WalkForwardConfig` that auto-selects crypto-native peer tickers.

**True-improvement gate.** Walk-forward R4 arm with ETH/SOL as peers. Lasso must produce
`nonZeroCoefCount ≥ 1` on at least 60% of walk-forward steps; dirAcc at h=30d must
improve ≥ 0.01 vs the current cross-asset-disabled arm.

**Effort:** Small — fixture data update + peer list change.

---

## 3. Idea #2 — Polymarket order-book depth weighting ⭐ structural fix

**Source.** R4 Idea 6. Documented in Round-4 brainstorm.

**Problem.** Current Polymarket integration weights each market anchor equally regardless of
liquidity. A market with $50 total depth gives the same mixture weight as one with $500K.
Thin markets are systematically biased toward extreme probabilities (the paper Dubach 2026
explicitly documents this as a spread-premium effect).

**Implementation.**
1. Pull `bid_size + ask_size` from the CLOB `/orderbook` endpoint (already called for
   velocity signal in W1 P1d).
2. Compute `depthUsd = (bid_size + ask_size) × current_price`.
3. Apply a weight scalar: `w = Math.min(depthUsd / 1000, 1.0)` — markets below $1K depth
   get fractional weight; markets above $1K are fully weighted.
4. Multiply the existing anchor's log-odds contribution by `w` before the mixture.
5. Add `depthUsd` to the anchor metadata so the agent tool can surface it.

**True-improvement gate.** On synthetic anchors with known depth disparity, the mixture
must converge toward the deep-market probability (not the shallow one). Real-data: Brier
improvement on Polymarket-anchored BTC h=30d predictions vs current equal-weight baseline.

**Effort:** Small (~60 lines). The CLOB call is already present.

---

## 4. Idea #3 — Crypto unpredictability sanity baseline ⭐ mandatory gate

**Source.** R3 Idea 8 (arXiv:2502.09079 — Puoti et al. 2025).

**Problem.** We have no systematic guard that any new flag actually beats a naïve no-change
baseline. The R4 backtest revealed GARCH *regresses* at h=14d — we would not have caught
this without a walk-forward. A formal baseline CI guard would have flagged it at unit-test
time.

**Implementation.**
1. Add `research/spikes/crypto_baseline/naive_baseline.py` — implements Naïve (next =
   current), Mean, Drift, and Seasonal baselines from §3 of Puoti et al.
2. Add a `computeNaiveBaseline(steps: BacktestStep[]): MetricBlock` helper to
   `src/tools/finance/backtest/metrics.ts` that computes directionalAccuracy, Brier, and
   meanEdge for a "always predict 50% up" and "always predict last-period direction"
   baseline over the same `steps` array.
3. Add an assertion in `markov-backtest.integration.test.ts`: the full improved arm must
   beat the naïve baseline on dirAcc at h=14d (the clearest horizon advantage we have).
4. Document in `docs/forecast-implementation-review.md` section per shipped feature.

**True-improvement gate.** The baseline itself. Once added, run every existing arm through
it to establish which flags are already beating naïve and which aren't.

**Effort:** Small. Pure evaluation code; no model changes.

---

## 5. Idea #4 — Wire conformalLow/High into forecast outputs

**Source.** Todo `w3-2b`. The conformal PID wrapper (`src/tools/finance/conformal.ts`) was
implemented in W3 Idea 2, and its state is updated during `computeMarkovDistribution`.
However, the resulting `conformalLow` and `conformalHigh` fields are computed but never
surfaced in the agent tool's response payload.

**What's missing.** In `markov-distribution.ts` the conformal PID controller is updated and
its output is available, but the return type (`MarkovDistributionResult`) and the
`markov_distribution` tool response do not include `conformalLow`/`conformalHigh`. The
user and the downstream agent see only the raw Monte-Carlo 5th/95th percentile bounds.

**Implementation.**
1. Add `conformalLow?: number` and `conformalHigh?: number` to `MarkovDistributionResult`.
2. Populate them from the PID controller output in the return block.
3. Surface in the tool response JSON so the agent can quote them alongside the CI.
4. Add a test asserting they are present and narrower than the raw MC bounds after
   a warm-up period.

**Effort:** Small — the computation is already there.

---

## 6. Idea #5 — Horizon-aware GARCH clamping / exponential decay ⭐ R4 regression fix

**Source.** Diagnosed from R4 backtest. GARCH vol layer regresses at h=14d (ΔBrier +0.006).

**Problem.** The GARCH(1,1) forecast `h_t+k` reverts toward unconditional variance at rate
`(α+β)^k`. For BTC with typical `α+β ≈ 0.97`, after k=5 days the forecast is
`≈ 0.97^5 = 0.86 × initial_variance` — still meaningfully elevated. After k=14 days it
decays to `0.97^14 = 0.65 × initial_variance`, which is 35% below the shock level. But the
current code applies the same GARCH scalar at day 1 AND day 14 with only the clamp
`[0.33, 3.0]` providing a ceiling. At h=14d the GARCH scalar for the forecast horizon is
almost irrelevant (the variance is near unconditional), yet it's still being applied as a
non-trivial multiplier.

**Hypothesis.** Add a `horizonDecay` parameter: when `horizon > garchHorizonCap` (default 7),
soft-blend the GARCH scalar toward 1.0 as `blend = max(0, 1 - (d - cap) / (2 * cap))`,
so `effectiveScalar = 1 + blend * (garchScalar - 1)`. Beyond `3 × cap` the scalar is
exactly 1.0 (GARCH is ignored).

**Implementation.**
1. Add `garchHorizonCap?: number` option (default 7) to `computeMarkovDistribution` params.
2. Modify `computeGarchScales` to accept an optional `decayCap` and apply the blend.
3. Re-run R4 backtest with the patch — target: ΔBrier ≤ 0 at all horizons vs no-GARCH arm.

**True-improvement gate.** R4 backtest re-run: GARCH must not regress vs `improvedHA` arm
at any horizon (ΔBrier ≤ 0.001 everywhere) before GARCH is promoted to default-ON.

**Effort:** Small — 15-line change to `garch-scales.ts` + `markov-distribution.ts`.

---

## 7. Idea #6 — Regime-Platt two-pass walk-forward validation

**Source.** R4 Phase A implemented `src/tools/finance/regime-calibrator.ts` and wired it
into `computeMarkovDistribution`. But the R4 backtest omitted it because a single-pass
walk-forward can't fit Platt without lookahead.

**Problem.** We don't know if the regime calibrator actually improves Brier on real BTC data.
The unit tests confirm correctness on synthetic data, but a proper real-data evaluation
requires a two-pass approach:

1. **Pass 1 (calibration):** run improved+HA arm over the first half of the fixture
   (days 0–365). Collect `(pUp, regime, realized)` triples from each `BacktestStep`.
2. **Fit Platt** on those triples using `fitRegimePlatt`.
3. **Pass 2 (test):** run the R4 arm with `regimePlattFits` over the second half
   (days 183–731) — no lookahead.
4. Compare second-half Brier vs identical arm without Platt.

**Implementation.** Extend `btc-multi-horizon-comparison-round4.ts` to add a fifth arm
`improvedR4+Platt` using this two-pass scheme. The fitting code already exists.

**True-improvement gate.** Second-half ΔBrier ≤ −0.005 at h=7d or h=14d vs `improvedR4`
(no Platt). Coverage should stay within ±2pp of target.

**Effort:** Medium — script change only; no new modules.

---

## 8. Idea #7 — Bayesian Model Averaging over forecast heads

**Source.** R4 Idea 7.

**Problem.** Three forecast heads exist — Markov, HMM, jump-diffusion — each with
known strengths by horizon and regime. Currently the HMM is used as an override signal
(weighted by convergence confidence), not a proper head. Manual flag-flipping per horizon
is needed to select the best configuration. We should let the data decide.

**Implementation.**
1. In `walk-forward.ts`, maintain a rolling `logScore[head]` for each head over the last
   `K=30` steps (log-probability of the realized binary outcome under that head's pUp).
2. Compute BMA weights: `w_i = exp(logScore_i) / Σ exp(logScore_j)` (softmax on
   accumulated log-scores).
3. Combine: `pUpBMA = Σ w_i × pUp_i`.
4. Wire as `enableBMA?: boolean` flag through MD + walk-forward.
5. Collect `logScore` per head in `BacktestStep` metadata.

**Why this works.** At h=1d the Markov head has the most data support; at h=14d the
HMM often fires with higher log-score. BMA auto-adjusts without us explicitly knowing
which head is dominant per horizon. The 30-step window makes it reactive to recent
regime changes.

**True-improvement gate.** BMA arm must outperform the best single-head arm at ≥ 4 of 6
horizons on dirAcc or Brier. If it only averages to the mean of the heads, it's not adding
value — the EV is in the adaptive weighting.

**Effort:** Medium (~150 lines). The heads already emit `pUp`; the combiner is new.

---

## 9. Idea #8 — On-chain Polymarket trade direction (critical structural fix)

**Source.** R3 Idea 1 (arXiv:2604.24366 — Dubach 2026). Rated 🔥🔥🔥 but deferred.

**Problem** (unchanged from R3). The public Polymarket WebSocket feed's trade direction
inference is correct only ~59% of the time (barely above 50% chance). The *effective
half-spread changes sign* between the feed and on-chain truth on 67% of markets. Every
flow-derived signal we compute (volume momentum, bid/ask pressure, lambda-style impact)
from the public feed is built on partially reversed direction data.

**Deferral reason.** Requires on-chain ingestion (RPC node or indexer API), which is an
infrastructure dependency not currently in the project. This is the right fix but the
highest engineering lift of any idea in this document.

**When to tackle.** Only after a stable Polymarket indexer API is available or the
replication package from the Dubach paper is confirmed queryable. Until then:
- **Mitigation:** explicitly halve the log-odds weight on any direction-derived signal
  (mark it "low-confidence (public-feed direction)").
- **Guard:** do NOT add new direction-based Polymarket signals until this is fixed.

**Effort:** Medium-High (new infra + signal rewiring).

---

## 10. Idea #9 — LASSO-PCA pooling across model variants

**Source.** R3 Idea 5 (arXiv:2207.04794 — Uniejewski & Maciejowska 2022).

**Distinction from R4 cross-asset Lasso.** R4 Lasso pooled *peer asset returns* to adjust
BTC drift. This idea pools *forecast model outputs* — the Markov `pUp`, the HMM `pUp`,
the jump-diffusion `pJump`, and any conformal-adjusted probabilities — via Lasso weights
learned from past calibration error. The two Lassos are orthogonal.

**Why now.** We now have three functioning forecast heads (Markov, HMM, jump-diffusion)
and can observe their individual Brier scores on backtest folds. The LASSO-PCA approach
from the source paper picked the best-calibrated convex combination automatically on 4/5
electricity markets, beating simple averaging.

**Prerequisite.** Need a rolling calibration store of `(head_pUp[head], realized)` pairs.
The `BacktestStep` structure already stores `probability`, `confidence`, and `realized` —
extend it to store per-head probabilities.

**Effort:** Medium — calibration store extension + new `forecast-pooling.ts` module.

---

## 11. Idea #10 — Intraday Hawkes evaluation (research spike)

**Source.** R4 Idea 5.

**Why first.** ADWIN and Hawkes are byte-identical null on daily BTC. Before we spend
engineering effort improving them, we must confirm the hypothesis holds on intraday data.

**Implementation.** Python research spike in `research/spikes/hawkes_intraday/`:
1. Fetch 90 days of 5-minute BTC bars.
2. Classify 3σ bars as "jump events".
3. Fit Hawkes(`α`, `β`) by MLE per trading day.
4. Report distribution of excitation ratio `α/β`.
5. **Gate:** if median `α/β > 0.2` over the 90-day sample, follow up with a TS port
   seeded from Polymarket event timestamps. Otherwise **drop the Hawkes wiring from
   defaults permanently** and document the decision.

**Effort:** Small (pure Python, ~1 day). High value as a decision gate.

---

## 12. Idea #11 — Longshot premium sanity check for P3 calibration

**Source.** Todo `w3-1b`. Round-3 W3 Idea 1b.

**Problem.** Dubach 2026 documents a systematic longshot spread premium in Polymarket
(thin markets near 0% or 100% have wider spreads, inflating or deflating the implied
probability). Our P3 domain×horizon recalibration (W1 P3) applies an offset without
checking for this artifact — we may be recalibrating correctly in mid-range but introducing
a longshot bias at the extremes.

**Implementation.** In `src/tools/finance/polymarket.ts`, add a post-calibration check:
if `pCalibrated < 0.05 or pCalibrated > 0.95`, apply a mild shrinkage toward the prior
(`0.5 × pCalibrated + 0.5 × pPrior`) to reduce the longshot-premium artifact. Flag this
as `longshotCorrected: boolean` in the anchor metadata.

**True-improvement gate.** Brier score on Polymarket-anchored predictions at the extremes
(`|pRaw − 0.5| > 0.4`) must not worsen vs no shrinkage.

**Effort:** Small (~30 lines).

---

## 13. Idea #12 — mfBm realised-vol forecast (research spike)

**Source.** R3 Idea 6 (arXiv:2504.15985 — Bibinger, Yu & Zhang 2025). Deferred
(high effort, Python spike first).

**Status.** Still deferred. The mfBm estimator beats vector-HAR on out-of-sample
realised-vol when component Hurst exponents differ. For BTC, the Hurst exponent is
time-varying (~0.55 in trending, ~0.45 in mean-reverting regimes). Synergistic with
any regime-aware σ estimate.

**Path to revisit.** Only after:
- the GARCH horizon-aware clamping is validated (Idea #5),
- and intraday data infrastructure is in place (from Idea #10 spike).

The mfBm estimator requires high-frequency data for reliable Hurst estimation; daily
closes alone produce too-noisy estimates.

**Effort:** High (Python spike ~3 days, TS port further 3 days).

---

## 14. Cross-cutting themes

### A. Short horizons are well-served; long horizons remain unsolved

Round-1 gains concentrated at h=14d. R4 gains (if any) at h=1–2d. H=30d is uniformly
negative or neutral. The root cause is information scarcity: 174 obs / arm at h=30d is
statistically insufficient to learn anything beyond the random-walk null. Fixes:

- **Crypto-native peers** (Idea #1) — the only path to adding information at 30d
  without structural model changes.
- **BMA** (Idea #7) — reduces noise by blending heads, may help at 30d via variance reduction.
- **Depth weighting** (Idea #2) — if any deep Polymarket market covers BTC 30d, it anchors
  the distribution better than a shallow one.

### B. GARCH is conditional: fix the horizon-decay before enabling by default

The R4 GARCH clamping caused regression at h=14d. It should not be promoted to default-ON
until Idea #5 (horizon-aware decay) is implemented and validated. Short-circuit in
release notes: "GARCH is opt-in (`enableGarchVol: true`); do not use without setting
`garchHorizonCap: 7`."

### C. Baseline hygiene must come first (Idea #3)

No new flag should be promoted to default-ON without first clearing the naïve-baseline
guard introduced by Idea #3. This is the most important single item in this document
for long-term quality hygiene.

### D. Polymarket direction bias is the silent killer

Until Idea #8 (on-chain direction) is addressed, any Polymarket flow signal is derived
from ~59%-accurate direction inference. We should apply the half-weight mitigation
immediately (one-line change to the log-odds weighting in `polymarket.ts`) even before
the full infrastructure fix lands.

---

## 15. Recommended execution sequence

### Sprint 1 — Quality gates and quick fixes (small effort, high hygiene value)
1. **Idea #3** — naïve baseline guard (small, becomes CI gate for everything else)
2. **Idea #4** — wire `conformalLow/High` into forecast outputs (one PR, existing code)
3. **Idea #5** — horizon-aware GARCH clamping (fix R4 regression, then re-run backtest)
4. **Idea #11** — longshot sanity check for P3 (completes open todo `w3-1b`)
5. **Idea #2** — Polymarket depth weighting (structural, independent of diffusion path)

### Sprint 2 — Signal improvements (medium effort, new information)
6. **Idea #1** — crypto-native Lasso peers (extend fixture, re-run R4 backtest)
7. **Idea #6** — two-pass Platt walk-forward (validates R4 regime calibrator on real data)
8. **Idea #10** — intraday Hawkes spike (kills or resurrects the daily-bar Hawkes path)

### Sprint 3 — Structural model upgrades (higher effort, highest ceiling)
9. **Idea #7** — Bayesian Model Averaging across heads
10. **Idea #9** — LASSO-PCA pooling across model variants

### Deferred
- **Idea #8** (on-chain Polymarket) — unblock when indexer API is available
- **Idea #12** (mfBm) — unblock when high-frequency data is available

---

## 16. Ideas explicitly NOT in scope

| Idea | Reason |
|------|--------|
| Hawkes re-tuning on daily BTC | Confirmed null; dead path |
| ADWIN on BTC return mean | Confirmed null; mean is stationary. Use KSWIN for variance |
| Particle filter HMM (ORCSMC) | Premature; only needed for sub-daily updates |
| WebSocket Polymarket subscriber | Pure infra, no forecast lift |
| SpotV2Net / GNN multivariate vol | Intraday infra dependency; revisit in Wave 5 |
| BTC-denominated prediction markets | Out of forecasting scope |

---

## 17. Open todos carried forward from prior rounds

| Todo ID | Title | Why still open |
|---------|-------|----------------|
| `w3-2b-conformal-wire` | Wire conformalLow/High into outputs | Idea #4 above — small but never landed |
| `w3-1b-longshot-check` | Longshot premium sanity check | Idea #11 above |
| `b1-trust-jsdoc` | Document minVolume24h trust gate | Minor docs; low priority |
| `b6-geopolitics` | Add geopolitics to JUMP_DEFAULTS | Low priority |
| `s53-applied-flag` | Add jumpDiffusionApplied to metadata | Minor metadata completeness |
| `s54-config-forecasting` | Add forecasting block to AppSettings | Config hygiene |
| `b8-review-docs` | Document Merton MC step | Docs only |

---

## 18. Reference inventory (new since Round 3)

Papers referenced but not yet used:

| arXiv | Authors | Key finding |
|-------|---------|-------------|
| 2604.24366 | Dubach 2026 | Polymarket direction inference only 59% accurate; depth-decay near expiry |
| 2307.16895 | Angelopoulos et al. 2023 | Conformal PID — online coverage under distribution shift |
| 2402.05203 | Yang et al. 2024 | Bellman conformal inference — shorter intervals under coverage constraint |
| 2207.04794 | Uniejewski & Maciejowska 2022 | LASSO-PCA forecast pooling beats fixed weights |
| 2504.15985 | Bibinger et al. 2025 | mfBm beats vector-HAR for realised-vol forecasting |
| 2502.09079 | Puoti et al. 2025 | Naïve baselines beat ML/DL on crypto across all horizons |
| 2312.16190 | Cestari et al. 2023 | Hawkes LOB crypto forecasting (intraday, not daily) |
| 2205.06338 | Oxenhorn 2022 | Multivariate Hawkes for stablecoin depeg contagion |

---

*Prepared 2026-04-29. No code changes in this document — pure planning.*
