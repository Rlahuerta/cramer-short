# Forecast Improvement Ideas — Round 4 (post W3R2 wiring)

**Date:** 2026-04-28
**Predecessors:**
- `docs/forecast-improvement-ideas-round3-2026-04-28.md` (Round-3 brainstorm — Ideas 1–8)
- `docs/btc-multi-horizon-backtest-2026-04-28.md` (Round-1: baseline vs improved)
- `docs/btc-multi-horizon-backtest-round2-2026-04-28.md` (Round-2: Hawkes+ADWIN null on BTC daily)

## What we just learned (informs every idea below)

1. **W2/W3 toggles together are worth ~+8pp dirAcc at h=14d** on BTC, and they flip meanEdge positive across all horizons ≤ 14d. Round-1 backtest validates them as defaults-worthy at least for BTC short-mid horizons.
2. **Hawkes + ADWIN are byte-identical no-ops on BTC daily.** Honest null. The wiring is safe but the data doesn't have the patterns these tools detect:
   - ADWIN: BTC return *mean* is roughly stationary; only variance shifts (which ADWIN doesn't track).
   - Hawkes: daily bars smooth out cascading liquidations; α-MLE collapses to 0.
3. **The 30d horizon is still bleeding.** Improved arm posts ΔdirAcc=−0.005 at 30d and a *negative* meanEdge of −0.004. Whatever we do next, **the long horizon is where we have the least edge and the most room to win**.
4. **Coverage is over-conservative.** Improved arm CIs cover 0.94–0.99 vs the honest 0.90 target — the model is still hedging too hard. Tighter calibration is leaving bps on the table.

## Round-3 idea status (carry-forward)

| #   | Title                                  | Round-3 status | Round-4 verdict |
|----:|----------------------------------------|----------------|-----------------|
| 1   | Conformal-PID interval calibration     | done           | — keep, monitor |
| 2   | Polymarket-informed jump-diffusion     | done           | — keep, monitor |
| 3   | Hawkes self-exciting jump intensity    | wired (W3R2)   | **null on BTC daily**; revisit on intraday data |
| 4   | ADWIN drift-detector trim              | wired (W3R2)   | **null on BTC mean-stationary returns**; pivot to *variance-aware* drift detector (KSWIN, Page-Hinkley on |r|) |
| 5   | LASSO-PCA cross-asset signal pooling   | deferred       | **strong revisit** — could finally reduce 30d noise |
| 6   | mfBm long-memory diffusion             | deferred       | weak revisit — high implementation cost |
| 7   | Particle filter for HMM regime tracking| deferred       | revisit if we add hidden volatility states |
| 8   | Crypto unpredictability baseline       | deferred       | **important sanity check** — implement before promoting any new flag |

## Round-4 ideas (ranked by expected EV / engineering cost)

### Idea 1 — Variance-aware drift detector (KSWIN on |r|) ⭐ top pick

**Problem.** ADWIN tracks the mean of the input stream. BTC return *means* are stationary, but the variance regime shifts dramatically (calm 1% days → 6% days during liquidation cascades). Trimming on the wrong statistic explains why Round-2 was a no-op.

**Hypothesis.** KSWIN (Kolmogorov-Smirnov windowing) compares the empirical distribution of recent vs older samples — it catches variance shifts ADWIN misses. Apply it to `|log_return|` on a rolling window; trim history at the most recent variance regime boundary.

**Implementation.** ~80 lines. Add `src/utils/kswin.ts` (pure module), extend `applyAdwinTrim` → `applyDriftTrim` with `mode: 'mean' | 'variance' | 'either'`. TDD: synthetic constant-mean / variance-step series.

**Expected EV.** Should reduce 30d Brier by 1–2pp by stripping pre-cascade calm history that biases the diffusion variance estimate downward. Safe to ship behind feature flag.

### Idea 2 — Cross-asset LASSO pooling for long-horizon edge ⭐ top pick

**Problem.** At h=30d, BTC alone provides too little signal (174 obs / arm) to beat the random-walk null. We have correlated assets in the fixture (MSTR, SPY, QQQ, GLD, DXY, ETH, etc.) that *do* contain leading information at multi-week horizons.

**Hypothesis.** Fit a Lasso(Cross-Asset Returns → BTC h-day return) using rolling windows; use the predicted return as a *bias term* added to the Markov drift. Lasso's L1 sparsity prevents the small-sample overfitting that killed prior cross-asset attempts.

**Implementation.** ~200 lines. Add `src/tools/finance/cross-asset-lasso.ts`. Use a simple coordinate-descent Lasso (no external dep). Hyperparam λ tuned by cross-validation on training folds. Wire as `enableCrossAssetBias` flag, default OFF. Pull the asset universe from the existing fixture so we don't add a data dependency.

**Expected EV.** This is the most plausible path to a meaningful 30d gain because it fundamentally changes the information set. Even a 5% pooling-derived bias correction at 30d could move dirAcc from 0.443 → 0.480.

### Idea 3 — Regime-conditional Brier-optimal recalibrator (Platt / isotonic per regime)

**Problem.** The current calibration applies a single offset across all regimes. Round-1 showed the improved arm Brier *worsens* slightly at h={3, 7} despite gaining dirAcc — the calibration is mis-tuned for sideways-into-trend transitions.

**Hypothesis.** Fit one Platt-style 2-param logistic per regime label (Bull/Bear/Sideways) on a held-out backtest fold. Apply the regime-dominant calibrator at inference. Total params ~6, well below over-fit risk.

**Implementation.** ~120 lines. Reuse the existing `BacktestStep` history. Add `src/tools/finance/regime-calibrator.ts`. Persist the fitted params to `data/calibration/regime-platt-{ticker}.json`. TDD: synthetic 2-regime mixture with deliberate miscalibration.

**Expected EV.** Brier improvement of 0.01–0.02 across all horizons; minor dirAcc movement. **High signal-to-cost ratio** — pure post-processing, no impact on the diffusion path.

### Idea 4 — GARCH(1,1) variance forecast layered on the Markov diffusion

**Problem.** Round-2 confirmed the model uses a *static* per-regime σ that doesn't react to recent volatility clustering. When BTC enters a cascade week, the model still uses last-month's σ for the trajectory MC, which is why CI coverage is 0.95 instead of 0.90 — we're projecting last-week's calm into next-month's path.

**Hypothesis.** Fit GARCH(1,1) on the trimmed return series; use σ_{t+1|t} as a multiplicative scaler on the regime-conditional σ inside `computeTrajectory`. This is the variance analog of the regime-specific σ flag (Phase 7) but uses the *forward-looking* GARCH forecast instead of a regime indicator.

**Implementation.** ~150 lines. Add `src/utils/garch.ts` (already in the P3a placeholder slot per `plan.md`). MLE via gradient descent, ~50 iterations max. TDD: known-σ Monte Carlo data with σ-bursts.

**Expected EV.** Tightens coverage toward 0.90 (currently 0.95) which directly improves Brier and meanEdge at all horizons. Also makes 1d/2d horizons more responsive to recent vol.

### Idea 5 — Intraday Hawkes evaluation (separate research spike, not a TS flag yet)

**Problem.** Round-2 showed Hawkes is a no-op on daily bars but the underlying mathematics may still be valuable for *intraday* event arrival.

**Hypothesis.** Re-fit Hawkes on 5-minute or 1-hour BTC bars where cascading liquidations *are* visible. If α/β > 0.3 reliably appears, the daily-bar wiring may still be useful when seeded with a Polymarket event whose intensity Hawkes can amplify (rather than detect from scratch).

**Implementation.** Pure research notebook in `research/`, no TS changes. Fetch 90 days of intraday BTC from any free source. Fit Hawkes per-day; report distribution of α/β. **Acceptance:** if median α/β > 0.2 over 90 days, write a follow-up TS PR; otherwise drop the daily-bar Hawkes wiring entirely.

**Expected EV.** Either resurrects Idea 3 with a proper data foundation, or kills it cleanly so we stop tuning a dead path.

### Idea 6 — Polymarket order-book *depth* as a confidence weight

**Problem.** Current Polymarket integration uses only price, not liquidity. A market with $100 of liquidity gives the same anchor weight as one with $50K. This is structurally wrong — thin markets are noise.

**Hypothesis.** Pull `bid_size + ask_size` from the CLOB `/orderbook` endpoint; weight each anchor's contribution to the regime mixture by `min(depth_usd / 1000, 1.0)`. Markets with <$100 depth become near-zero weight; deep markets dominate. This is structurally the right way to combine venues.

**Implementation.** ~60 lines in `src/tools/finance/polymarket.ts` + 1 new field in the anchor type. TDD: synthetic anchors with varying depth, assert weight scaling.

**Expected EV.** Removes thin-market noise that currently mis-pulls the mixture toward illiquid extremes. Should improve calibration at all horizons; biggest impact on h=30d where the fewest deep markets exist (so depth-weighting becomes a sparsity-aware aggregator).

### Idea 7 — Bayesian model averaging over MC + jump-diffusion + HMM heads

**Problem.** We currently use a single Markov head with optional jump-diffusion overlay. The HMM tracker is computed but only used as an ensemble *override*, not weighted.

**Hypothesis.** Compute log-score (log-likelihood of realized outcome) for each head over a rolling window; combine the heads' P(up) by exp-log-score weights (proper BMA). Self-tunes which head dominates per regime/horizon without manual flag-flipping.

**Implementation.** ~150 lines. Compute log-score per step in `walk-forward.ts`; persist rolling weights. Wire the BMA combiner before the calibration step.

**Expected EV.** Adaptive — the model stops needing manual horizon-specific flag tuning. Should narrow the gap between best and worst horizon.

## Recommended next sequence (TDD, separate commits)

1. **Idea 3 (regime Platt)** — lowest risk, pure post-processing, can ship in a day. Validates whether per-regime calibration moves the needle before any structural changes.
2. **Idea 4 (GARCH variance)** — tightens coverage, directly addresses the over-conservative CI problem visible in Round-1.
3. **Idea 1 (KSWIN variance drift)** — natural follow-on after GARCH establishes a variance-aware baseline.
4. **Idea 6 (Polymarket depth weighting)** — independent of the diffusion path; can run in parallel with the above.
5. **Idea 2 (cross-asset Lasso)** — biggest expected EV at 30d but largest engineering cost; do it last when the cheaper fixes have settled.
6. **Idea 5 (intraday Hawkes spike)** — research-only; defer until we have a confirmed need.
7. **Idea 7 (BMA)** — defers cleanly; useful once we have ≥3 well-behaved heads to combine.

## What we're explicitly *not* pursuing this round

- **mfBm long-memory diffusion** — high implementation cost, weak prior evidence on crypto.
- **Particle filter HMM** — premature until we have hidden states worth tracking.
- **WebSocket Polymarket subscriber** — pure infra, no forecast lift.
- **Re-tuning Hawkes σ on BTC daily** — confirmed dead path (Round-2).
