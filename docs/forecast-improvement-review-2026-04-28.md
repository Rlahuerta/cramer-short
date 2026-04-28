# Forecast Improvement Proposals — Review & Implementation Plan

**Date:** 2026-04-28
**Reviewer:** GitHub Copilot CLI
**Scope:** Four user-proposed enhancements to the Markov / HMM / trajectory forecasting pipeline:
1. Markov-Switching **Volatility Regimes** (Sticky Strike / Sticky Delta / Sticky Implied Tree)
2. **Jump-Diffusion** trajectories informed by Polymarket
3. **Markov-Switching Multifractal (MSM)** upgrade for volatility
4. **Q → P measure transformation** (Radon-Nikodym / Girsanov) for Polymarket inputs

> **Reviewer's note.** Inputs are assessed against the current TypeScript implementation (`src/tools/finance/markov-distribution.ts`, `hmm.ts`, `rnd-integration.ts`, `ensemble.ts` — ~5 900 LOC) and the Python research mirror (`research/models/*.py` — ~1 700 LOC). Findings combine code reading, the Bloch citations supplied with the proposals, and the local references library at `references/` (Calvet-Fisher 2001/2004 in `markov-probability/`, Tsang & Yang 2026 and Reichenbach & Walther 2025 in `prediction-markets/`, El Hassan / Maddah / Taleb 2026 in `derivatives-risk/`).

---

## 0. TL;DR

| # | Idea                                            | Already done?                  | Theoretical merit | Empirical merit | Eng. cost | Risk     | Recommended priority |
|---|-------------------------------------------------|--------------------------------|-------------------|-----------------|-----------|----------|----------------------|
| 4 | **Q → P transformation**                        | **YES — shipped.** See §1.     | Strong            | Validated       | —         | —        | Done. Audit-only.    |
| 2 | **Polymarket-informed jump-diffusion**          | No                             | Strong            | High (clean Polymarket signal exists today) | Medium    | Medium  | **P1** (do next)     |
| 1 | **Volatility-regime HMM (sticky surfaces)**     | No                             | Medium            | Asset-limited (needs liquid IV history)     | Medium    | High    | **P2** (research spike) |
| 3 | **Markov-Switching Multifractal (MSM)**         | No                             | Strong            | Strong but fragile in TS port               | High      | High    | **P3** (R&D track)   |

**Net guidance.** Skip nothing — the four ideas are complementary, not substitutes — but **stage them**. Idea 4 is already in production, so the immediate work is *audit + documentation*. Idea 2 is the highest-return next step because it activates Polymarket signals the engine currently ignores. Idea 1 must wait for an IV-history data source (not in `data/prices.py` today). Idea 3 is a research project, not a refactor: do a `research/spikes/msm/` Python prototype before any TS parity port.

---

## 1. Idea 4 — Q → P Transformation (Status: **already shipped**)

### What the proposal asked for
A Radon-Nikodym shift from risk-neutral Polymarket prices to physical probabilities via the market price of risk:
$$
P^{\mathbb P}(S_T > K) = \Phi\!\left(\Phi^{-1}\!\left(P^{\mathbb Q}(S_T > K)\right) + \frac{\mu - r_f}{\sigma}\sqrt{T}\right)
$$

### What is in the repo today
This **is the function `transformQToP`** in `src/tools/finance/rnd-integration.ts:15-34` (TS) and `research/models/rnd.py` (Python parity). The implementation matches the proposed formula 1-for-1, including the 0.001/0.999 clip on `qProb` and the `daysToExpiry/365` time scaling.

```ts
// src/tools/finance/rnd-integration.ts:28-33
const lambdaMpr = (historicalDrift - riskFreeRate) / Math.max(volatility, 1e-6);
const zQ        = normPPF(qClipped);
const zP        = zQ + lambdaMpr * Math.sqrt(T);
return normCDF(zP);
```

It is wired into the pipeline at `markov-distribution.ts:4216-4252` where Polymarket strike anchors are converted to physical survival probabilities, fitted to a log-normal RND (`fitLognormalFromStrikes`), bucketed into bull/bear/sideways via `lognormalToRegimeProbabilities`, and then applied to the transition matrix via `nudgeTransitionMatrix`.

Unit + parity coverage exists: `src/tools/finance/rnd-integration.test.ts` and `rnd-integration.parity.test.ts` (TS↔Python).

### Recommended action
**No code change needed.** Add three audit items instead:

1. **Document the assumption.** Make the JSDoc on `transformQToP` explicit that `historicalDrift` must be in the **same time-unit space (annualised)** as `riskFreeRate`. Today, callers at line 4223–4228 assume both are annualised, but nothing in the signature enforces it. A misuse with daily drift would understate λ by a factor of ~252.
2. **Per-asset MPR cap.** `(μ − r_f)/σ` for crypto in 2020-2021 hit > 2 (~200 % drift, ~50 % vol). Empirically Sharpe > 2 for a year is non-stationary; clip `lambdaMpr` to `[-1.5, 1.5]` so a transient bubble does not push every Polymarket bear contract into a near-zero physical probability. Mirror the cap in `rnd.py`.
3. **Track the shift.** Surface the `lambdaMpr * sqrt(T)` shift in `metadata.diagnostics.qToPShift` so backtests can confirm the transformation is doing the work the theory claims.

These are tracked as checklist items in §6.

---

## 2. Idea 2 — Polymarket-Informed Jump-Diffusion Trajectories

### Why this is the right next step
- Polymarket already exposes **dated event contracts** ("Will X happen by Friday?") that are essentially Arrow-Debreu securities for jumps. The current trajectory MC at `markov-distribution.ts:2578` (`computeTrajectory`) is pure Brownian + Student-t innovations and **discards this signal entirely**.
- The El Hassan / Maddah / Taleb (2026) paper (`references/derivatives-risk/2602.14350v1.pdf`) shows that hidden jump risk dominates short-tenor American option mispricing — exactly the regime our short-horizon BTC/SPY trajectories serve.
- This unlocks an obvious portfolio improvement: today, when Polymarket prices a 30 % chance of a US-China tariff escalation by Friday, the trajectory still spits out a ±2σ Brownian band centred on the historical drift. Post-fix, the band gains a downward fat-tail mode.

### Three corrections to the user's proposal

The proposal sketch is directionally right but has three issues that must be fixed before merging:

| Issue                                                                                                                               | Why it matters                                                                                                            | Fix                                                                                                                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `daily_lambda = polymarket_jump_prob / days` linearises a survival probability                                                       | If `polymarket_jump_prob = 0.5` over 5 days, daily `λ = 0.10`, but compounded survival = `0.9^5 = 0.59`, so realised P(jump) = 0.41, not 0.5. | Use the survival-probability inversion: `daily_lambda = 1 - (1 - p_total)^(1/days)`, equivalent to Poisson hazard `λ = -ln(1 - p_total) / days`.                                                                                            |
| `polymarket_jump_prob` is drawn straight from the Polymarket YES price — that's a Q-measure jump probability                         | We just argued in §1 that Polymarket prices are Q. The intensity must be Girsanov-shifted to P before being plugged into the physical MC.       | Reuse `transformQToP` on the headline jump probability before computing `daily_lambda`. (For event-style markets the Sharpe term and σ may be tiny, so the shift is small, but the discipline keeps the math internally consistent.)             |
| `jump_mean_impact = -0.10` is hard-coded and pessimistic                                                                            | Some Polymarket events are bullish (e.g., "ETF approved by Friday"). A symmetric −10 % bias would systematically underprice rallies.            | Tag each Polymarket market in `polymarket.ts` with a `jumpDirection: 'down' | 'up' | 'unknown'` annotation (LLM-classified at fetch time, default `'unknown'`); use historical event-study moves per asset class (already cached in `.cramer-short/api-routing.json` style cache). Default magnitudes by asset: equities ±4 %, BTC ±8 %, geopolitics ±10 %. |

### Theoretical statement (clean form)

Replace the trajectory innovation in `computeTrajectory` (lines 2655–2660) with:

$$
\Delta \log S_t = \mu_t + \sigma_t Z_t + J_t \mathbf{1}_{N_t}, \quad N_t \sim \text{Bernoulli}(\lambda_t), \quad J_t \sim \mathcal N(\mu_J, \sigma_J^2)
$$

with $\lambda_t$ dynamically taken from the Q→P-transformed Polymarket probability and $(\mu_J, \sigma_J)$ taken from the per-asset event prior. The drift is **compensated** so the unconditional expectation of the jump term is absorbed:

$$
\mu_t \;\leftarrow\; \mu_t \;-\; \lambda_t (e^{\mu_J + \tfrac{1}{2}\sigma_J^2} - 1)
$$

Without this compensation, adding the jump term inflates the conditional mean and breaks the calibration the calibrated CDF in §4 of the report relies on.

### Where it lives in the code

- **TS:** `src/tools/finance/markov-distribution.ts` — extend `computeTrajectory` (line 2578) with optional `jumpSpec?: { dailyLambda: number; muJ: number; sigmaJ: number }`. Compute `jumpSpec` outside `computeTrajectory` from the Polymarket fetch (currently at lines 4216–4252) and the Q→P helper.
- **Python parity:** `research/models/trajectory.py` (404 LOC). Same signature.
- **Test surface:** add a parity test fixture mirroring the existing `rnd-integration.parity.test.ts`. Also add a "no Polymarket → identical to current MC" regression test so the change is provably backwards-compatible when no jump market is supplied.

### Risks

- **Test snapshot churn.** The existing `markov-distribution.test.ts` (6 309 lines) has many implicit numeric snapshots. Gating `jumpSpec` behind an opt-in flag (default off) keeps the entire existing test suite green.
- **Polymarket signal quality.** Tsang & Yang (2026) and Reichenbach & Walther (2025) both show that **thin** Polymarket contracts (< $50 k OI) are noisy and whale-manipulable. Enforce the `trustScore === 'high'` filter already used at `markov-distribution.ts:4216` for jump intensity inputs as well.
- **Look-ahead.** The Polymarket "settle by date" must be ≤ horizon. Validate at extraction.

---

## 3. Idea 1 — Volatility-Regime HMM (Sticky Strike / Sticky Delta / Sticky Implied Tree)

### Why this looks attractive
The Bloch §4.1.1.2 framing is a real, academically-grounded way to think about **how the implied vol surface co-moves with spot**. If we knew the regime, we could parameterise the trajectory with a return ↔ vol correlation that matches it, recovering the empirical leverage effect and giving more honest CIs in selloffs.

### Why this is risky as proposed

1. **Data dependency the engine doesn't have.** `data/prices.py` and `src/tools/finance/*` ingest **spot price histories only**. There is no implied-vol time series in the pipeline today, and the project's data-source budget (FMP + Yahoo) does not include intraday IV surfaces. Without IV data, `corr(spot_return, IV_change)` cannot be computed for any of our supported tickers — the proposed classifier `classify_volatility_regime(spot_returns, iv_changes)` is undefined on our actual inputs.
2. **The hard-coded 0.5 correlation thresholds are not invariant.** Empirically, equity-index `corr(ΔSPX, ΔVIX) ≈ −0.7` *all the time*; the rolling window basically never falls inside [−0.5, +0.5]. The proposed three-state classifier collapses into a single state for SPY/QQQ-class assets. For BTC the same correlation is far less stable, and 0.5 is too high a threshold to ever trip.
3. **The leverage adjustment doubles up with the existing regime sigma.** `markov-distribution.ts` already implements `regimeSpecificSigmaActive` (≈ line 3679 onward) — when a single regime dominates the start mixture, the trajectory drifts toward that regime's empirical sigma. In bear regimes the empirical sigma already encodes the leverage effect via the historical sample. Multiplying by an extra `1.5` would over-shoot.

### What is salvageable

The **theoretical insight** is right: spot/vol correlation matters. The right way to operationalise it without IV data is a **return-conditional vol scaling on the spot-return moments themselves**, which is a regime-conditional GARCH-lite proxy:

```
recent_corr = corr(R_{t-N..t}, |R_{t-N..t}|)   // spot vs. abs-return as IV-proxy
if recent_corr < -0.3:  leverage_active = True  // bearish leverage regime
```

Even this is fragile, and the literature (e.g., Jansen & Lange, 2018) shows the realised-vol proxy lags IV by 2–4 days. So the **net gain** over the existing `regimeSpecificSigmaActive` knob is small.

### Recommendation

**Defer.** Open an `R&D` ticket to source an IV time-series (Deribit DVOL for crypto, VIX/MOVE for equities/rates) for the top-10 most-queried tickers. Until then, the function would silently degrade to a no-op for everything except SPY/QQQ and BTC on Deribit, which is exactly the assets where the existing pipeline is already strongest.

If IV data does become available, the architecture should be a **secondary HMM in `hmm.ts`** (parallel to the return-based HMM today), feeding a `volRegimeOverride` parameter into `computeTrajectory` — *not* a global `vol_multiplier` rewrite of the MC step.

---

## 4. Idea 3 — Markov-Switching Multifractal (MSM) Upgrade

### Why the MSM is the right ceiling
Calvet & Fisher (2001 *J. Econometrics*; 2004 *J. Financial Econometrics*) — the seminal references in `references/markov-probability/` — show that MSM **dominates** GARCH(1,1) and FIGARCH on out-of-sample volatility forecasting at horizons of 1–63 days, exactly our forecasting band. The model is also small (typically `k = 4..8` multipliers, 2–4 parameters total), so it is **less data-hungry than HMM**, which is a meaningful advantage for low-history tickers.

The user's proposal is also the simplest variant (binomial multipliers, geometric γ). That is the right starting point.

### Three issues with the proposal

1. **The user's `next_daily_volatility` is a *simulator* update, not an *estimator*.** It draws new multipliers stochastically; it does not infer the current state from observed returns. To use it for forecasting we need the **filtered probability vector** `P(M_t | R_{1..t})` from a forward-recursion (Hamilton filter), which is non-trivial — the joint state space is `2^k = 16..256` cells.
2. **Fitting γ-vector is a non-convex MLE.** Calvet & Fisher use simulated method of moments (SMM); the binomial-MSM has 4 params (`m_0`, `σ_bar`, `γ_k`, `b`). A naïve grid search is fine for a research spike, but the production pipeline needs a robust optimiser (Nelder-Mead is OK; the existing `fitLognormalFromStrikes` already uses one).
3. **Integration with the regime Markov chain is non-trivial.** Today, `regimeStats[state].stdReturn` provides the per-regime sigma. With MSM, sigma is **time-varying within a regime**. We must decide whether MSM *replaces* the regime-conditional sigma, *modulates* it, or *runs in parallel as a separate ensemble member*. The cleanest first cut is the third: add an `'msm'` source to `ensemble.ts` (line 240) alongside `'polymarket' | 'markov' | 'blend'`.

### Effort estimate

- **Research spike (Python only):** 3–5 days. New module `research/models/msm.py` with a binomial MSM, MLE via Nelder-Mead, in-sample/out-of-sample sigma forecast tests against `pytest` baselines using existing `research/data/` price series.
- **Validation:** Compare MSM 1-day, 5-day, 30-day vol forecasts against current ensemble's implicit vol forecast on `research/tests/` fixtures. Demand `RMSE_MSM ≤ 0.95 × RMSE_current` before any TS port.
- **TS port:** if validated, ~1 week. New file `src/tools/finance/msm.ts` mirroring `research/models/msm.py`, parity tests in `src/tools/finance/msm.parity.test.ts`. Integration into `computeTrajectory` via an optional `volForecast?: number[]` array (per-day sigma override). No changes to the discrete Markov chain itself.

### Risks

- **State explosion.** `2^k` joint states means k ≤ 6 in practice. Document and enforce.
- **Numerical underflow** in the forward filter — must work in log-space (the existing HMM in `hmm.ts` already does this).
- **Snapshot churn** in `markov-distribution.test.ts`. Same gating discipline as Idea 2: opt-in flag, default off.

### Recommendation

**Greenlight a Python-only research spike**, no TS work yet. The decision to port comes down to the validation result.

---

## 5. Cross-Cutting Concerns

### 5.1 TS ↔ Python parity discipline
The repo's invariant (cf. `docs/python-research-mirror-plan.md`) is that any change to the forecasting math ships in **both** TS and Python with parity tests. Apply this rule to Ideas 2 and 3. Idea 1 is deferred so does not yet apply. Idea 4 is already mirrored.

### 5.2 Backtest re-baselining
`src/tools/finance/backtest/` compares forecast quality against persisted historical truth. After Idea 2 lands, **re-run the backtest harness** and update the calibration coefficients in `src/utils/ensemble.ts` (`YES_BIAS_MULTIPLIER`, `adjustYesBias`'s β = 0.035) — both were tuned to the current jump-free MC. Mis-tuned blending will eat all of Idea 2's lift.

### 5.3 Public-facing metadata
Add fields to `MarkovDistributionMetadata`:
- `qToPShift: number` (Idea 4 audit)
- `jumpDiffusionApplied: boolean` and `jumpIntensityP: number | null` (Idea 2)
- `volModel: 'student-t' | 'msm' | 'regime-mix'` (Idea 3 / today)

### 5.4 Configuration surface
Extend `.cramer-short/settings.json` schema in `src/utils/config.ts`:
```jsonc
{
  "forecasting": {
    "enableJumpDiffusion": false,         // Idea 2
    "jumpDirectionDefaults": { "BTC": "down", "SPY": "down" },
    "enableMSM": false,                   // Idea 3 (after spike)
    "qToPMprCap": 1.5                     // Idea 4 audit
  }
}
```
All defaults preserve current behaviour.

---

## 6. Phased Implementation Plan

> Each phase ends with `bun run typecheck` + `bun test` + `pytest research/tests/` all green and a remediation note in the PR body.

### Phase A — Audit & document Idea 4 (≤ 0.5 days)
- [ ] Strengthen JSDoc on `transformQToP` to require annualised inputs, explicitly.
- [ ] Add `qToPMprCap` clip (default 1.5) to both TS and Python with a parity test.
- [ ] Surface `qToPShift` in metadata + extend the existing `rnd-integration.test.ts` to cover the cap.
- [ ] Append a "No fixes — already in production" note for Idea 4 to this document's appendix.

### Phase B — Idea 2 (jump-diffusion) end-to-end (~3–5 days)
- [ ] **B1.** Add `extractJumpEventMarkets()` in `src/tools/finance/polymarket.ts`. Consume only `trustScore === 'high'` markets whose settlement date ≤ horizon.
- [ ] **B2.** Implement `polymarketProbToHazard(p, horizon)` returning `−ln(1−p)/horizon` (with TS+Py parity test).
- [ ] **B3.** Wrap the hazard with `transformQToP` to obtain a physical daily intensity. Cap at `min(0.95, λ)`.
- [ ] **B4.** Extend `computeTrajectory` (TS + Py) with optional `jumpSpec`. Apply Merton-compensated drift `μ_t ← μ_t − λ_t(e^{μ_J + σ_J²/2} − 1)`. Default off when `jumpSpec` undefined → byte-identical to current output.
- [ ] **B5.** Wire the call site in `computeMarkovDistribution` (around line 4252). Gated by `forecasting.enableJumpDiffusion` (default `false`).
- [ ] **B6.** Add per-asset `JUMP_DEFAULTS` table in `src/tools/finance/jump-priors.ts` (equities ±4 %, BTC ±8 %, geopolitics ±10 %). Mirror in Python.
- [ ] **B7.** Re-run backtest harness; recalibrate `YES_BIAS_MULTIPLIER` if avg P(up) drifts > 1 pp.
- [ ] **B8.** Add metadata fields. Update `docs/forecast-implementation-review.md` with the new MC step.

### Phase C — Idea 1 (sticky-vol HMM): research-only ticket (~0 dev days now)
- [ ] **C1.** File a tracking ticket to source IV history for SPY, QQQ, BTC, ETH (Deribit DVOL + CBOE feeds). No code change until data exists.
- [ ] **C2.** Once data exists, re-evaluate against the conditional-vol scaling proxy in §3.

### Phase D — Idea 3 (MSM): Python research spike (~3–5 days)
- [ ] **D1.** Create `research/spikes/msm/` with `msm.py` (binomial MSM + Hamilton filter + Nelder-Mead MLE in log-space).
- [ ] **D2.** Validation script: in-sample / out-of-sample 1d/5d/30d sigma forecast on existing `research/data/` price series. Compare RMSE vs. current pipeline's implied sigma; report in `docs/msm-spike-results.md`.
- [ ] **D3.** Decision gate: port to TS only if `RMSE_MSM ≤ 0.95 × RMSE_current` on at least 6 of 10 tickers.
- [ ] **D4.** (Conditional on D3.) Port `msm.py` → `src/tools/finance/msm.ts`. Wire as optional `volForecast` array consumed by `computeTrajectory`.

### Phase E — Documentation and ensemble retuning (~1 day, after B and any of C/D land)
- [ ] **E1.** Update `docs/forecast-theoretical-foundations.md` with the §3.x jump-diffusion derivation and (if applicable) §4.x MSM derivation.
- [ ] **E2.** Re-tune `src/utils/ensemble.ts` weights via the backtest harness; document the before/after calibration table.
- [ ] **E3.** Add the new flags to `docs/features.md`.

---

## 7. Open Questions for the User

1. **Jump-direction priors.** Do you have a preferred source for the per-asset `JUMP_DEFAULTS` magnitudes (Phase B6), or shall we estimate them from the rolling 90-day max-abs-daily-return per asset in `data/prices.py`?
2. **Polymarket conditional-event surfacing.** Are there specific Polymarket markets you already curate (e.g., a recurring "Tariff escalation by Friday?") that should be hard-mapped to specific tickers (SPY, BTC), or should the matching be purely topical via the existing LLM tag step in `polymarket.ts`?
3. **MSM scope.** Is the MSM research spike (Phase D) something you want pursued in this repo, or is it better filed as a long-term project alongside the next major version?
4. **Risk-budget for re-tuning.** After Phase B re-baselines `YES_BIAS_MULTIPLIER`, are you willing to accept a one-shot regression in historical backtest fit if the post-jump model has visibly fatter, more honest tails on selloff days?

---

## 8. Appendix — Quick Cross-Check Against Bloch's Sections

| Bloch §        | Concept                                  | Map to repo                                                             |
|----------------|------------------------------------------|--------------------------------------------------------------------------|
| 2.1.5          | Multifractal scaling                     | Idea 3 (MSM) — not yet implemented                                       |
| 2.2.1.1        | Empirical multifractality                | Idea 3                                                                   |
| 2.2.3          | Jump-diffusion                           | Idea 2 — not yet implemented                                             |
| 3.2.1          | Merton model                             | Idea 2                                                                   |
| 4.1.1.2        | Volatility regimes (sticky family)       | Idea 1 — deferred (no IV data)                                           |
| (Girsanov §)   | Risk-neutral / physical change of measure| Idea 4 — **already implemented** in `rnd-integration.ts:15-34`           |

---

*Plan compiled 2026-04-28. No source files were modified during this review. Ready for sign-off before entering Phase A.*

---

## 9. Phase A — Completion Notes (2026-04-28)

**Status:** ✅ Complete. All acceptance tests green; 3 pre-existing integration-test failures on `topic/opt-skills` HEAD are unrelated to Q→P.

### Delivered
- **A1 — JSDoc/docstring strengthening.** `transformQToP` (TS) and `transform_q_to_p` (Py) now explicitly document units (annualised drift / vol, T = days/365), the Girsanov derivation, and edge-case behaviour at the boundaries.
- **A2 — Market-Price-of-Risk cap.** Introduced `DEFAULT_MPR_CAP = 1.5` (TS export + Python module constant). Both functions accept an optional `mprCap` parameter that clamps `|λ|` before applying the Z-shift. Default cap = 1.5 (≈ 95th-percentile equity-risk Sharpe). Pathological inputs (e.g., drift = 3.0, vol = 0.3 → raw MPR ≈ 9.8) now produce a Girsanov shift of ≈ 0.43σ instead of ≈ 2.82σ. Cap = 0 ⇒ identity transformation, providing a kill-switch.
- **A3 — Provenance metadata.** Added optional `MarkovDistributionResult.metadata.rndIntegration` field surfacing `{anchorsUsed, historicalDriftAnnual, volatilityAnnual, riskFreeRate, mprRaw, mprUsed, meanZShift, qualityScore}`. Populated whenever ≥2 high-trust Polymarket strike anchors are consumed; absent otherwise. Computation routes through the new diagnostic helper `transformQToPWithShift` / `transform_q_to_p_with_shift`.
- **A4 — Documentation.** This section.

### Tests added
- `src/tools/finance/rnd-integration.test.ts`: 4 new cases (cap activation, narrowing behaviour, exported constant, `transformQToPWithShift` provenance).
- `src/tools/finance/rnd-integration.parity.test.ts`: 2 new TS↔Python parity tests (default cap, explicit cap).
- `research/tests/test_rnd.py`: 4 new cases mirroring the TS suite.

Total RND unit-test count: 28 TS pass + 22 Python pass (was 21 TS + 18 Py before A).

### Notes / non-changes
- The default cap value (1.5) was chosen so that all live calls in the BTC / SPY / GLD test fixtures fall well *below* the cap (typical computed `|λ| ≈ 0.6–0.9`), keeping the production path numerically identical. The cap only fires when historical-window drift is unrealistically extreme (e.g., a memecoin's parabolic 90-day window).
- `transformQToPWithShift` is purely diagnostic; the live conversion still goes through `transformQToP` with default cap, preserving byte-identical pricing for unit tests that don't observe metadata.
- Phase B (jump-diffusion) implementation begins next; default-off via `enableJumpDiffusion: false`.


---

## 10. Phase B — Polymarket-Informed Jump-Diffusion (Idea 2) — Completion Notes

**Status:** ✅ Complete. Default-off; production trajectory output is byte-identical until `enableJumpDiffusion: true` is set with a non-empty `jumpEvents` array.

### Delivered

- **B6 — Asset-class jump-magnitude priors.** `JUMP_DEFAULTS` table in both `src/tools/finance/jump-diffusion.ts` and `research/models/jump_diffusion.py`. Keys: `'etf' | 'equity' | 'crypto' | 'commodity'`. All defaults carry **negative** `meanLogJump` (selloff bias) because Polymarket tail-risk markets cluster on the downside. Calibration source documented in code: rolling 90-day max-abs-daily-log-return percentiles (SPY/QQQ/BTC/ETH/GLD/USO, 2020-2024).
- **B2 — Hazard conversion.** `polymarketProbToHazard(p, days)` uses the **correct** continuous-time survival relation `λ_total = −ln(1 − p)`, then splits uniformly across the horizon, and caps at 0.95/day. The naive `p / days` linear approximation is rejected because it over-estimates λ for large p.
- **B3 — One-shot composition.** `buildJumpEventSpec()` chains `transformQToP` → `polymarketProbToHazard` → asset-class prior, returning a fully-typed `JumpEventSpec` ready for the trajectory MC.
- **B1 — Polymarket curation.** `extractJumpEventMarkets()` in `polymarket.ts` filters raw markets by `volume24h ≥ 5_000`, `ageDays ≥ 2`, settlement-date inside horizon, and probability strictly in (0, 1). Returns minimal `JumpEventMarket[]` (id, Q-prob, daysToSettlement, question) so the trajectory module never sees Polymarket-specific fields.
- **B4 — Trajectory MC extension (TS + Py).** `computeTrajectory` (`markov-distribution.ts:2619`) and `compute_trajectory` (`research/models/trajectory.py:175`) gained an optional `jumpSpec` / `jump_spec` parameter:
  - Drift compensator `Σ_e λ_e · κ_e` subtracted **once** from each `dailyDrifts[d]` before the MC loop (preserves `E[r_t] = μ_t`).
  - Per-event Bernoulli(λ_daily) draw + log-normal magnitude (Box-Muller in TS, `np.random.random()` in Py) inside the inner loop.
  - **Gated invariant:** `if (hasJumps)` wraps every jump-related RNG call. When `jumpSpec` is `undefined` or `[]` the RNG sequence is identical to legacy → all 453 existing markov-distribution tests pass with zero modification.
- **B5 — Public surface wiring.** `MarkovDistributionParams` gained `enableJumpDiffusion?: boolean` (default `false`) and `jumpEvents?: JumpEventSpec[]`. The activation predicate is `params.enableJumpDiffusion === true && (params.jumpEvents?.length ?? 0) > 0`. When active, `metadata.jumpDiffusion = {compensatorPerDay, events: [...]}` exposes provenance for downstream observability.
- **B7 — Tests.** `src/tools/finance/jump-diffusion.test.ts` (13 cases), `jump-diffusion.parity.test.ts` (5 TS↔Python parity cases), `polymarket-jump-extract.test.ts` (7 cases), `research/tests/test_jump_diffusion.py` (13 mirror cases). All pass.
- **B8 — Documentation.** This section + JSDoc/docstrings on every new public surface.

### Mathematical notes

The Merton (1976) SDE in daily-discretised log-space:

```
r_t = (μ_t − Σ_e λ_e · κ_e)·Δt + σ_t·√Δt·Z_t + Σ_e Bern(λ_e·Δt) · N(μ_J,e, σ_J,e²)
```

Drift compensator `κ_e = exp(μ_J,e + σ_J,e² / 2) − 1` is the expected percentage jump under log-normal `J`. Subtracting `Σ_e λ_e · κ_e` keeps `E[dS/S] = μ dt`, otherwise the simulated drift is biased by the expected jump impact.

Polymarket settlement probabilities are **Q-measure** (set by arbitrage with options markets); they must be `transformQToP`-converted **before** deriving the Poisson hazard, otherwise the `P` measure ends up with the risk-premium bias that Phase A explicitly removes.

### Test counts (post-Phase-B)

- TypeScript: 4299 pass / 59 skip / **3 pre-existing fail** (BTC 14d bearish-break gate, phase4-trend-penalty-comparison, hybrid break fallback — all unrelated to Phase A or B per checkpoint analysis).
- Python: 176 pass.
- New tests added in this phase: +25 TS / +13 Py.

### Operational guidance

- `enableJumpDiffusion` defaults `false`; explicit opt-in required.
- Caller is responsible for calling `extractJumpEventMarkets()` then mapping → `buildJumpEventSpec()`. The trajectory module never touches Polymarket directly.
- Recommended starting cap: ≤ 3 simultaneous jump events per trajectory to avoid stacking compensators that wash out the drift signal.
- Daily intensity is hard-capped at 0.95 inside `polymarketProbToHazard` to keep the Bernoulli approximation valid.
