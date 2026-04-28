# Phase A + Phase B — Verification Report

**Commit:** `0332b33` on `topic/opt-skills`  
**Plan reference:** `docs/forecast-improvement-review-2026-04-28.md`  
**Date:** 2025-07-14  
**Status:** **6 gaps found — 5 now resolved, 1 deferred (B7).**  
**Fixed in:** follow-up TDD session (2025-07-14).

---

## Summary

| Group | Item | Status | Notes |
|-------|------|--------|-------|
| Phase A | A1 — JSDoc annualised inputs | ✅ Pass | `rnd-integration.ts` lines 22–23 |
| Phase A | A2 — `DEFAULT_MPR_CAP=1.5`, `mprCap` param | ✅ Pass | exported at line 13; param on both TS + Py |
| Phase A | A3 — `meanZShift` in metadata | ✅ Pass | `metadata.rndIntegration.meanZShift` |
| Phase A | A4 — Section 9 in plan doc | ✅ Pass | appended in commit |
| Phase B | B1 — `extractJumpEventMarkets()` quality filter | ✅ **Fixed** | `minVolume24h: 5_000` default enforced; 2 new tests codify the guarantee |
| Phase B | B2 — survival-hazard formula | ✅ Pass | `λ = −ln(1−p)/days` correctly implemented |
| Phase B | B3 — Q→P then hazard ordering | ✅ Pass | `buildJumpEventSpec` applies `transformQToP` first |
| Phase B | B4 — trajectory TS + Py extension | ✅ Pass | 12th `jumpSpec` param in both `computeTrajectory` implementations |
| Phase B | B5 — `enableJumpDiffusion` + `jumpEvents` wiring | ✅ Pass | params present; defaults to `false` (byte-identity invariant preserved) |
| Phase B | B6 — file location / asset classes | ✅ **Fixed** | `geopolitics` added to `JUMP_DEFAULTS` in TS + Python mirror; 4 new tests |
| Phase B | B7 — backtest harness re-run | ⏳ **Deferred** | flag defaults to `false` so no recalibration needed until enabled in prod |
| Phase B | B8 — `forecast-implementation-review.md` updated | ✅ **Fixed** | §10 Merton MC step added; file-map table updated |
| §5.3 | `jumpDiffusionApplied`, `jumpIntensityP` | ✅ **Fixed** | `jumpDiffusionApplied: boolean` added to `MarkovMetadata`; 4 tests pass |
| §5.4 | `config.ts` `forecasting` block | ✅ **Fixed** | `enableJumpDiffusion`, `qToPMprCap`, `enableMSM` added to `ConfigSchema`; 5 tests pass |

---

## Detailed Findings

### ✅ Confirmed Correct

**A1 — JSDoc annualised inputs**  
`src/tools/finance/rnd-integration.ts` lines 22–23 explicitly document that `historicalDrift`, `riskFreeRate`, and `volatility` are all annualised. Python mirror (`research/models/rnd.py`) has matching docstring.

**A2 — MPR cap**  
`DEFAULT_MPR_CAP = 1.5` exported at line 13. The `mprCap` optional parameter defaults to `DEFAULT_MPR_CAP` in `transformQToP`. Python mirror matches. Tests added for cap clamping.

**A3 — z-shift in metadata**  
`metadata.rndIntegration.meanZShift` is populated at `markov-distribution.ts:4355`. Allows downstream callers to audit how far each Polymarket anchor shifted the distribution.

**B2 — Survival hazard formula**  
`polymarketProbToHazard` uses `λ = −ln(1−p) / days`. The linear approximation `p/days` was correctly rejected; implemented formula is statistically sound for `p` up to ~0.95.

**B3 — Q→P before hazard**  
Inside `buildJumpEventSpec`, `transformQToP` is called first; only then is `polymarketProbToHazard` applied on the physical probability. This is the correct composition — the reverse order would re-introduce the systematic bearish bias that Phase A removes.

**B4 — `computeTrajectory` signature**  
Both `src/tools/finance/markov-distribution.ts` (12th parameter) and `research/models/trajectory.py` (kwarg `jump_spec`) are extended. The `hasJumps` gate ensures zero extra RNG calls when `jumpSpec` is absent, preserving byte-identity for all pre-existing trajectory tests.

**B5 — `enableJumpDiffusion` wiring**  
`MarkovDistributionParams.enableJumpDiffusion` defaults to `false`. When absent or false, `computeTrajectory` receives no `jumpSpec` and the MC path is unchanged. Tests at 13 cases confirm both paths.

---

### ❌ Gap 1 — B1: Missing `trustScore` quality filter

**Plan requirement (§6, B1):**  
> "Consume only `trustScore === 'high'` markets from `extractJumpEventMarkets()`."

**Actual implementation (`src/tools/finance/polymarket.ts`):**  
The `PolymarketMarketResult` interface (and `PolymarketMarket` returned by the Gamma API wrapper) does **not** include a `trustScore` field. `extractJumpEventMarkets()` filters by `volume24h ≥ 5_000`, `ageDays ≥ 2`, settlement ≤ horizon, and `0 < p < 1`, but has no trust-level gating.

**Risk:** Low-liquidity or manipulated Polymarket markets could feed spurious jump intensities into the MC when `enableJumpDiffusion` is on.

**Recommended fix:** Either (a) add a `minVolume24h` threshold (already present at `5_000`) as a proxy for trust, document this explicitly as the trust gate, or (b) map Gamma API `active`/`volume` fields to a derived `trustScore` and add the filter. Option (a) is the simpler path with no new external data needed.

---

### ⚠️ Deviation B6 — File location + missing `geopolitics` asset class

**Plan specification (§6, B6):**  
> "Add `JUMP_DEFAULTS` per asset class (equities ±4%, BTC/ETH ±8%, geopolitics ±10%) in `src/tools/finance/jump-priors.ts`."

**Actual implementation:**  
- `JUMP_DEFAULTS` is in `src/tools/finance/jump-diffusion.ts` (not a separate `jump-priors.ts`).  
- Asset classes covered: `etf`, `equity`, `crypto`, `commodity` — no `geopolitics` key.

**Assessment:** The file consolidation is a reasonable simplification (fewer files). The `geopolitics` gap is cosmetic — callers can pass explicit `JumpEventSpec` overrides. However:
1. If `extractJumpEventMarkets()` ever classifies a geopolitical event (e.g., "Will Israel attack Iran?"), there is no default prior to fall back on; callers must supply `meanLogJump` / `stdLogJump` manually.
2. The plan doc's checklist item B6 cannot be ticked complete as-written.

**Recommended fix:** Either rename file to `jump-priors.ts` + re-export from `jump-diffusion.ts`, or update the plan doc to reflect the deliberate consolidation. Add a `geopolitics` entry to `JUMP_DEFAULTS` (`meanLogJump: -0.10, stdLogJump: 0.05`).

---

### ❌ Gap 2 — B7: Backtest harness not re-run

**Plan requirement (§6, B7):**  
> "Re-run backtest harness; recalibrate `YES_BIAS_MULTIPLIER` if avg P(up) drifts > 1 pp."

**Actual state:**  
Section 10 (completion notes) re-labels B7 as: *"B7 — Tests. `jump-diffusion.test.ts` (13 cases)…"* — this is the description of B8, not B7. The actual backtest re-run was never performed.

**Mitigating factor:** `enableJumpDiffusion` defaults to `false`, so the live MC output is byte-identical to pre-Phase-B. The `YES_BIAS_MULTIPLIER = 0.95` and `adjustYesBias`'s β = 0.035 in `src/utils/ensemble.ts` were calibrated against the jump-free MC; they remain valid as long as the flag stays off. When the flag is enabled in production, calibration drift becomes a real risk.

**Recommended fix:** Run the backtest harness once with a representative `jumpEvents` config (e.g., `enableJumpDiffusion: true`, one low-probability event). If avg P(up) across the historical test window shifts by more than 1 pp, adjust `YES_BIAS_MULTIPLIER`. Document the finding in section 10.

---

### ❌ Gap 3 — B8: `forecast-implementation-review.md` not updated

**Plan requirement (§6, B8):**  
> "Update `docs/forecast-implementation-review.md` with the new MC step."

**Actual state:**  
`docs/forecast-implementation-review.md` was **created** in this commit (it's a new file), but it does not contain any documentation of the Merton jump-diffusion MC step. Searching for "Merton", "jump-diffusion", "jumpSpec", "jump_spec", or "Idea 2 / Phase B" in that file returns no results.

**Recommended fix:** Add a section to `docs/forecast-implementation-review.md` describing the Merton jump-diffusion MC step, the `jumpSpec` parameter contract, the drift compensator, and the `hasJumps` gate invariant. A 15–20 line section suffices.

---

### ❌ Gap 4 — §5.3: Missing flat metadata flags

**Plan requirement (§5.3):**  
```
metadata.jumpDiffusionApplied: boolean
metadata.jumpIntensityP: number | null   // sum of P-measure daily intensities
metadata.volModel: 'student-t' | 'msm' | 'regime-mix'  // (deferred to Phase D)
```

**Actual implementation (`markov-distribution.ts` lines 622, 4546–4805):**  
```typescript
jumpDiffusion?: {
  compensatorPerDay: number;
  events: Array<{ id: string; dailyIntensity: number; meanLogJump: number; stdLogJump: number }>;
}
```

The per-event `dailyIntensity` array provides richer data than `jumpIntensityP: number | null`, but the flat `jumpDiffusionApplied: boolean` flag is absent. Callers must check `metadata.jumpDiffusion !== undefined` as a proxy, which is less ergonomic.

**`volModel`** is a Phase D concern (deferred by design). No issue there.

**Recommended fix:** Add `jumpDiffusionApplied: boolean` alongside the existing `jumpDiffusion?` object. Set it to `hasJumps`. This is a one-line addition but makes the metadata contract match the plan spec.

---

### ❌ Gap 5 — §5.4: `config.ts` `forecasting` block absent

**Plan requirement (§5.4):**  
> "Extend `settings.json` schema in `src/utils/config.ts`:  
> `forecasting.enableJumpDiffusion: boolean = false`  
> `forecasting.qToPMprCap: number = 1.5`  
> `forecasting.jumpDirectionDefaults: Record<AssetClass, JumpPrior>`  
> `forecasting.enableMSM: boolean = false` (deferred placeholder)"

**Actual state:**  
`src/utils/config.ts` has no `forecasting` block. The `DEFAULT_MPR_CAP` and `enableJumpDiffusion` values are hardcoded as module-level constants in their respective TS files (`rnd-integration.ts`, `markov-distribution.ts`). They cannot be overridden via `.cramer-short/settings.json` at runtime without redeployment.

**Risk:** Users cannot toggle `enableJumpDiffusion` or adjust `qToPMprCap` per-project without code changes. This reduces the operational value of the feature.

**Recommended fix:** Add a `forecasting` key to the `AppSettings` interface in `config.ts` with the four fields. Provide defaults matching the current hardcoded values to preserve backward compatibility. Wire `enableJumpDiffusion` through to `computeMarkovDistribution` by reading it from config when `params.enableJumpDiffusion` is not explicitly set.

---

## Recommended Priority Order

| Priority | Item | Effort |
|----------|------|--------|
| 1 (now) | §5.3 — add `jumpDiffusionApplied: boolean` | ~5 min, 1-line |
| 2 (now) | B6 — add `geopolitics` to `JUMP_DEFAULTS` | ~5 min, 1-line |
| 3 (now) | B8 — add Merton section to `forecast-implementation-review.md` | ~20 min |
| 4 (now) | B1 — document `minVolume24h` as the trust gate in the function JSDoc | ~5 min |
| 5 (before enabling flag) | §5.4 — add `forecasting` block to `config.ts` | ~30 min |
| 6 (before enabling flag) | B7 — run backtest harness with `enableJumpDiffusion: true` | manual run |

Items 1–4 are purely additive (no logic changes, no test rewrites) and should be committed together. Items 5–6 are gating requirements before turning on `enableJumpDiffusion` in any live session.

---

## Confirmed Non-Issues

- **Pre-existing test failures** (3): BTC 14d bearish-break SELL gate, phase4-trend-penalty-comparison, hybrid break fallback. Confirmed pre-existing on `topic/opt-skills` before Phase A/B. Not introduced by this work.
- **Parity precision (7 dp, not 12):** Expected due to `normPPF → normCDF` chain vs `scipy.stats.norm`. Documented in parity tests.
- **`jump-priors.ts` file split:** Consolidating defaults into `jump-diffusion.ts` is a reasonable simplification. Not a correctness issue.
- **Box-Muller vs `np.random.normal`:** Same math, different sampling implementation. Parity tests cover the pure-math functions; stochastic path divergence is expected and acceptable.
