# Verification Note: `deep-theoretical-review-topic-opt-skills-2026-05-01.md`

**Date:** 2026-05-01  
**Scope:** verify the findings in `docs/deep-theoretical-review-topic-opt-skills-2026-05-01.md` against current HEAD and outline a remediation plan for the issues that are still live.

---

## Executive summary

The source review is **useful but not fully current**. Two findings still apply to current HEAD:

1. **`ConformalPID` still defaults `integralDecay` to `1.0`** in `src/tools/finance/conformal.ts:90-99`.
2. **Student-t HMM emissions still use fixed weakly informative priors** in `src/tools/finance/hmm.ts:173-200`.

The main stale section is the report's **Markov-block conformal** analysis. There is **no current-head implementation** of `MarkovBlockConformalPID` or any other block-conformal surface in `src/tools/finance/`.

The practical read is:

- **Conformal integral decay** is a real follow-up and should be evaluated with tests and backtests before any default change.
- **Student-t prior sensitivity** is a real but moderate follow-up and should be hardened only if it survives the same keep/drop gate used for earlier literature phases.
- **Markov-block conformal** should be treated as **historical / reverted work**, not as an active issue in current code.

---

## Finding-by-finding verification

| Finding from source review | Current-head verdict | Evidence | Notes |
| --- | --- | --- | --- |
| `ConformalPID` default `integralDecay = 1.0` is theoretically weaker than the paper's decaying-integral guidance | **Valid** | `src/tools/finance/conformal.ts:41-42`, `src/tools/finance/conformal.ts:90-99`, `src/tools/finance/conformal.ts:119-126` | The code still uses `opts.integralDecay ?? 1.0`. The review is directionally right, but this is a tuning/default issue, not a proven correctness bug. |
| `MarkovBlockConformalPID` exists and uses a static block size not tied to mixing time | **Stale / not applicable** | Repo-wide current-head symbol search on 2026-05-01 found no `MarkovBlockConformalPID`, `MarkovBlock`, `block-conformal`, or `conformalBlockSize` implementation in `src/tools/finance/` | This section describes a prototype that is no longer present in current code. It should not be read as a live-code audit finding. |
| Student-t HMM emissions use fixed priors that may be brittle for sparse states | **Partially valid** | `src/tools/finance/hmm.ts:153-223`, especially `src/tools/finance/hmm.ts:173-200` | The fixed priors are real. The concern is legitimate as a Bayesian modeling follow-up, but current code already passed the repo-local keep/drop gate when the Student-t phase was accepted. |
| Walk-forward validation is theoretically sound | **Valid** | `src/tools/finance/backtest/walk-forward.ts:154-340` | Expanding-window evaluation, strict out-of-sample target, and the current conformal wiring are all present. |
| Metrics implementation is theoretically sound | **Valid** | Source review points to `metrics.ts`; this note did not re-audit those sections because no contradictory code changes were introduced later | No evidence from current HEAD suggests those positive findings are stale. |

---

## Phase 0: documentation discovery for remediation planning

This section freezes the **real extension points** that exist today so later implementation work does not plan against reverted or imagined APIs.

### Allowed APIs and live surfaces

- `ConformalPIDOptions.integralDecay` — `src/tools/finance/conformal.ts:28-43`
- `class ConformalPID` — `src/tools/finance/conformal.ts:75-153`
- `class AdaptiveConformalPID` — `src/tools/finance/conformal.ts:155-260`
- `attachStudentTPredictiveEmissions()` — `src/tools/finance/hmm.ts:153-223`
- `HMMParams.studentTEmissions` / `StudentTEmission` — `src/tools/finance/hmm.ts:20-44`
- `computeMarkovDistribution({ enableStudentTEmission })` — `src/tools/finance/markov-distribution.ts:4449-4450`, `src/tools/finance/markov-distribution.ts:4743-4745`
- `WalkForwardConfig.enableStudentTEmission` — `src/tools/finance/backtest/walk-forward.ts:121-123`

### Copy-ready verification surfaces

- `src/tools/finance/conformal.test.ts:147-155` already covers decaying-integral behavior when `integralDecay` is explicitly supplied.
- `src/tools/finance/conformal.test.ts:176-255` already exercises the adaptive conformal wrapper against the baseline PID.
- `src/tools/finance/hmm.test.ts:325-355` already covers finite Student-t parameters and heavier-tail behavior on outliers.
- `src/tools/finance/markov-distribution.test.ts` contains the accepted Student-t path tests that guard default-off behavior and enabled-path changes.
- `src/tools/finance/backtest/walk-forward.ts:308-340` is the live walk-forward conformal wiring that any conformal-default change must preserve.

### Anti-patterns to avoid

- Do **not** plan or code against `MarkovBlockConformalPID`; it is not present in current HEAD.
- Do **not** assume there is already a configurable Student-t prior API; there is not.
- Do **not** change both conformal defaults and Student-t priors in the same phase; they need separate keep/drop evidence.
- Do **not** treat literature alignment alone as acceptance; this repo already requires local backtest lift.

---

## Remediation plan

### Phase 1 — Documentation hygiene and baseline freeze

**Goal:** make the stale/non-stale split explicit before more code work starts.

**What to do**

1. Keep this verification note as the current-head reference for the source review.
2. Treat all Markov-block conformal items in the source review as historical context unless that feature is re-proposed and reimplemented.
3. Freeze a baseline snapshot for the two live follow-ups:
   - current `ConformalPID` default behavior,
   - current Student-t predictive-emission behavior.

**Verification checklist**

1. Repo search still shows no `MarkovBlockConformalPID` in current HEAD.
2. Existing conformal and Student-t tests pass unchanged.
3. Baseline backtest command and slices are recorded before any tuning starts.

**Anti-pattern guards**

- Do not start implementing a replacement block-conformal feature under the guise of "fixing" this review.
- Do not mix documentation cleanup with behavior changes.

### Phase 2 — Conformal integral-decay evaluation

**Goal:** decide whether the default should remain `1.0`, be changed to a decaying value such as `0.95`, or stay unchanged with explicit rationale.

**What to implement**

1. Add or extend tests around the constructor/default path in `src/tools/finance/conformal.test.ts` so the default is asserted explicitly.
2. Add one focused backtest/config arm that compares:
   - baseline default (`integralDecay = 1.0`)
   - candidate decaying default / explicit configuration (`integralDecay = 0.95` or similar)
3. If the decaying setting improves calibration without materially hurting sharpness or direction metrics, promote the change; otherwise, keep the current default and document why.

**Documentation references**

- `src/tools/finance/conformal.ts:28-43`
- `src/tools/finance/conformal.ts:90-127`
- `src/tools/finance/conformal.test.ts:147-155`
- `src/tools/finance/backtest/walk-forward.ts:308-340`

**Verification checklist**

1. Red/green test proving what the default is and how a decaying setting behaves.
2. `bun run typecheck`
3. Targeted conformal tests
4. Comparative backtest on the existing BTC walk-forward slices with coverage, sharpness, and directional metrics
5. Keep/drop decision recorded in `docs/`

**Anti-pattern guards**

- Do not change PID gains and `integralDecay` in the same phase.
- Do not declare success from synthetic coverage behavior alone; require walk-forward evidence.
- Do not silently change the default without updating the code comment and constructor contract.

### Phase 3 — Student-t prior-calibration hardening

**Goal:** determine whether sparse-state prior handling should stay fixed as-is or gain a narrowly scoped hardening path.

**What to implement**

1. Keep the accepted Student-t path intact and default behavior unchanged while evaluating changes.
2. Add tests that specifically stress low-effective-sample states in `src/tools/finance/hmm.test.ts`.
3. Prototype the smallest viable hardening mechanism, for example:
   - expose explicit prior constants in one helper,
   - or add a constrained empirical-Bayes-style prior adjustment based on the full return series,
   - but keep it isolated to `attachStudentTPredictiveEmissions()`.
4. Run the same keep/drop gate used for the earlier literature phase before keeping any prior retune.

**Documentation references**

- `src/tools/finance/hmm.ts:153-223`
- `src/tools/finance/hmm.test.ts:325-355`
- `src/tools/finance/markov-distribution.ts:4743-4760`
- `src/tools/finance/markov-distribution.ts:4803-4806`

**Verification checklist**

1. Red/green sparse-state tests for finite parameters and bounded behavior
2. `bun run typecheck`
3. Targeted HMM / Markov tests
4. Comparative backtest versus the currently accepted Student-t implementation
5. Keep/drop decision recorded in `docs/`

**Anti-pattern guards**

- Do not redesign the whole HMM stack to address a moderate prior-tuning concern.
- Do not change regime mapping semantics; the accepted Student-t predictive-mean path must remain intact.
- Do not default-enable a prior-retune variant without measurable forecast benefit.

### Phase 4 — Final acceptance record

**Goal:** keep only changes that improve the current forecast stack.

**What to implement**

1. Compare the following arms:
   - current HEAD baseline
   - conformal-decay candidate only
   - Student-t prior-hardening candidate only
   - combined candidate, only if both single-feature arms are individually promising
2. Publish a short docs report with:
   - metric deltas,
   - keep/drop decision per phase,
   - final defaults to retain.

**Verification checklist**

1. `bun run typecheck`
2. Targeted unit tests for every changed surface
3. Repo-standard backtest comparison
4. Written keep/drop record in `docs/`

**Anti-pattern guards**

- Do not keep a literature-backed change that fails the repo-local forecast gate.
- Do not combine both changes prematurely and lose attribution of gains/losses.

---

## Recommended execution order

1. Phase 1 — baseline freeze and documentation hygiene
2. Phase 2 — conformal integral-decay evaluation
3. Phase 3 — Student-t prior-calibration hardening
4. Phase 4 — comparative verification and final keep/drop record

---

## Bottom line

The source review is still valuable, but the active engineering backlog should be narrowed to **two** live items:

1. **Conformal PID default decay policy**
2. **Student-t sparse-state prior calibration**

Everything related to **Markov-block conformal** should remain out of scope unless that feature is deliberately reintroduced in a new phase.

---

## Post-remediation update

This note captured the state of current HEAD **before** the remediation pass began. After the follow-up work recorded in `docs/deep-theoretical-review-remediation-2026-05-01.md`:

1. **Conformal PID default decay** remains unresolved in code: `ConformalPID` still defaults `integralDecay` to `1.0` because the attempted `0.95` promotion failed the repo's own quantile-convergence gate and was reverted.
2. **Student-t sparse-state prior handling** is now partially addressed: `src/tools/finance/hmm.ts` no longer uses only the fixed `priorKappa = 0.01` / `priorAlpha = 2.0` path; it now derives Student-t prior hyperparameters from series kurtosis plus effective sample size via `resolveStudentTPriorHyperparameters(...)`.
3. **Markov-block conformal** remains stale / out of scope because no such implementation exists in current HEAD.
