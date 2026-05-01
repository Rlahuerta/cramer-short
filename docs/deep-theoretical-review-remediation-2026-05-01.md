# Deep Theoretical Review Remediation — 2026-05-01

**Source review:** `docs/deep-theoretical-review-topic-opt-skills-2026-05-01.md`  
**Verification note:** `docs/deep-theoretical-review-verification-2026-05-01.md`

---

## Outcome

This remediation pass executed the two **live** follow-ups from the verification note and kept only the change that cleared the repo's safety gate.

| Phase | Status | Result |
| --- | --- | --- |
| Phase 1 — baseline freeze and documentation hygiene | **Done** | Baseline behavior and current-head audit note were frozen before code changes. |
| Phase 2 — conformal integral-decay evaluation | **Rejected** | A promoted `integralDecay = 0.95` default was tested and reverted. |
| Phase 3 — Student-t prior calibration hardening | **Kept** | Sparse-state prior hardening was implemented in `hmm.ts` and retained. |
| Phase 4 — final comparative verification | **Done** | Final keep/drop decision recorded here. |

---

## Phase 1 — Baseline freeze

Baseline validation completed successfully before any remediation work:

- `bun run typecheck`
- `bun test src/tools/finance/conformal.test.ts`
- `bun test src/tools/finance/hmm.test.ts`
- `bun test src/tools/finance/markov-distribution.test.ts`
- `bun test src/tools/finance/backtest/walk-forward-r5.test.ts`
- `RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts`

That baseline established:

1. the current `ConformalPID` default is still `integralDecay = 1.0`,
2. the current Student-t path is still opt-in via `enableStudentTEmission`,
3. there is still no current-head Markov-block conformal implementation.

---

## Phase 2 — Conformal integral-decay evaluation

### Candidate

The candidate changed the `ConformalPID` default from `1.0` to `0.95` to align more closely with the decaying-integral guidance in the cited conformal-PID paper.

### Result

**Rejected and reverted.**

### Why it was dropped

The candidate did **not** earn promotion:

1. It failed the repo's own PID convergence expectation in `src/tools/finance/conformal.test.ts`.
   - The Gaussian quantile-convergence check fell to:
     - expected radius `> 1.4`
     - observed radius `1.3551047290783742`
2. It did **not** produce a meaningful BTC fixture lift when compared against the prior `integralDecay = 1.0` behavior.
3. Since this was a default-policy change rather than a bug fix with obvious correctness impact, the lack of measurable benefit was enough to reject it.

### Final state after Phase 2

- `ConformalPID` default remains **`integralDecay = 1.0`**
- no conformal-default change is kept in the codebase from this phase

---

## Phase 3 — Student-t prior calibration hardening

### Candidate

The retained change hardens `attachStudentTPredictiveEmissions()` in `src/tools/finance/hmm.ts` by replacing the fixed prior constants with a small helper:

- `resolveStudentTPriorHyperparameters(observations, priorStd, effectiveSampleSize)`

The helper now:

1. **strengthens prior mean weight when a state is sparse** via a larger `priorKappa`,
2. **keeps thinner-tail priors for calmer series and heavier-tail priors for more kurtotic series** via an adaptive `priorAlpha`,
3. recomputes `priorBeta` from the adjusted hyperparameters.

### Retained files

- `src/tools/finance/hmm.ts`
- `src/tools/finance/hmm.test.ts`

### Why it was kept

This change is retained as a **robustness hardening**, not as a promoted forecasting lever:

1. It directly addresses the still-live theoretical concern about sparse-state prior sensitivity.
2. It passed the full validation gate:
   - `bun run typecheck`
   - focused HMM / Markov tests
   - `RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts`
3. It produced **no measurable regression** on the BTC fixture under the opt-in Student-t path.

### BTC fixture comparison vs prior Student-t baseline

Using `enableStudentTEmission: true`, `warmup=180`, and `stride=3`, the post-change metrics remained effectively unchanged:

| Horizon | covErr | ΔcovErr | breakCovErr | ΔbreakCovErr | sharp | Δsharp | brier | Δbrier | dir | Δdir |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1d | 0.0783 | +0.0000 | 0.0781 | +0.0000 | 0.1731 | -0.0001 | 0.2537 | +0.0000 | 0.5217 | +0.0000 |
| 7d | 0.0835 | +0.0000 | 0.0834 | +0.0000 | 0.3684 | +0.0002 | 0.2598 | +0.0000 | 0.5000 | +0.0000 |
| 14d | 0.0832 | +0.0000 | 0.0831 | +0.0000 | 0.4995 | -0.0003 | 0.2580 | +0.0000 | 0.5028 | +0.0000 |
| 30d | 0.0655 | +0.0000 | 0.0649 | +0.0000 | 0.6968 | +0.0004 | 0.2669 | +0.0000 | 0.4540 | +0.0000 |

### Interpretation

The retained change did **not** create a visible BTC fixture improvement, but it also did **not** damage calibration, sharpness, Brier score, or direction quality. That makes it acceptable as a narrow theoretical hardening for the opt-in Student-t path.

---

## Final keep/drop decision

### Keep

- **Student-t sparse-state prior hardening** in `src/tools/finance/hmm.ts`

### Drop

- **Conformal PID default decay promotion** (`integralDecay = 0.95`)

### Continue to exclude

- **Markov-block conformal** remains out of scope because it is not present in current HEAD.

---

## Final state

After this remediation pass:

1. `ConformalPID` still defaults to **`integralDecay = 1.0`**
2. Student-t predictive emissions now have **adaptive prior hyperparameters** for sparse-state hardening
3. all targeted checks and integration tests pass
4. the repo keeps only the change that cleared the safety gate
