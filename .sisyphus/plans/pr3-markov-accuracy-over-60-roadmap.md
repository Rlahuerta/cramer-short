# PR3 roadmap — path to BTC short-horizon directional accuracy above 60%

## Goal

Raise **both BTC-USD 7d and BTC-USD 14d full-coverage directional accuracy above 60%** on one canonical backtest harness.

This roadmap is explicitly about **measured directional accuracy**, not vague model quality claims. It is also explicitly grounded in what has already been tried and falsified:

- PR3B calibration-weight / kappa tuning did **not** solve the problem
- PR3C base-rate floor tuning did **not** solve the problem
- PR3D raw-vs-calibrated decision switching did **not** solve the problem

The remaining path is:

1. freeze a single evaluation contract,
2. improve **upstream crypto short-horizon signal quality**,
3. add **replayable external-signal evaluation**,
4. only then claim any additive Polymarket lift.

---

## Frozen metric contract

Four distinct things are tracked and must never be conflated in PRs or review comments:

| Term | Definition |
|------|------------|
| **full-coverage directional accuracy** | accuracy over every prediction, including HOLD behavior as scored by the backtest |
| **selective directional accuracy** | accuracy restricted to predictions above a declared confidence threshold; coverage and selected count must always be reported |
| **pUpDirectionalAccuracy** | directional accuracy computed from the sign of `P(up) - 0.5`, bypassing the recommendation path |
| **effective decision utility** | direction quality combined with abstain behavior, edge, and calibration; useful diagnostically, but not the success metric for PR3 |

Success for this roadmap means:

- **BTC-USD 7d > 60% full-coverage directional accuracy**
- **BTC-USD 14d > 60% full-coverage directional accuracy**
- both measured on the **same canonical harness**
- with no major Brier / reliability collapse

A >60% claim that relies only on selective accuracy, aggregate numbers, or unreplayed external signals is not a valid PR3 success claim.

---

## Verified baseline and ablation ceilings

These numbers come from passing integration tests and completed ablations.

| Surface | Harness | BTC 7d | BTC 14d | Verdict |
|---------|---------|--------|---------|---------|
| Current working Markov baseline | PR3A/PR3B/PR3C harness (`WARMUP=120`, `STRIDE=10`) | **44.3%** | **46.7%** | baseline remains weak |
| PR3B conditional-weight / kappa ablation | same harness | **47.5%** best | **46.7%** best | improved 7d slightly, failed 14d |
| PR3C base-rate floor ablation | same harness | **49.2%** best | **46.7%** best | improved 7d modestly, failed 14d |
| PR3D raw-decision ablation | alternate comparison block | **45.5%** vs default **47.1%** | **45.8%** vs default **45.8%** | eliminated HOLDs, no useful lift |
| PR3G crypto state-model improvement | canonical BTC harness (`WARMUP=120`, `STRIDE=10`) | **52.5%** | **48.3%** | best internal-only ceiling, still below the PR3 target |
| PR3I replayed Polymarket additive evaluation | same harness | **52.5% → 52.5%** | **48.3% → 46.7%** | replay-safe evaluation showed no additive lift |

Additional verified observations:

- BTC weak performance is concentrated in short-horizon crypto slices, especially **sideways 7d** and **bull / overconfident 14d** behavior
- the calibrated `>0.55` bucket is still weak on BTC 14d
- PR3H replay infrastructure now reproduces the canonical Markov-only baseline and rejects future-dated replay input
- bounded downstream decision/calibration levers and bounded internal state-model tuning are now **ceiling-documented**
- PR3I replayed, quality-gated Polymarket affected only a small subset of BTC steps and did not improve both horizons
- post-PR3 internal experiments did not move the directional ceiling: start-state mixture was null, sideways split was negative, and mature-bull calibration only improved BTC 14d Brier from **0.2976 → 0.295** on **4/60** activated steps with no directional lift

---

## Tested thesis and verdict

The previous roadmap assumed the next wins were likely to come from thresholding, calibration, floor tuning, or raw-vs-calibrated decision routing.

That thesis has now been tested.

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| crypto short-horizon calibration is over-compressed by weight / kappa tuning | **tested, insufficient** | PR3B lifted 7d slightly, left 14d below baseline target |
| base-rate floor is the main blocker | **tested, insufficient** | PR3C lifted 7d to 49.2%, left 14d flat |
| calibrated decision routing is the main blocker | **tested, insufficient** | PR3D raw-decision ablation was neutral-to-negative |
| replayed, quality-gated Polymarket signals add value beyond the best internal-only model | **tested, insufficient** | PR3I left BTC 7d flat at 52.5% and regressed BTC 14d to 46.7% |

Current verdict:

- PR3 is closed with a negative result: neither BTC short-horizon slice reached the roadmap success bar of **>60%** full-coverage directional accuracy
- the best internal-only ceiling is PR3G at **52.5%** (BTC 7d) and **48.3%** (BTC 14d) on the canonical harness
- replay-safe external-signal evaluation now exists, but PR3I showed no additive Polymarket lift beyond that ceiling
- the PR3I stop/go gate is triggered: do not claim PR3 success or merge external-signal complexity into the production path under a lift claim

---

## Blocking constraints

### 1. `walkForward` started price-only, but replay is now available

`walkForward` began as a price-only harness. PR3H added deterministic replay via `walkForwardWithReplay(...)`, committed replay fixtures, empty-replay baseline parity, and future-dated signal rejection, so external-signal evaluation is now measurable without future leakage.

### 2. PR3D was intentionally a decision-layer-only experiment

The raw-decision ablation changed `actionSignal` generation while keeping calibrated `distribution`, `scenarios`, Brier, reliability, and CI reporting intact. That was correct for diagnosis, but it is **not** a production-ready mixed mode and should not become the canonical evaluation surface.

### 3. Do not keep sweeping exhausted levers

Do **not** spend more time on:

- `conditionalWeight` sweeps
- `kappaMultiplier` sweeps
- floor clamp / bear-margin sweeps
- raw-vs-calibrated decision toggles

Those levers now have measured ceilings.

---

## Verified function inventory and current status

| Function / surface | Current status | Next action |
|--------------------|----------------|-------------|
| `computePredictionConfidence` | ablated indirectly; not the main fix | kept stable through PR3I |
| `computeActionSignal` | ablated directly; ceiling documented | kept stable except provenance / harness work |
| `calibrateProbabilities` | ablated indirectly via PR3B / PR3C / PR3D | no more tuning sweeps |
| `computeBaseRateFloor` | ablated directly; ceiling documented | no more tuning sweeps |
| `computeRegimeUpRates` | reworked in PR3F / PR3G, still below the PR3 target | internal ceiling documented |
| `classifyRegimeState` / `estimateTransitionMatrix` | reworked for crypto recency behavior in PR3G, still below the PR3 target | internal ceiling documented |
| `walkForward` | canonicalized in PR3E and paired with replay-safe evaluation in PR3H | closed for PR3 |
| `extractSignals` | exercised in PR3I signal-quality validation | no additive lift claim |
| `impact-map` / delta mapping | exercised in PR3I signal-quality validation | no additive lift claim |
| `polymarket-forecast` quality grading | exercised in PR3I QA | no additive lift claim |
| `polymarket-injector` | exercised in PR3I QA | no additive lift claim |
| `fetchPolymarketAnchorMarkets` | replayed through PR3H / PR3I evaluation | no additive lift claim |

---

## Completed work to date

| PR | Status | What it established |
|----|--------|---------------------|
| **PR3A** | completed | froze richer BTC diagnostics, added raw-vs-calibrated tracking, and exposed the short-horizon weakness clearly |
| **PR3B** | completed, negative | conditional-weight / kappa tuning does not solve both horizons |
| **PR3C** | completed, negative | floor tuning does not solve both horizons |
| **PR3D** | completed, negative | raw-decision routing does not solve both horizons |
| **PR3E** | completed | froze the canonical BTC harness and provenance reporting surface |
| **PR3F** | completed, negative | crypto short-horizon meta-prior work did not clear the continuation bar |
| **PR3G** | completed, negative | crypto state-model changes produced the best internal-only ceiling at **52.5% / 48.3%**, still below target |
| **PR3H** | completed | added replay-safe historical Polymarket evaluation with empty-replay baseline parity and future-date rejection |
| **PR3I** | completed, negative | replayed, quality-gated Polymarket was flat on BTC 7d and worse on BTC 14d, so additive lift was not established |

PR3 is closed with a negative result. The directional-accuracy goal (BTC 7d and BTC 14d both above 60%) was not reached by any stage. Any continuation should treat PR3G as the best internal-only ceiling and PR3I as a replay-safe negative external-signal result.

## Post-PR3 experiments (terminal)

Three bounded internal experiments were run after PR3I was closed. All are terminal and none changed the PR3 directional ceiling.

| Experiment | Flag / surface | Canonical BTC result | Verdict |
|------------|----------------|----------------------|---------|
| Start-state mixture | `startStateMixture` | 7d `52.5% → 52.5%`, 14d `48.3% → 48.3%`, no Brier change | **null** |
| Sideways split | `sidewaysSplit` | 7d `52.5% → 47.5%`, 14d `48.3% → 45.0%`, Brier worsened at 14d (`0.2976 → 0.308`) | **negative** |
| Mature bull calibration | `matureBullCalibration` | 14d `48.3% → 48.3%`, calibrated P(up) `46.7% → 46.7%`, Brier `0.2976 → 0.295`, active `4/60` | **terminal, modest calibration-only improvement** |

These flags remain useful as reviewer-visible experiment seams, but none should be treated as a production-path win or as evidence that the PR3 target was close.

---

## Historical execution roadmap (completed)

The stage sections below preserve the original PR3E–PR3I scope definitions as historical planning context. Actual outcomes are recorded in the completed-work tables and status tables in this document; they are not pending work.

## PR3E — canonical evaluation contract

### Objective

Standardize every future PR3 experiment on one BTC short-horizon harness so results are directly comparable.

### Why this is next

The PR3A/PR3B/PR3C work uses one BTC sampling shape, while PR3D used another. That is why there are two different default baselines in the logs. The next modeling PR must not inherit that ambiguity.

### Files

- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/backtest/metrics.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`
- `src/tools/finance/markov-distribution.ts`

### Planned changes

- freeze one canonical BTC harness:
  - `ticker = BTC-USD`
  - `horizon in {7, 14}`
  - `WARMUP = 120`
  - `STRIDE = 10`
- align the PR3D comparison block to that same harness or remove the alternate block
- add explicit provenance fields:
  - `decisionSource: 'calibrated' | 'raw' | 'hybrid'`
  - `probabilitySource: 'calibrated' | 'raw'`
- thread provenance into `BacktestStep` and reporting output
- freeze one reviewer-visible baseline table for all future PR3 comparisons

### TDD sequence

1. add a failing test that asserts the canonical BTC harness constants are shared by all PR3 blocks
2. add a failing test that asserts `decisionSource` / `probabilitySource` are present in the reported steps or summary path
3. implement the smallest harness/provenance change that satisfies both
4. rerun the integration suite and confirm the canonical BTC baseline table is stable

### Acceptance criteria

- all PR3 comparison blocks use the same BTC harness
- the baseline table is printed exactly once in the canonical format
- default Markov behavior is unchanged except for added provenance fields

### Stop/go gate

Do not proceed to PR3F until there is one canonical BTC scoreboard. If two PR3 blocks still report different default baselines because of harness drift, PR3E is not complete.

### QA commands

```bash
bun test src/tools/finance/backtest/metrics.test.ts
bun test src/tools/finance/markov-distribution.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

### Commit boundary

One PR for harness/provenance freezing only. No model logic changes.

---

## PR3F — crypto short-horizon meta-prior

### Objective

Improve BTC 7d/14d directional separation by adding a crypto `<=14d` prior built from signals already computed inside the Markov path.

### Why this is next

The bounded downstream levers are exhausted. The next internal step is to improve the **upstream directional prior**, not to keep tuning the last-mile thresholds.

### Files

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/markov-distribution.test.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`

### Primary function targets

- new helper such as `computeCryptoShortHorizonPrior(...)`
- `computeRegimeUpRates`
- existing short-horizon crypto logic inside `computeMarkovDistribution`

### Planned changes

- add a flagged crypto-short-horizon prior that uses only existing signals:
  - `rawPUp`
  - `conditionalPUp`
  - `baseRate`
  - `regimeState`
  - `regimeRunLength`
  - `momentumAgreement`
  - `ensembleConsensus`
  - `structuralBreak`
  - `recentVol`
  - `outOfSampleR2`
- scope it to `assetType === 'crypto' && horizon <= 14`
- use it to improve the internal directional center / prior before the final decision layer
- keep non-crypto behavior unchanged
- keep the current calibrated reporting surfaces intact

### TDD sequence

1. add unit tests for scope, monotonicity, and deterministic fixture behavior of the new prior
2. add an integration comparison block that prints baseline vs meta-prior BTC 7d/14d results on the canonical harness
3. implement the smallest flagged prior that satisfies the tests
4. verify that non-crypto surfaces remain unchanged

### Acceptance criteria

- the new prior is BTC-short-horizon-scoped and flag-protected
- canonical BTC output shows a material improvement over PR3E baseline
- non-crypto integration behavior does not regress

### Stop/go gate

Proceed to PR3G only if the best PR3F candidate shows either:

- at least **+3pp** lift on both BTC horizons, or
- one horizon reaching **>=55%** without the other regressing by more than **1pp**

If no PR3F candidate meets that bar, document the ceiling and move directly to PR3H/PR3I rather than stacking more meta-prior variants.

### QA commands

```bash
bun test src/tools/finance/markov-distribution.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

### Commit boundary

One PR for the flagged prior and its comparison harness. Do not bundle unrelated calibration or replay work.

---

## PR3G — crypto state-model improvement

### Objective

If PR3F is not enough, improve the **raw crypto short-horizon signal** by changing the state / transition model itself.

### Why this is later than PR3F

This is a deeper model-path change. It should happen only after the lighter-weight prior has been tested, because state-model changes affect trajectory behavior and non-BTC surfaces more easily.

### Files

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/hmm.ts`
- `src/tools/finance/markov-distribution.test.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`
- `src/tools/finance/markov-trajectory.integration.test.ts`

### Primary function targets

- `computeRegimeUpRates`
- `classifyRegimeState`
- `estimateTransitionMatrix`
- crypto HMM weighting / bypass logic

### Planned changes

- add recency weighting for crypto short-horizon transition estimation
- add recency weighting or smoothing for crypto short-horizon `computeRegimeUpRates`
- review HMM participation for `crypto <= 14`:
  - if convergence is usually poor, downweight or bypass it for this surface
  - do not broaden the change to non-crypto assets
- keep the richer state behavior BTC-limited until the evidence justifies generalization

### TDD sequence

1. add unit tests for transition-matrix invariants under recency weighting
2. add unit tests for crypto-scoped HMM gating behavior
3. add a canonical BTC integration comparison block for the new state-model candidate
4. rerun trajectory tests if the state path changes

### Acceptance criteria

- BTC short-horizon direction improves beyond the best PR3F candidate
- trajectory behavior remains valid
- the change is still clearly attributable to one bounded state-model idea

### Stop/go gate

If BTC 7d or BTC 14d still remains below **55%** after the best PR3G candidate, stop bounded internal tuning. At that point the remaining path is external-signal replay and additive evaluation, not more internal Markov tweaking.

### QA commands

```bash
bun test src/tools/finance/markov-distribution.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-trajectory.integration.test.ts
bun run typecheck
```

### Commit boundary

One PR for one state-model hypothesis. Never bundle HMM changes, regime changes, and external-signal work in the same PR.

---

## PR3H — Polymarket replay infrastructure

### Objective

Build a deterministic historical replay path so external signals can be evaluated without future leakage.

### Why this is required

The current backtest path is price-only. Until replay exists, external-signal claims remain unmeasured.

### Files

- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/polymarket.ts`
- `src/tools/finance/polymarket-injector.ts`
- committed replay fixture/schema paths under `src/tools/finance/fixtures/`

### Planned changes

- add replay-compatible historical signal input to `walkForward` or a parallel backtest harness
- add a replay path for historical Polymarket snapshots / anchor inputs
- keep live fetch logic separate from replay logic
- guarantee deterministic fixture loading and no future data leakage
- preserve the ability to reproduce the Markov-only baseline when replay input is empty
- define and commit one canonical schema for timestamped historical signal records before replay logic is generalized
- commit at least one deterministic historical Polymarket fixture/snapshot set that is sufficient to run the replay harness in CI

### TDD sequence

1. add a failing test that asserts the replay harness accepts a documented timestamped historical signal schema
2. add a failing test that asserts empty replay input reproduces the canonical Markov-only baseline
3. add a failing test that rejects future-dated signal records
4. implement the smallest replay path that satisfies all three

### Acceptance criteria

- replay harness accepts deterministic historical signal input
- empty replay mode reproduces the canonical Markov-only baseline
- future leakage is explicitly tested and rejected
- the historical signal schema and at least one deterministic replay fixture are committed in-repo

### Stop/go gate

Do not proceed to PR3I until the replay harness reproduces the canonical Markov-only baseline. If replay changes baseline behavior before any external signal is layered on, PR3H is not complete.

### QA commands

```bash
bun test src/tools/finance/markov-distribution.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

Expected results:

- replay-schema tests pass
- future-dated signal rejection tests pass
- empty replay input reproduces the canonical Markov-only baseline
- typecheck passes

### Commit boundary

One PR for replay infrastructure only. No signal-quality or model-lift claims in the same PR.

---

## PR3I — Polymarket signal quality and additive evaluation

### Objective

Measure whether replayed, quality-gated Polymarket signals add directional value beyond the best internal-only model.

### Files

- `src/tools/finance/signal-extractor.ts`
- `src/tools/finance/signal-extractor.test.ts`
- `src/tools/finance/impact-map.ts`
- `src/tools/finance/impact-map.test.ts`
- `src/tools/finance/polymarket-forecast.ts`
- `src/tools/finance/polymarket-forecast.test.ts`
- `src/tools/finance/polymarket-injector.ts`
- `src/tools/finance/polymarket-injector.test.ts`
- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`

### Primary function targets

- `extractSignals`
- Polymarket quality grading in `polymarket-forecast`
- impact mapping / return-space translation
- quality filtering and dedup in `polymarket-injector`
- replay-compatible anchor blending inside `markov-distribution.ts`

### Planned changes

- add explicit, testable quality filters for replayed Polymarket signals
- validate ticker-to-query extraction and impact mapping before claiming any model lift
- compare these paths on the canonical BTC harness:
  - best internal-only model
  - internal model + replayed Polymarket signals
- report additive lift only after replay, quality filtering, and baseline reproduction are all in place

### Signal quality requirements

- persistence test for market shocks before treating them as signals
- YES-bias discounting where the existing design requires it
- cross-market divergence treated as a noise flag, not a second truth source
- Polymarket must never become a single source of truth
- all quality filters must be independently unit-tested

### TDD sequence

1. add failing unit tests for ticker/query extraction, impact mapping, and quality filters
2. add a failing integration comparison block for canonical BTC with and without replayed Polymarket signals
3. implement the smallest signal-quality improvements that satisfy the tests
4. only then evaluate additive lift against the best internal-only baseline

### Acceptance criteria

- signal-quality filters are independently tested
- replayed Polymarket signals show measurable additive value beyond the best internal-only model
- both BTC 7d and BTC 14d exceed **60%** before any PR3 success claim is made

### Stop/go gate

If replayed Polymarket signals do not lift both BTC horizons above the best internal-only ceiling, document the result and do not merge external-signal complexity into the production path under a PR3 success claim.

### QA commands

```bash
bun test src/tools/finance/signal-extractor.test.ts
bun test src/tools/finance/impact-map.test.ts
bun test src/tools/finance/polymarket-forecast.test.ts
bun test src/tools/finance/polymarket-injector.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

Expected results:

- ticker/query extraction tests pass
- impact mapping tests pass
- Polymarket forecast quality-grading tests pass
- Polymarket injector quality-filter tests pass
- canonical BTC integration output shows baseline vs replayed-Polymarket comparison explicitly
- typecheck passes

### Commit boundary

One PR for signal-quality surfaces, one PR for end-to-end additive evaluation if and only if the measured replay results justify it.

---

## Ablation discipline

Every active stage must report all of the following for BTC 7d and BTC 14d:

1. previous-stage baseline directional accuracy
2. candidate directional accuracy
3. `pUpDirectionalAccuracy`
4. recommendation-path accuracy
5. confidence-filtered accuracy at declared thresholds with coverage
6. hold / abstain rate
7. regime-specific breakdown
8. P(up) band breakdown: `<0.45`, `0.45–0.50`, `0.50–0.55`, `>0.55`
9. decision / probability provenance when mixed modes are involved

If a stage cannot beat the previous stage on the declared target metric, it stops and documents the ceiling. It does not proceed on wishful thinking.

---

## Atomic commit strategy

| PR | Status | Scope |
|----|--------|-------|
| PR3A | completed | metric freezing, BTC diagnostics, raw-vs-calibrated reporting |
| PR3B | completed | conditional-weight / kappa ablation |
| PR3C | completed | base-rate floor ablation |
| PR3D | completed | raw-decision ablation |
| PR3E | completed | canonical evaluation contract and provenance |
| PR3F | completed, negative | crypto short-horizon meta-prior |
| PR3G | completed, negative | crypto state-model improvement |
| PR3H | completed | replay infrastructure |
| PR3I | completed, negative | Polymarket signal quality and additive evaluation |
| Post-PR3 experiments | completed, terminal | start-state mixture (null), sideways split (negative), mature-bull calibration (Brier-only, no directional lift) |

Each PR gets one bounded hypothesis. Reporting changes, model changes, replay infrastructure, and external-signal changes are never bundled together.

---

## TDD expectation for every active stage

For every active stage, the implementation order is:

1. write or extend the failing unit / integration test for the BTC target slice first
2. implement the smallest code change that satisfies the stage goal
3. run the QA commands listed for that stage
4. inspect the BTC 7d / 14d output before declaring success
5. do not claim a win based on aggregate numbers if the BTC-specific slice is not shown

---

## Merge-readiness QA command set

```bash
bun test src/tools/finance/markov-distribution.test.ts
bun test src/tools/finance/backtest/metrics.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

Add the trajectory suite only for stages that touch state-model or trajectory behavior:

```bash
RUN_INTEGRATION=1 bun test src/tools/finance/markov-trajectory.integration.test.ts
```

Add replay / signal-quality suites when PR3H or PR3I touches them.

For PR3H and PR3I, the merge-ready QA set must also include:

```bash
bun test src/tools/finance/polymarket-forecast.test.ts
bun test src/tools/finance/polymarket-injector.test.ts
```

---

## Staged success targets

| PR | Target metric | Threshold | Notes |
|----|---------------|-----------|-------|
| PR3A | diagnostics surface | completed | no accuracy target |
| PR3B | calibration lever | ceiling at 47.5% / 46.7% | documented negative result |
| PR3C | floor lever | ceiling at 49.2% / 46.7% | documented negative result |
| PR3D | raw-decision lever | 45.5% / 45.8% on comparison block | documented negative result |
| PR3E | canonical harness | completed | one frozen scoreboard with provenance added |
| PR3F | internal crypto prior | threshold not met | documented negative result |
| PR3G | internal state-model lift | ceiling at **52.5% / 48.3%** | best internal-only result, still below the >=55% stop/go bar |
| PR3H | replay baseline reproduction | completed | empty replay matches the canonical Markov-only baseline and future leakage is rejected |
| PR3I | final PR3 success | threshold not met | replayed Polymarket left BTC 7d flat at **52.5%** and BTC 14d worse at **46.7%** |
| Post-PR3: start-state mixture | bounded internal follow-up | null | no directional or Brier lift over PR3G |
| Post-PR3: sideways split | bounded internal follow-up | negative | worse than PR3G on canonical BTC 7d/14d |
| Post-PR3: mature bull calibration | bounded internal follow-up | terminal | small BTC 14d Brier improvement only (`0.2976 → 0.295`), no directional lift |

---

## Review questions

Reviewers should be able to answer for every PR:

1. Did BTC 7d and BTC 14d improve compared with the previous stage? Show both numbers explicitly.
2. Is the claim based on **full coverage** or a filtered subset? If filtered, what are the coverage and selected count?
3. Is the comparison being made on the **canonical harness**, or did the PR quietly change the sampling shape?
4. What single hypothesis is this PR testing? Is the gain attributable to that one change?
5. Is the claimed lift large enough and stable enough to be believable, or could it be noise from a small slice?
6. Does the PR build on the current roadmap stage, or does it reopen a lever that is already ceiling-documented?
7. If the PR mentions Polymarket or anchors, were those signals replayed historically in the evaluation used for the claim?
8. If mixed decision/probability provenance is involved, is that provenance shown explicitly so reviewers can tell what actually generated the recommendation?

If the answer to question 7 is "no," the PR must not claim that Polymarket improved the measured result.
