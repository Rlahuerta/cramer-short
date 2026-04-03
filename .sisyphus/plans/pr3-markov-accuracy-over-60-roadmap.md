# PR3 roadmap — path to BTC short-horizon directional accuracy above 60%

## Goal

Create a staged implementation plan to push **BTC short-horizon directional accuracy above 60%** in Dexter’s Markov pipeline, using the current PR1 measurement layer and PR2 decision-layer improvement as the baseline.

This roadmap is explicitly about **measured directional accuracy**, not vague model quality claims.

## Current measured baseline

After PR1 measurement/reporting and the first PR2 decision-layer fix:

- BTC 7d directional accuracy improved from **33% → 44%**
- BTC 14d directional accuracy improved from **42% → 47%**
- the raw-vs-recommendation gap on the target slice narrowed materially
- weak performance remains concentrated in the **0.45–0.55 `P(up)` band**
- `horizon_return` slices remain materially weaker than `daily_return`
- regime-specific weakness remains, especially in certain sideways/bull windows

## Critical framing

Getting above **60% full-coverage directional accuracy** on BTC short horizons is unlikely to come from one more threshold tweak.

The strongest current evidence says the first credible >60% result will likely be **selective accuracy at declared coverage**, not immediate >60% full-coverage BTC direction.

A credible path must separate three targets:

1. **full-coverage directional accuracy** — every prediction counted
2. **selective directional accuracy** — only high-confidence / high-quality windows counted
3. **effective decision utility** — direction quality plus hold/abstain behavior and edge

This roadmap should aim for:

- **Stage A:** push selective BTC short-horizon direction above 60% at usable coverage
- **Stage B:** push full-coverage BTC short-horizon direction into the mid/high 50s
- **Stage C:** only then attempt >60% full-coverage with additional model-path changes if the evidence supports it

Working guardrail:

- any >60% claim must report coverage alongside accuracy
- if coverage becomes too small to be operationally meaningful, the stage does not count as success

## Working thesis

The remaining error is not concentrated in one single bug. It is likely a combination of:

- weak decision quality in the coin-flip band (`P(up)` near 0.5)
- regime compression from the current 3-state setup on crypto short horizons
- validation / calibration choices that are not horizon-native enough
- lack of a meta-decision layer that can abstain or route ambiguous windows differently

That implies the path above 60% should be **staged and evidence-driven**, not a broad rewrite.

---

## Stage 0 — sharpen the target metric before deeper modeling

### Objective

Define exactly what “above 60%” means so later PRs cannot claim success by silently changing coverage or target definition.

### Files

- `src/tools/finance/backtest/metrics.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`

### Primary function targets

- `computeRCCurve(...)`
- `selectiveDirectionalAccuracy(...)`
- `pUpDirectionalAccuracy(...)`
- `generateReport(...)`

### Planned changes

- add explicit PR3 reporting for:
  - full-coverage BTC 7d / 14d direction
  - selective BTC 7d / 14d direction at fixed confidence thresholds
  - coverage at each threshold
  - hold / abstain rate
  - direction conditioned on `P(up)` bands such as:
    - `<0.45`
    - `0.45–0.50`
    - `0.50–0.55`
    - `>0.55`
- add a reviewer-visible success table showing:
  - current baseline
  - candidate policy result
  - delta

### Why this stage exists

The current evidence already shows that the hardest band is `0.45–0.55`. We need that reported explicitly in a stable format before attempting larger changes.

### Acceptance criteria

- every future PR can show whether gains come from better handling of ambiguous windows or from global improvement
- full-coverage and selective metrics are always reported separately

### QA scenario

**Primary commands**

```bash
bun test src/tools/finance/backtest/metrics.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

**Expected result**

- exit code 0
- BTC short-horizon reports include fixed banded and selective views
- no regression of current PR1/PR2 reporting

---

## Stage 1 — selective prediction / abstention for the ambiguous band

### Objective

Cross **60% selective directional accuracy** on BTC short horizons by stopping the system from making low-value calls in the weakest `P(up)` band.

### Why this stage is first

This is the most credible route above 60% in the near term. The current backtests already indicate that much of the remaining error is concentrated in ambiguous windows. Raising full-coverage accuracy directly from ~44–47% to >60% in one jump is unlikely; selective prediction is the first realistic milestone.

### Files

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/backtest/metrics.ts`
- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`
- `src/tools/finance/markov-distribution.test.ts`

### Primary function targets

- `computePredictionConfidence(...)`
- `markovDistributionTool`
- `computeActionSignal(...)`
- `selectiveDirectionalAccuracy(...)`
- `computeRCCurve(...)`

### Planned changes

- add an explicit short-horizon abstention / HOLD policy for ambiguous crypto windows based on:
  - `P(up)` proximity to 0.5
  - prediction confidence
  - regime-specific historical reliability
  - structural-break status
  - anchor quality / absence
- add selective scoring/reporting for fixed confidence and ambiguity thresholds
- expose the minimum step metadata required to attribute abstentions to reasons rather than opaque model behavior
- calibrate a **coverage frontier** for BTC 7d/14d using walk-forward outputs rather than hand-picking thresholds

### Stop/go gate

- if no declared confidence / ambiguity threshold produces >60% selective accuracy at acceptable coverage, do not move on assuming abstention solved the problem

### Acceptance criteria

- at one or more fixed, reviewable thresholds, BTC 7d/14d selective directional accuracy exceeds **60%** with explicitly reported coverage
- reviewers can see exactly how much coverage was sacrificed to get there

### QA scenario

**Primary commands**

```bash
bun test src/tools/finance/markov-distribution.test.ts
bun test src/tools/finance/backtest/metrics.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

**Expected result**

- exit code 0
- selective BTC short-horizon reporting crosses 60% at at least one declared threshold
- coverage is reported alongside the gain

---

## Stage 2 — meta-decision layer on top of the calibrated distribution

### Objective

Improve **full-coverage** directional accuracy by learning a better mapping from existing features to `BUY/HOLD/SELL`, instead of relying only on fixed expected-return or `P(up)` rules.

### Why this is likely necessary

The remaining error pattern suggests the problem is no longer just a single threshold. We already have useful metadata that can inform a better decision layer:

- regime
- confidence
- validation metric
- structural break
- ensemble consensus
- anchor quality
- `outOfSampleR2`

### Files

- `src/tools/finance/backtest/metrics.ts`
- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`
- `src/tools/finance/markov-distribution.test.ts`

### Primary function targets

- `computeActionSignal(...)`
- `computeRegimeUpRates(...)`
- `computePredictionConfidence(...)`
- `computeFailureDecomposition(...)`

### Planned changes

- create a small rule-based or score-based **meta-decision policy** for BTC short horizons using existing metadata only
- compare these policy families:
  - current recommendation path
  - raw `P(up)` sign path
  - confidence-gated path
  - regime-aware path
  - metadata-scored path
- keep this as a decision-layer change, not a transition-model rewrite
- include explicit candidate features already present in the pipeline:
  - `regime`
  - `confidence`
  - `validationMetric`
  - `structuralBreakDetected`
  - `ensembleConsensus`
  - `anchorQuality`
  - `outOfSampleR2`

### Stop/go gate

- if the best metadata-aware policy cannot push full-coverage BTC 7d/14d direction beyond the current PR2 baseline by a meaningful margin, do not proceed to more complex model-state changes without documenting the ceiling

### Acceptance criteria

- full-coverage BTC 7d/14d directional accuracy improves beyond the current PR2 baseline
- improvements are attributable to explainable decision features rather than opaque heuristics

### QA scenario

**Primary commands**

```bash
bun test src/tools/finance/markov-distribution.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

**Expected result**

- exit code 0
- integration logs show side-by-side policy comparison on BTC 7d/14d
- at least one metadata-aware policy clearly beats the current PR2 baseline

---

## Stage 3 — short-horizon regime expansion / richer state model

### Objective

Address the possibility that the current 3-state regime model is too compressed for BTC short-horizon direction, especially when bull / sideways / high-vol windows are being merged too aggressively.

### Why this stage is later

This is a deeper model-path change. It should only happen after the decision-layer ceiling becomes clear, because it is more invasive and more likely to create side effects across non-target surfaces.

### Files

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/hmm.ts`
- `src/tools/finance/markov-distribution.test.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`
- `src/tools/finance/markov-trajectory.integration.test.ts`

### Primary function targets

- `classifyRegimeState(...)`
- `estimateTransitionMatrix(...)`
- `computeRegimeUpRates(...)`
- `baumWelch(...)` / `predict(...)` in `hmm.ts`
- crypto `hmmWeightMultiplier`

### Planned changes

- test whether a richer short-horizon crypto regime representation improves direction:
  - reintroduce or emulate high-vol sub-states for crypto short horizons
  - separate momentum-driven bull windows from noisy sideways windows
  - re-evaluate HMM contribution for short-horizon crypto only
- keep the expanded regime path target-surface limited until evidence supports generalization

### Stop/go gate

- if richer states fragment the sample and make the walk-forward slices unstable, revert to the coarser regime design and keep the gains in the decision layer instead

### Acceptance criteria

- the richer state path materially improves BTC short-horizon direction beyond the best Stage 2 policy
- gains are not offset by severe calibration or CI degradation

### QA scenario

**Primary commands**

```bash
bun test src/tools/finance/markov-distribution.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-trajectory.integration.test.ts
bun run typecheck
```

**Expected result**

- exit code 0
- BTC short-horizon directional gain survives backtest validation
- trajectory quality does not collapse

---

## Stage 4 — horizon-native validation and calibration redesign

### Objective

Reduce the mismatch between what is being optimized and what short-horizon direction actually needs.

### Why this stage matters

PR1 showed `horizon_return` slices underperform `daily_return`. That is a sign that current validation/calibration may not be aligned to short-horizon directional success. However, this should come after the smaller decision/meta-decision layers because it is broader and easier to get wrong.

### Files

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/backtest/metrics.ts`
- `src/tools/finance/markov-backtest.integration.test.ts`
- `src/tools/finance/markov-distribution.test.ts`

### Primary function targets

- `calibrateProbabilities(...)`
- `computeValidationR2OS(...)`
- `computeRegimeUpRates(...)`
- `computeActionSignal(...)`

### Planned changes

- redesign short-horizon validation / calibration so the chosen path is judged more directly on directional skill
- compare:
  - daily-return-native calibration
  - horizon-return-native calibration
  - mixed or band-specific calibration
- add explicit ablation reporting so calibration gains are not mistaken for decision-policy gains

### Stop/go gate

- if calibration changes improve one metric while worsening directional accuracy on the target slice, reject the change regardless of prettier reliability plots

### Acceptance criteria

- short-horizon BTC direction improves under a validation choice that is clearly superior in backtests
- the redesign is still explainable and stable

### QA scenario

**Primary commands**

```bash
bun test src/tools/finance/markov-distribution.test.ts
RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
bun run typecheck
```

**Expected result**

- exit code 0
- backtest logs clearly separate calibration effects from decision-policy effects

---

## Stage 5 — external signal integration only if internal ceiling is reached

### Objective

If internal decision/model-path changes plateau below 60% full-coverage, consider adding stronger external short-horizon signals in a disciplined way.

### Why this is late-stage only

The local references strongly caution against treating prediction-market prices as a single source of truth. External signals should be added only after the internal path is well understood, or else the gains will be impossible to attribute.

### Candidate signals

- persistence-qualified prediction-market shocks
- cross-market divergence / arbitrage quality flags
- skilled-trader / whale-quality filters if data is actually available and reliable
- options-vol / macro-event context for crypto-sensitive windows

### Files

- `src/tools/finance/polymarket.ts`
- `src/tools/finance/markov-distrib