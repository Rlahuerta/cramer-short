# Literature Review Eight-Suggestion Implementation Report — 2026-05-01

## Scope

This report covers the eight implementation ideas listed in
`docs/literature-review-markov-chains-price-forecasting-2026-05-01.md`
under "Suggested Future Directions".

Execution followed the requested TDD/backtest-gated rule:

1. write or identify a failing/acceptance test,
2. implement only the smallest viable candidate,
3. run the relevant forecast/backtest gate,
4. commit only phases that should remain in the code,
5. drop or leave blocked phases that do not improve the forecast tools.

## arXiv/reference verification notes

The arXiv MCP abstracts support most high-level directions, but one citation in
the review does not match the described method exactly:

| Reference | Verification result |
|---|---|
| Waghmare & Ziegel, `2504.01781` | Proper scoring rules review supports adding richer probabilistic forecast metrics. |
| Bolin & Wallin, `1912.05642` | Scaled CRPS motivation is relevant for heterogeneous uncertainty. |
| Du, `2012.12499` | Confirms that CRPS should not be the only probabilistic score; log/Brier-style diagnostics remain useful. |
| Angelopoulos et al., `2307.16895` | Supports conformal PID and anti-windup/integral-decay discussion. |
| Blake et al., `2510.03236` | Supports soft regime weighting instead of hard assignment. |
| Ammann et al., `2601.03760` | Supports covariate-dependent non-homogeneous Markov-switching transitions. |
| Bernardi et al., `2202.12644` | arXiv metadata resolves to a large Bayesian VAR paper, not the Markov-switching hierarchical-shrinkage title cited in the review. Transition shrinkage was therefore treated as shrinkage-inspired rather than a direct implementation of that paper. |
| Cini et al., `2502.09443` | Supports relational conformal prediction for correlated time series, but requires a multi-series calibration substrate. |
| Asanjarani et al., `2005.14462` and Francis-Staite & White, `2206.10865` | Support semi-Markov sojourn-time modeling; repo-local prior attempts did not improve forecast quality. |

## Phase decisions

| # | Idea | Decision | Evidence |
|---|---|---|---|
| 1 | CRPS / scaled CRPS in backtest metrics | **Kept** | Added evaluation-only diagnostics to `BacktestReport`; no live forecast behavior changed. |
| 8 | Murphy-Winkler interval score decomposition | **Kept** | Added interval score decomposition alongside CRPS so future keep/drop gates can evaluate interval quality, not only Brier/direction. |
| 2 | Conformal PID default `integralDecay = 0.99` | **Dropped** | The candidate failed the existing Gaussian quantile-convergence gate: final radius fell below the accepted lower bound (`1.376 < 1.4`). The code was reverted. |
| 3 | Soft regime probability weighting | **No new code; already satisfied** | Current code already has `enableSoftRegimeWeighting`, soft-regime metadata, disabled parity tests, CI/confidence effects, and raw-distribution movement tests. |
| 5 | Post-break transition shrinkage | **No new code; already tried/dropped** | Existing `BreakFallbackCandidate` implements shrinkage-like structural-break matrix blending and has unit/backtest tests. Prior gate dropped this candidate family due to ambiguous/negative lift. |
| 4 | Covariate-dependent transitions | **Blocked/dropped for now** | A Kalshi volatility covariate adapter exists and is tested, but there is still no offline historical transition substrate to honestly prove transition-forecast lift. |
| 6 | Cross-asset conformal residual sharing | **Dropped for now** | Current cross-asset code is trajectory drift bias, not conformal pooling. Prior relational conformal prototypes widened intervals without improving coverage, Brier, or direction. |
| 7 | Semi-Markov sojourn layer | **Dropped for now** | Previous causal sojourn attempt produced no measurable forecast movement; no materially new duration substrate was found. |

## Kept implementation

### Distributional backtest diagnostics

Commit: `9784b1d Add distributional backtest diagnostics`

Changed files:

- `src/tools/finance/backtest/metrics.ts`
- `src/tools/finance/backtest/metrics.test.ts`

New exported metrics:

- `crps(steps)`
- `scaledCrps(steps)`
- `murphyWinklerDecomposition(steps, alpha?)`
- `murphyWinklerScore(steps, alpha?)`

New `BacktestReport` fields:

- `crps`
- `scaledCrps`
- `murphyWinklerScore`
- `murphyWinklerDecomposition`

Implementation note: `BacktestStep` does not currently carry a full predictive
CDF or mixture distribution. The CRPS implementation therefore uses the
existing 90% confidence interval as a central-normal approximation:

- forecast center is derived from current price and `predictedReturn`,
- scale is inferred from `ciUpper - ciLower`,
- scaled CRPS divides each step's CRPS by that local interval-implied scale.

This avoids pretending the backtest has richer distribution data than it
currently stores. If future phases persist the full Markov mixture in
`BacktestStep`, CRPS can be upgraded from interval-normal approximation to exact
mixture scoring.

## Rejected candidate details

### Conformal default decay

The proposed one-line change from default `integralDecay = 1.0` to `0.99` was
tested directly. The new test first proved default behavior would need to match
explicit `0.99`; after implementation, the existing quantile-convergence test
failed:

```text
Expected currentRadius > 1.4
Received: 1.376239614021879
```

Because this would weaken an existing calibration invariant and there was no
compensating BTC gate improvement, the candidate was reverted and not committed.

## Existing implementation coverage for no-op phases

### Soft regime weighting

Current code already implements the core recommendation:

- `src/tools/finance/markov-distribution.ts` builds soft current and forecast
  regime mixtures from HMM posterior probabilities.
- `enableSoftRegimeWeighting` blends distribution construction toward those
  mixtures, widens CI, and reduces confidence under high posterior entropy.
- `src/tools/finance/backtest/walk-forward.ts` forwards the flag and records
  provenance fields.
- Tests verify disabled parity, metadata, confidence/CI effects, and raw
  distribution movement.

### Transition shrinkage / structural-break fallback

Current code already has a shrinkage-like structural-break fallback family:

- `BreakFallbackCandidate`
- `buildConservativeFallbackMatrix`
- `buildProfileFallbackMatrix`
- `blendMatrices`
- `applyBreakFallbackCandidate`

This is not default-on and prior backtest evidence did not justify promotion.

### Covariates, cross-asset, and semi-Markov

The repo has useful adjacent pieces, but not enough evidence to promote new
forecast behavior:

- Kalshi covariates are parsed and tested, but not historically wired into a
  transition backtest substrate.
- Cross-asset trajectory bias exists, but conformal residual sharing previously
  failed the acceptance gate.
- Semi-Markov sojourn modeling was already tried as a causal post-processing
  layer and produced no measurable lift.

## Final recommendation

Keep the new distributional diagnostics. They improve the forecast tooling by
making future phases harder to overfit to Brier or directional accuracy alone.

Do not change conformal defaults, transition behavior, covariate routing,
cross-asset conformal pooling, or semi-Markov behavior from this review without
a materially new backtest substrate. The literature ideas remain valuable, but
the current repo evidence supports only the evaluation-metrics phase as a code
change today.
