/**
 * R5 Idea #3 — Naive baseline guards for forecast quality CI.
 *
 * Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #3),
 * arXiv:2502.09079 (Puoti et al. 2025) — "Crypto unpredictability".
 *
 * Computes simple model-free baselines that any non-trivial forecast head
 * must beat:
 *   - Coin-flip:    always P(up)=0.5, recommends HOLD
 *   - Last-period:  P(up)=1 if previous step was up, recommends BUY/SELL
 *
 * Returned metrics use the same {@link BacktestStep} shape so they can be
 * compared head-to-head with `directionalAccuracy`/`brierScore` from
 * `metrics.ts`.
 *
 * The CI gate (used in `markov-backtest.integration.test.ts`) is:
 *   "for every horizon h ≥ 7d, the deployed arm's `directionalAccuracy`
 *   must be ≥ baseline.lastPeriod.directionalAccuracy − 0.02".  We allow
 *   a 2pp slack to absorb sampling noise on small fixtures.
 */
import { brierScore, directionalAccuracy, type BacktestStep } from './metrics.js';

export interface BaselineMetricBlock {
  n: number;
  brierScore: number;
  directionalAccuracy: number;
}

export interface NaiveBaselineReport {
  /** "Always P=0.5, recommend HOLD" — the strict random null. */
  coinFlip: BaselineMetricBlock;
  /** "Predict last realized direction continues" — momentum-1 baseline. */
  lastPeriod: BaselineMetricBlock;
}

/**
 * Build a synthetic step that mirrors `actual` but with the synthetic
 * baseline prediction, so we can reuse the existing metric implementations.
 */
function synth(
  step: BacktestStep,
  predictedProb: number,
  recommendation: BacktestStep['recommendation'],
): BacktestStep {
  return {
    ...step,
    predictedProb,
    rawPredictedProb: predictedProb,
    recommendation,
    confidence: 0.5,
    gofPasses: null,
  };
}

export function computeCoinFlipBaseline(steps: BacktestStep[]): BaselineMetricBlock {
  if (steps.length === 0) return { n: 0, brierScore: 1, directionalAccuracy: 0 };
  const synthetic = steps.map(s => synth(s, 0.5, 'HOLD'));
  return {
    n: steps.length,
    brierScore: brierScore(synthetic),
    directionalAccuracy: directionalAccuracy(synthetic),
  };
}

/**
 * "Last period" baseline: predicts the direction of the *previous*
 * realized return continues. For the first step (no prior), it falls
 * back to the coin-flip prior (P=0.5, HOLD).
 *
 * For probability assignment we use a confident-but-not-degenerate value
 * (0.65 / 0.35) — picking 1.0 / 0.0 would hand the baseline an unfair
 * Brier score (0 when right, 1 when wrong, dominated by the 50%
 * directional accuracy seen on random walks).
 */
export function computeLastPeriodBaseline(steps: BacktestStep[]): BaselineMetricBlock {
  if (steps.length === 0) return { n: 0, brierScore: 1, directionalAccuracy: 0 };
  let prevReturn = 0;
  const synthetic: BacktestStep[] = steps.map((step, i) => {
    if (i === 0) {
      prevReturn = step.actualReturn;
      return synth(step, 0.5, 'HOLD');
    }
    const predUp = prevReturn > 0;
    const predictedProb = predUp ? 0.65 : 0.35;
    const recommendation = predUp ? 'BUY' : 'SELL';
    prevReturn = step.actualReturn;
    return synth(step, predictedProb, recommendation);
  });
  return {
    n: steps.length,
    brierScore: brierScore(synthetic),
    directionalAccuracy: directionalAccuracy(synthetic),
  };
}

export function computeNaiveBaselines(steps: BacktestStep[]): NaiveBaselineReport {
  return {
    coinFlip: computeCoinFlipBaseline(steps),
    lastPeriod: computeLastPeriodBaseline(steps),
  };
}

/**
 * Strict gate: deployed arm must have directional accuracy at least as
 * good as the better of the two naive baselines, minus a slack.
 *
 * @param armSteps      — steps from the candidate arm
 * @param slack         — pp slack absorbed (default 0.02)
 * @returns             — {passes, armDirAcc, baselineDirAcc, gap}
 */
export function naiveBaselineGuard(
  armSteps: BacktestStep[],
  slack = 0.02,
): { passes: boolean; armDirAcc: number; baselineDirAcc: number; gap: number } {
  const armDirAcc = directionalAccuracy(armSteps);
  const baselines = computeNaiveBaselines(armSteps);
  const baselineDirAcc = Math.max(
    baselines.coinFlip.directionalAccuracy,
    baselines.lastPeriod.directionalAccuracy,
  );
  const gap = armDirAcc - baselineDirAcc;
  return { passes: gap >= -slack, armDirAcc, baselineDirAcc, gap };
}
