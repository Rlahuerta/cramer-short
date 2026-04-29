/**
 * Tests for R5 Idea #3 — naive baseline metrics + CI guard.
 */
import { describe, expect, it } from 'bun:test';
import {
  computeCoinFlipBaseline,
  computeLastPeriodBaseline,
  computeNaiveBaselines,
  naiveBaselineGuard,
} from './baselines.js';
import type { BacktestStep } from './metrics.js';

function step(opts: Partial<BacktestStep> & {
  actualReturn: number;
  predictedProb: number;
  recommendation: BacktestStep['recommendation'];
}): BacktestStep {
  const realizedPrice = 100 * (1 + opts.actualReturn);
  return {
    t: opts.t ?? 0,
    predictedProb: opts.predictedProb,
    actualBinary: opts.actualReturn > 0 ? 1 : 0,
    predictedReturn: opts.predictedReturn ?? 0,
    actualReturn: opts.actualReturn,
    ciLower: opts.ciLower ?? 90,
    ciUpper: opts.ciUpper ?? 110,
    realizedPrice: opts.realizedPrice ?? realizedPrice,
    recommendation: opts.recommendation,
    gofPasses: opts.gofPasses ?? null,
    confidence: opts.confidence ?? 0.5,
  };
}

describe('R5 Idea #3 — naive baselines', () => {
  it('coin-flip returns 0.25 Brier on a balanced random series', () => {
    const steps: BacktestStep[] = [
      step({ actualReturn:  0.05, predictedProb: 0.7, recommendation: 'BUY' }),
      step({ actualReturn: -0.05, predictedProb: 0.3, recommendation: 'SELL' }),
      step({ actualReturn:  0.04, predictedProb: 0.6, recommendation: 'BUY' }),
      step({ actualReturn: -0.06, predictedProb: 0.4, recommendation: 'SELL' }),
    ];
    const cf = computeCoinFlipBaseline(steps);
    expect(cf.n).toBe(4);
    expect(cf.brierScore).toBeCloseTo(0.25, 4);
    // HOLD is correct only when |actualReturn| < 0.03 — none of these qualify
    expect(cf.directionalAccuracy).toBe(0);
  });

  it('last-period baseline scores ~50% on alternating returns', () => {
    // Alternating: +5%, -5%, +5%, -5%, ... after step 0 it always predicts
    // the *opposite* of what happens, so dirAcc = 0 (excluding step 0).
    const seq = [+0.05, -0.05, +0.05, -0.05, +0.05, -0.05, +0.05, -0.05];
    const steps = seq.map(r => step({ actualReturn: r, predictedProb: 0.5, recommendation: 'HOLD' }));
    const lp = computeLastPeriodBaseline(steps);
    // Step 0 is always HOLD on a >3% move ⇒ wrong; subsequent steps predict
    // wrong direction every time.  Expect dirAcc = 0.
    expect(lp.directionalAccuracy).toBeLessThan(0.2);
    // Brier is bounded above by 0.65² = 0.4225 (when always wrong with p=0.65)
    expect(lp.brierScore).toBeGreaterThan(0.3);
  });

  it('last-period baseline scores 100% on a persistent uptrend', () => {
    // All up moves > 3%: after step 0, lastPeriod always predicts BUY → correct.
    const seq = [+0.05, +0.05, +0.05, +0.05, +0.05];
    const steps = seq.map(r => step({ actualReturn: r, predictedProb: 0.5, recommendation: 'BUY' }));
    const lp = computeLastPeriodBaseline(steps);
    // Step 0 is HOLD on +5% move ⇒ wrong. Steps 1..4 all correctly BUY.
    expect(lp.directionalAccuracy).toBeCloseTo(0.8, 2);
  });

  it('computeNaiveBaselines returns both blocks', () => {
    const steps = [step({ actualReturn: 0.05, predictedProb: 0.5, recommendation: 'BUY' })];
    const r = computeNaiveBaselines(steps);
    expect(r.coinFlip.n).toBe(1);
    expect(r.lastPeriod.n).toBe(1);
  });
});

describe('R5 Idea #3 — naive baseline CI guard', () => {
  it('passes when arm beats baseline', () => {
    const steps: BacktestStep[] = [
      step({ actualReturn:  0.05, predictedProb: 0.8, recommendation: 'BUY'  }),  // correct
      step({ actualReturn: -0.05, predictedProb: 0.2, recommendation: 'SELL' }),  // correct
      step({ actualReturn:  0.05, predictedProb: 0.7, recommendation: 'BUY'  }),  // correct
      step({ actualReturn: -0.05, predictedProb: 0.3, recommendation: 'SELL' }),  // correct
    ];
    const g = naiveBaselineGuard(steps);
    expect(g.armDirAcc).toBe(1);
    expect(g.passes).toBe(true);
    expect(g.gap).toBeGreaterThan(0);
  });

  it('fails when arm is worse than baseline by more than slack', () => {
    // Arm always predicts WRONG; lastPeriod baseline (after step 0)
    // always predicts wrong too — but the arm's HOLD is also wrong on
    // every step ⇒ dirAcc=0 vs baseline=0 ⇒ gap=0 ⇒ passes (within slack).
    // To force failure we need a scenario where lastPeriod is right.
    const seq = [+0.05, +0.05, +0.05];  // persistent uptrend
    const steps = seq.map(r => step({ actualReturn: r, predictedProb: 0.5, recommendation: 'SELL' }));
    const g = naiveBaselineGuard(steps, 0.01);
    // Arm dirAcc = 0; lastPeriod dirAcc = 2/3 ≈ 0.67; gap = -0.67 < -0.01 ⇒ fails
    expect(g.armDirAcc).toBe(0);
    expect(g.baselineDirAcc).toBeGreaterThan(0.5);
    expect(g.passes).toBe(false);
  });
});
