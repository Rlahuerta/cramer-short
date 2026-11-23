import { describe, expect, it } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { runComparison } from './phase4-trend-penalty-comparison.js';

describe('phase4-trend-penalty-comparison', () => {
  // NOTE: Headline metrics refreshed after commit 0332b33 corrected
  // computeRegimeUpRates to sum log returns instead of simple returns
  // (the prior summation under-counted up moves at multi-day horizons).
  // They were refreshed again after the zero-anchor cryptoModelOnly break fix
  // enabled the conservative PR3F blend during structural breaks.
  // Refreshed 2026-11-22: the break-significance gate (83992ca) cut break steps from 1147 to 12, so the
  // break-context RC point below now rests on 3 steps; this case had not run in any tier until integrationIt
  // cases in ordinary test files joined test:integration.
  // Refreshed 2026-11-23: the 400-iteration return-HMM budget lets the HMM converge, so its +0.10 confidence bonus
  // lifts most steps over the 0.2 threshold (coverage 0.911 -> 0.995, accuracy 0.6505 -> 0.6484).
  integrationIt('reproduces the verified Phase 4 headline metrics on the fixture universe', async () => {
    const artifact = await runComparison();

    expect(artifact.baseline.totalSteps).toBe(1320);
    expect(artifact.baseline.breakSteps).toBe(12);
    expect(artifact.delta.changedStepCount).toBe(0);
    expect(artifact.delta.changedBreakChopCount).toBe(0);
    expect(artifact.delta.changedBreakTrendingCount).toBe(0);

    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6483771251931993, 6);
    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.9954545454545455, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6483771251931993, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.9954545454545455, 6);

    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.6666666666666666, 6);
    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.25, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.6666666666666666, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.25, 6);
  }, 480_000);
});
