import { describe, expect, it } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { runComparison } from './phase4-trend-penalty-comparison.js';

describe('phase4-trend-penalty-comparison', () => {
  // NOTE: Headline metrics refreshed after commit 0332b33 corrected
  // computeRegimeUpRates to sum log returns instead of simple returns
  // (the prior summation under-counted up moves at multi-day horizons).
  // They were refreshed again after the zero-anchor cryptoModelOnly break fix
  // enabled the conservative PR3F blend during structural breaks.
  integrationIt('reproduces the verified Phase 4 headline metrics on the fixture universe', async () => {
    const artifact = await runComparison();

    expect(artifact.baseline.totalSteps).toBe(1320);
    expect(artifact.baseline.breakSteps).toBe(1147);
    expect(artifact.delta.changedStepCount).toBe(0);
    expect(artifact.delta.changedBreakChopCount).toBe(0);
    expect(artifact.delta.changedBreakTrendingCount).toBe(0);

    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6476345840130505, 6);
    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.9287878787878788, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6476345840130505, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.9287878787878788, 6);

    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.6516666666666666, 6);
    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.5231037489102005, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.6516666666666666, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.5231037489102005, 6);
  }, 480_000);
});
