import { describe, expect, it } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { runComparison } from './phase4-trend-penalty-comparison.js';

describe('phase4-trend-penalty-comparison', () => {
  // NOTE: Headline metrics refreshed after commit 0332b33 corrected
  // computeRegimeUpRates to sum log returns instead of simple returns
  // (the prior summation under-counted up moves at multi-day horizons).
  // The numbers below reflect the mathematically correct baseline and
  // are therefore the new ground truth.
  integrationIt('reproduces the verified Phase 4 headline metrics on the fixture universe', async () => {
    const artifact = await runComparison();

    expect(artifact.baseline.totalSteps).toBe(1320);
    expect(artifact.baseline.breakSteps).toBe(1187);
    expect(artifact.delta.changedStepCount).toBe(313);
    expect(artifact.delta.changedBreakChopCount).toBe(313);
    expect(artifact.delta.changedBreakTrendingCount).toBe(0);

    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6392459297343616, 6);
    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.884090909090909, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6435070306038048, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.9159090909090909, 6);

    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.6322115384615384, 6);
    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.35046335299073295, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.6598639455782312, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.4953664700926706, 6);
  }, 480_000);
});
