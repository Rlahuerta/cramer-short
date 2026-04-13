import { describe, expect, it } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { runComparison } from './phase4-trend-penalty-comparison.js';

describe('phase4-trend-penalty-comparison', () => {
  integrationIt('reproduces the verified Phase 4 headline metrics on the fixture universe', async () => {
    const artifact = await runComparison();

    expect(artifact.baseline.totalSteps).toBe(1320);
    expect(artifact.baseline.breakSteps).toBe(1187);
    expect(artifact.delta.changedStepCount).toBe(313);
    expect(artifact.delta.changedBreakChopCount).toBe(313);
    expect(artifact.delta.changedBreakTrendingCount).toBe(0);

    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6452879581151832, 6);
    expect(artifact.baseline.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.5787878787878787, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.accuracy).toBeCloseTo(0.6543778801843319, 6);
    expect(artifact.experiment.overallRC.find(point => point.threshold === 0.2)?.coverage).toBeCloseTo(0.6575757575757576, 6);

    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.6071428571428571, 6);
    expect(artifact.baseline.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.09435551811288964, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.accuracy).toBeCloseTo(0.657243816254417, 6);
    expect(artifact.experiment.breakContextRC.find(point => point.threshold === 0.3)?.coverage).toBeCloseTo(0.2384161752316765, 6);
  }, 480_000);
});
