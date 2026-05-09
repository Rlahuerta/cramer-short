import { describe, expect, it } from 'bun:test';

import { MARKOV_PHASE0_BASELINES } from './markov-phase-baselines.js';

describe('MARKOV_PHASE0_BASELINES', () => {
  it('freezes the expected BTC and GOLD horizon slices for abstain-reduction work', () => {
    expect(Object.keys(MARKOV_PHASE0_BASELINES.btc)).toEqual(['h1', 'h2', 'h3', 'h14']);
    expect(Object.keys(MARKOV_PHASE0_BASELINES.gold)).toEqual(['h1', 'h2', 'h3', 'h7', 'h14']);
  });

  it('captures non-empty baseline metrics for every tracked horizon', () => {
    for (const metrics of Object.values(MARKOV_PHASE0_BASELINES.btc)) {
      expect(metrics.directionalAccuracy).toBeGreaterThan(0);
      expect(metrics.brierScore).toBeGreaterThan(0);
      expect(metrics.ciCoverage).toBeGreaterThan(0);
      expect(metrics.abstainCount).toBeGreaterThanOrEqual(0);
      expect(metrics.rerunRate).toBeGreaterThanOrEqual(0);
    }

    for (const metrics of Object.values(MARKOV_PHASE0_BASELINES.gold)) {
      expect(metrics.directionalAccuracy).toBeGreaterThan(0);
      expect(metrics.brierScore).toBeGreaterThan(0);
      expect(metrics.ciCoverage).toBeGreaterThan(0);
      expect(metrics.abstainCount).toBeGreaterThanOrEqual(0);
      expect(metrics.structuralBreakCount).toBeGreaterThan(0);
    }
  });
});
