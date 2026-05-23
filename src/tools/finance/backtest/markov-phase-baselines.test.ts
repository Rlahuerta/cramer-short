import { describe, expect, it } from 'bun:test';

import { MARKOV_PHASE0_BASELINES } from './markov-phase-baselines.js';

describe('MARKOV_PHASE0_BASELINES', () => {
  it('freezes the expected BTC and GOLD horizon slices for abstain-reduction work', () => {
    expect(MARKOV_PHASE0_BASELINES).toEqual({
      btc: {
        h1: {
          directionalAccuracy: 0.5573770491803278,
          brierScore: 0.2580125170653137,
          ciCoverage: 0.9754098360655737,
          abstainCount: 34,
          rerunRate: 0,
        },
        h2: {
          directionalAccuracy: 0.5573770491803278,
          brierScore: 0.26155232084959035,
          ciCoverage: 0.9836065573770492,
          abstainCount: 39,
          rerunRate: 0,
        },
        h3: {
          directionalAccuracy: 0.5245901639344263,
          brierScore: 0.26640877159390347,
          ciCoverage: 0.9918032786885246,
          abstainCount: 35,
          rerunRate: 0,
        },
        h14: {
          directionalAccuracy: 0.4666666666666667,
          brierScore: 0.2747328793942764,
          ciCoverage: 0.9833333333333333,
          abstainCount: 27,
          rerunRate: 0,
        },
      },
      gold: {
        h1: {
          directionalAccuracy: 0.6447368421052632,
          brierScore: 0.24032334188017065,
          ciCoverage: 0.9473684210526315,
          abstainCount: 5,
          structuralBreakCount: 61,
        },
        h2: {
          directionalAccuracy: 0.5657894736842105,
          brierScore: 0.2422450455513245,
          ciCoverage: 0.9736842105263158,
          abstainCount: 0,
          structuralBreakCount: 61,
        },
        h3: {
          directionalAccuracy: 0.631578947368421,
          brierScore: 0.22920645210607415,
          ciCoverage: 0.9868421052631579,
          abstainCount: 2,
          structuralBreakCount: 61,
        },
        h7: {
          directionalAccuracy: 0.6933333333333334,
          brierScore: 0.22971821073275564,
          ciCoverage: 0.9866666666666667,
          abstainCount: 2,
          structuralBreakCount: 52,
        },
        h14: {
          directionalAccuracy: 0.7702702702702703,
          brierScore: 0.20693683117021866,
          ciCoverage: 1,
          abstainCount: 0,
          structuralBreakCount: 51,
        },
      },
    });
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
