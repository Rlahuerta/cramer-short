export interface MarkovPhaseBaselineMetric {
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  abstainCount: number;
}

export interface BtcMarkovPhaseBaselineMetric extends MarkovPhaseBaselineMetric {
  rerunRate: number;
}

export interface GoldMarkovPhaseBaselineMetric extends MarkovPhaseBaselineMetric {
  structuralBreakCount: number;
}

/**
 * Phase 0 baseline freeze captured on 2026-05-09 from the current shipped
 * BTC/GOLD walk-forward suites before abstain-reduction changes.
 *
 * Sources:
 * - src/tools/finance/backtest/walk-forward-btc-ultra-short-horizon.test.ts
 * - src/tools/finance/backtest/btc-live-short-horizon-policy.integration.test.ts
 * - src/tools/finance/backtest/walk-forward-gold-short-horizon.test.ts
 */
export const MARKOV_PHASE0_BASELINES = {
  btc: {
    h1: {
      directionalAccuracy: 0.59375,
      brierScore: 0.2550127431393302,
      ciCoverage: 0.9791666666666666,
      abstainCount: 20,
      rerunRate: 0.75,
    },
    h2: {
      directionalAccuracy: 0.4895833333333333,
      brierScore: 0.2536990360321187,
      ciCoverage: 0.9895833333333334,
      abstainCount: 18,
      rerunRate: 0,
    },
    h3: {
      directionalAccuracy: 0.5416666666666666,
      brierScore: 0.26222299097692103,
      ciCoverage: 0.96875,
      abstainCount: 18,
      rerunRate: 0.3020833333333333,
    },
    h14: {
      directionalAccuracy: 0.5161290322580645,
      brierScore: 0.2695575622668006,
      ciCoverage: 0.978494623655914,
      abstainCount: 12,
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
} as const satisfies {
  btc: Record<'h1' | 'h2' | 'h3' | 'h14', BtcMarkovPhaseBaselineMetric>;
  gold: Record<'h1' | 'h2' | 'h3' | 'h7' | 'h14', GoldMarkovPhaseBaselineMetric>;
};
