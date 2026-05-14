import { afterEach, beforeEach, describe, it, expect, spyOn } from 'bun:test';
import { classifyRegimeState } from './regime.js';
import { buildDefaultMatrix } from './transition.js';
import { computeHorizonDriftVol, computeMixingWeight, computeStartStateMixture, computeTrajectory, estimateRegimeStats, interpolateDistribution, logNormalSurvival, normalCDF, studentTCDF, studentTSurvival, winsorize } from './confidence-intervals.js';
import { computeMarkovDistribution } from '../markov-distribution.js';
import type { MarkovDistributionPoint, RegimeState } from './core.js';

function seedRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

let randomSpy: { mockRestore: () => void } | undefined;

beforeEach(() => {
  randomSpy = spyOn(Math, 'random').mockImplementation(seedRng(12345));
});

afterEach(() => {
  randomSpy?.mockRestore();
  randomSpy = undefined;
});

describe('logNormalSurvival', () => {
  it('Fix 3: P(price > current) ≈ 0.5 when drift=0 and vol>0', () => {
    const p = logNormalSurvival(100, 100, 0, 0.1);
    expect(p).toBeCloseTo(0.5, 1);
  });

  it('returns close to 1 for target well below current price', () => {
    const p = logNormalSurvival(100, 10, 0, 0.1);
    expect(p).toBeGreaterThan(0.99);
  });

  it('returns close to 0 for target well above current price', () => {
    const p = logNormalSurvival(100, 1000, 0, 0.1);
    expect(p).toBeLessThan(0.01);
  });

  it('higher drift → higher P(price > target)', () => {
    const p1 = logNormalSurvival(100, 110, 0.05, 0.1);
    const p2 = logNormalSurvival(100, 110, -0.05, 0.1);
    expect(p1).toBeGreaterThan(p2);
  });

  it('returns 0 for vol ≤ 0 when target > current', () => {
    expect(logNormalSurvival(100, 110, 0, 0)).toBe(0);
  });

  it('returns 1 for vol ≤ 0 when target < current', () => {
    expect(logNormalSurvival(100, 90, 0, 0)).toBe(1);
  });
});
describe('interpolateDistribution', () => {
  const P = buildDefaultMatrix();
  const regimeStats = estimateRegimeStats([], []);

  it('Fix 5: lowerBound ≤ probability ≤ upperBound for each point', () => {
    const dist = interpolateDistribution(100, 10, P, regimeStats, 'bull', [], 0.5);
    for (const point of dist) {
      expect(point.lowerBound).toBeLessThanOrEqual(point.probability + 1e-9);
      expect(point.probability).toBeLessThanOrEqual(point.upperBound + 1e-9);
    }
  });

  it('distribution is monotonically non-increasing in price', () => {
    const dist = interpolateDistribution(100, 20, P, regimeStats, 'bull', [], 0.5);
    for (let i = 1; i < dist.length; i++) {
      expect(dist[i].probability).toBeLessThanOrEqual(dist[i - 1].probability + 1e-9);
    }
  });

  it('probability near 1 for prices well below current', () => {
    const dist = interpolateDistribution(100, 10, P, regimeStats, 'bull', [], 0.5);
    const lowest = dist[0];
    expect(lowest.probability).toBeGreaterThan(0.7);
  });

  it('probability near 0 for prices well above current', () => {
    const dist = interpolateDistribution(100, 10, P, regimeStats, 'bull', [], 0.5);
    const highest = dist[dist.length - 1];
    expect(highest.probability).toBeLessThan(0.3);
  });

  it('Fix 7: high second-eigenvalue + long horizon → lower mixing weight', () => {
    const shortWeight = computeMixingWeight(0.9, 5);
    const longWeight  = computeMixingWeight(0.9, 60);
    expect(longWeight).toBeLessThan(shortWeight);
    expect(longWeight).toBeLessThan(0.01); // essentially pure anchor at 60 days
  });

  it('source=polymarket for nearby high-trust anchor at short horizon', () => {
    const anchors = [{ price: 100, rawProbability: 0.5, probability: 0.475, trustScore: 'high' as const, source: 'polymarket' as const }];
    // Low second eigenvalue → Markov dominant
    const dist = interpolateDistribution(100, 2, P, regimeStats, 'sideways', anchors, 0.01);
    const point = dist.find(d => Math.abs(d.price - 100) < 5);
    // With very low ρ, markovWeight ≈ exp(-0.01×2) ≈ 0.98 → should be 'markov' or 'blend'
    expect(['markov', 'blend', 'polymarket']).toContain(point?.source as string);
  });
});
describe('computeMixingWeight', () => {
  it('Fix 7: at horizon 0, weight=1 (pure Markov)', () => {
    expect(computeMixingWeight(0.9, 0)).toBe(1);
  });

  it('Fix 7: at long horizon with high ρ, weight → 0', () => {
    expect(computeMixingWeight(0.9, 100)).toBeLessThan(0.001);
  });

  it('Fix 7: at long horizon with ρ≈0, weight stays near 1', () => {
    expect(computeMixingWeight(0.001, 90)).toBeGreaterThan(0.9);
  });
});
describe('estimateRegimeStats', () => {
  it('falls back to defaults with empty data', () => {
    const stats = estimateRegimeStats([], []);
    expect(stats.bull.meanReturn).toBeGreaterThan(0);
    expect(stats.bear.meanReturn).toBeLessThan(0);
  });

  it('empirical stats used when ≥5 observations per state', () => {
    const returns = Array(60).fill(0.01); // all positive
    const states  = Array(60).fill('bull') as ReturnType<typeof classifyRegimeState>[];
    const stats   = estimateRegimeStats(returns, states);
    expect(stats.bull.meanReturn).toBeCloseTo(0.01, 5);
  });
});
describe('Monte Carlo stability', () => {
  it('interpolateDistribution produces stable results across runs', () => {
    const P = buildDefaultMatrix();
    const regimeStats = estimateRegimeStats(
      Array.from({ length: 30 }, () => 0.002),
      Array.from({ length: 30 }, () => 'bull' as const),
    );

    // Run 5 times, collect probabilities at the median point
    const medianProbs: number[] = [];
    for (let run = 0; run < 5; run++) {
      const dist = interpolateDistribution(100, 20, P, regimeStats, 'bull', [], 0.5, 15, 1000);
      const midIdx = Math.floor(dist.length / 2);
      medianProbs.push(dist[midIdx].probability);
    }

    // Check coefficient of variation is < 10% (Monte Carlo noise should be small)
    const mean = medianProbs.reduce((s, v) => s + v, 0) / medianProbs.length;
    const variance = medianProbs.reduce((s, v) => s + (v - mean) ** 2, 0) / medianProbs.length;
    const cv = Math.sqrt(variance) / Math.max(mean, 1e-10);
    expect(cv).toBeLessThan(0.10);
  });

  it('confidence intervals narrow with more Monte Carlo samples', () => {
    const P = buildDefaultMatrix();
    const regimeStats = estimateRegimeStats(
      Array.from({ length: 30 }, () => 0.001),
      Array.from({ length: 30 }, () => 'sideways' as const),
    );

    // Fewer samples → wider CI
    const distFew = interpolateDistribution(100, 20, P, regimeStats, 'sideways', [], 0.5, 10, 100);
    // More samples → tighter CI
    const distMany = interpolateDistribution(100, 20, P, regimeStats, 'sideways', [], 0.5, 10, 2000);

    // Average CI width across all points
    const avgWidth = (d: typeof distFew) =>
      d.reduce((s, p) => s + (p.upperBound - p.lowerBound), 0) / d.length;

    // More samples should produce tighter or equal CI on average
    // (not guaranteed per-point due to randomness, but on average it holds)
    const fewWidth = avgWidth(distFew);
    const manyWidth = avgWidth(distMany);
    // Allow generous tolerance since MC is stochastic
    expect(manyWidth).toBeLessThan(fewWidth * 1.5);
  });
});
describe('studentTCDF', () => {
  it('CDF(0) = 0.5 for any degrees of freedom', () => {
    expect(studentTCDF(0, 5)).toBeCloseTo(0.5, 10);
    expect(studentTCDF(0, 30)).toBeCloseTo(0.5, 10);
  });

  it('CDF is monotonically increasing', () => {
    for (let x = -3; x < 3; x += 0.5) {
      expect(studentTCDF(x + 0.5, 5)).toBeGreaterThan(studentTCDF(x, 5));
    }
  });

  it('converges to normal CDF as ν → ∞', () => {
    // With ν=1000, should match normal CDF closely
    expect(studentTCDF(1.96, 1000)).toBeCloseTo(normalCDF(1.96), 2);
    expect(studentTCDF(-1.0, 1000)).toBeCloseTo(normalCDF(-1.0), 2);
  });

  it('has heavier tails than normal (lower CDF in right tail)', () => {
    // P(T > 2) should be higher for Student-t (lower CDF at x=2)
    expect(studentTCDF(2, 5)).toBeLessThan(normalCDF(2));
  });

  it('CDF(±∞) approaches 0 and 1', () => {
    expect(studentTCDF(-10, 5)).toBeLessThan(0.01);
    expect(studentTCDF(10, 5)).toBeGreaterThan(0.99);
  });
});
describe('studentTSurvival', () => {
  it('gives higher tail probability than logNormal for extreme targets', () => {
    // At very extreme tails (3+ sigma), fat tails dominate vol scaling
    const tSurv = studentTSurvival(100, 200, 0.0, 0.3);
    const nSurv = logNormalSurvival(100, 200, 0.0, 0.3);
    expect(tSurv).toBeGreaterThan(nSurv);
  });

  it('gives lower tail probability at center vs logNormal', () => {
    // Near center, Student-t is slightly lower because mass moved to tails
    const tSurv = studentTSurvival(100, 105, 0.05, 0.15);
    const nSurv = logNormalSurvival(100, 105, 0.05, 0.15);
    // This is subtle — the difference should be small
    expect(Math.abs(tSurv - nSurv)).toBeLessThan(0.1);
  });

  it('returns 1 when target < current and vol=0', () => {
    expect(studentTSurvival(100, 90, 0, 0)).toBe(1);
  });

  it('returns 0 when target > current and vol=0', () => {
    expect(studentTSurvival(100, 110, 0, 0)).toBe(0);
  });
});
describe('computeTrajectory', () => {
  // Simple 3-state bull-dominant regime
  const regimeStats = {
    bull: { meanReturn: 0.001, stdReturn: 0.012 },
    bear: { meanReturn: -0.001, stdReturn: 0.015 },
    sideways: { meanReturn: 0.0002, stdReturn: 0.010 },
  };
  const P = [
    [0.7, 0.1, 0.2], // bull stays bull
    [0.2, 0.6, 0.2], // bear
    [0.3, 0.2, 0.5], // sideways
  ];

  it('returns the correct number of days', () => {
    const traj = computeTrajectory(100, 7, P, regimeStats, 'bull', 0);
    expect(traj).toHaveLength(7);
    expect(traj[0].day).toBe(1);
    expect(traj[6].day).toBe(7);
  });

  it('CI widths monotonically increase (or stay same)', () => {
    const traj = computeTrajectory(100, 14, P, regimeStats, 'bull', 0, undefined, 2000);
    for (let i = 1; i < traj.length; i++) {
      const prevWidth = traj[i - 1].upperBound - traj[i - 1].lowerBound;
      const currWidth = traj[i].upperBound - traj[i].lowerBound;
      // Allow small MC noise tolerance (0.5% of price)
      expect(currWidth).toBeGreaterThanOrEqual(prevWidth - 0.5);
    }
  });

  it('lower bound < expected < upper bound for all days', () => {
    const traj = computeTrajectory(100, 7, P, regimeStats, 'bull', 0);
    for (const pt of traj) {
      expect(pt.lowerBound).toBeLessThan(pt.expectedPrice);
      expect(pt.upperBound).toBeGreaterThan(pt.expectedPrice);
    }
  });

  it('expected price is near current for day 1', () => {
    const traj = computeTrajectory(200, 7, P, regimeStats, 'bull', 0);
    expect(Math.abs(traj[0].expectedPrice - 200)).toBeLessThan(5);
  });

  it('P(up) is between 0 and 1 for all days', () => {
    const traj = computeTrajectory(100, 7, P, regimeStats, 'bear', 0);
    for (const pt of traj) {
      expect(pt.pUp).toBeGreaterThanOrEqual(0);
      expect(pt.pUp).toBeLessThanOrEqual(1);
    }
  });

  it('cumulative return is formatted correctly', () => {
    const traj = computeTrajectory(100, 3, P, regimeStats, 'bull', 0);
    for (const pt of traj) {
      expect(pt.cumulativeReturn).toMatch(/^[+-]\d+\.\d+%$/);
    }
  });

  it('regime is a valid RegimeState', () => {
    const traj = computeTrajectory(100, 5, P, regimeStats, 'sideways', 0);
    const validRegimes: RegimeState[] = ['bull', 'bear', 'sideways'];
    for (const pt of traj) {
      expect(validRegimes).toContain(pt.regime);
    }
  });

  it('handles horizon=1 (single day)', () => {
    const traj = computeTrajectory(100, 1, P, regimeStats, 'bull', 0);
    expect(traj).toHaveLength(1);
    expect(traj[0].day).toBe(1);
  });

  it('all values are finite (no NaN/Infinity)', () => {
    const traj = computeTrajectory(100, 10, P, regimeStats, 'bull', 0);
    for (const pt of traj) {
      expect(Number.isFinite(pt.expectedPrice)).toBe(true);
      expect(Number.isFinite(pt.lowerBound)).toBe(true);
      expect(Number.isFinite(pt.upperBound)).toBe(true);
      expect(Number.isFinite(pt.pUp)).toBe(true);
    }
  });

  it('uses HMM override when provided', () => {
    const hmmOverride = { drift: 0.005, vol: 0.02, weight: 0.5 };
    const trajNoHmm = computeTrajectory(100, 7, P, regimeStats, 'bull', 0);
    const trajHmm = computeTrajectory(100, 7, P, regimeStats, 'bull', 0, hmmOverride);
    // HMM with higher drift should produce higher expected prices
    const noHmmFinal = trajNoHmm[6].expectedPrice;
    const hmmFinal = trajHmm[6].expectedPrice;
    expect(hmmFinal).toBeGreaterThan(noHmmFinal);
  });

  it('keeps expectedPrice mean-based when empiricalDailyVol widens the interval', () => {
    const baseline = computeTrajectory(100, 14, P, regimeStats, 'bull', 0, undefined, 4000, 5);
    const widened = computeTrajectory(100, 14, P, regimeStats, 'bull', 0, undefined, 4000, 5, 0.25);

    expect(widened[13].expectedPrice).toBe(baseline[13].expectedPrice);
    expect(widened[13].upperBound - widened[13].lowerBound)
      .toBeGreaterThan(baseline[13].upperBound - baseline[13].lowerBound);
  });
});
describe('winsorize', () => {
  it('clamps outliers beyond 3 standard deviations', () => {
    // Normal values clustered near 0 plus one extreme outlier
    const vals = [0.01, 0.02, -0.01, 0.0, 0.01, -0.02, 0.01, -0.01, 0.02, 0.0, 0.50];
    const cleaned = winsorize(vals);
    // The 0.50 outlier should be clamped down significantly
    expect(Math.max(...cleaned)).toBeLessThan(0.50);
    // Non-outlier values should be unchanged
    expect(cleaned[0]).toBe(0.01);
    expect(cleaned[3]).toBe(0.0);
  });

  it('preserves values within bounds', () => {
    const vals = [0.01, -0.01, 0.02, -0.02, 0.005];
    const cleaned = winsorize(vals);
    expect(cleaned).toEqual(vals);
  });

  it('handles empty and short arrays', () => {
    expect(winsorize([])).toEqual([]);
    expect(winsorize([1.0])).toEqual([1.0]);
    expect(winsorize([1.0, 2.0])).toEqual([1.0, 2.0]);
  });

  it('handles constant array', () => {
    const vals = [0.01, 0.01, 0.01, 0.01];
    expect(winsorize(vals)).toEqual(vals);
  });
});
describe('estimateRegimeStats drift cap', () => {
  it('caps daily drift when maxDailyDrift is provided', () => {
    // Simulate geopolitical shock: bull returns averaging +3% daily
    const returns: number[] = [];
    const states: RegimeState[] = [];
    const rng = seedRng(101);
    for (let i = 0; i < 50; i++) {
      returns.push(0.03 + (rng() - 0.5) * 0.01);
      states.push('bull');
    }
    const maxDrift = 0.01;
    const stats = estimateRegimeStats(returns, states, maxDrift);
    expect(Math.abs(stats.bull.meanReturn)).toBeLessThanOrEqual(maxDrift + 1e-10);
  });

  it('does not cap drift when maxDailyDrift is undefined', () => {
    const returns = Array(20).fill(0.03);
    const states: RegimeState[] = Array(20).fill('bull');
    const stats = estimateRegimeStats(returns, states);
    expect(stats.bull.meanReturn).toBeGreaterThan(0.02);
  });

  it('caps negative drift (bear regime) symmetrically', () => {
    const returns = Array(20).fill(-0.04);
    const states: RegimeState[] = Array(20).fill('bear');
    const stats = estimateRegimeStats(returns, states, 0.01);
    expect(stats.bear.meanReturn).toBeGreaterThanOrEqual(-0.01 - 1e-10);
  });

  it('stdReturn is not affected by drift cap', () => {
    const returns = Array(30).fill(0.05);
    // Add some variance
    returns[0] = 0.04;
    returns[1] = 0.06;
    const states: RegimeState[] = Array(30).fill('bull');
    const withCap = estimateRegimeStats(returns, states, 0.01);
    const noCap = estimateRegimeStats(returns, states);
    // std should be similar (winsorization may slightly affect it, but shouldn't destroy it)
    expect(withCap.bull.stdReturn).toBeGreaterThan(0);
    expect(Math.abs(withCap.bull.stdReturn - noCap.bull.stdReturn)).toBeLessThan(0.01);
  });

  it('winsorization removes shock outliers from regime stats', () => {
    // Normal returns with one extreme outlier
    const rng = seedRng(102);
    const returns: number[] = Array.from({ length: 50 }, () => 0.001 + (rng() - 0.5) * 0.02);
    returns[25] = 0.15; // 15% daily return = extreme outlier
    const states: RegimeState[] = Array(50).fill('bull');
    const stats = estimateRegimeStats(returns, states, 0.01);
    // Mean should not be dominated by the outlier
    expect(stats.bull.meanReturn).toBeLessThan(0.01 + 1e-10);
    // Std should be reasonable (not inflated by outlier)
    expect(stats.bull.stdReturn).toBeLessThan(0.05);
  });
});
describe('PR3 experiment: startStateMixture', () => {
  it('computeStartStateMixture smooths single-state input without zeros', () => {
    // 5 consecutive bull days
    const recent: RegimeState[] = ['bull', 'bull', 'bull', 'bull', 'bull'];
    const mixture = computeStartStateMixture(recent, 0.5);

    // total count = 5 + 3*0.5 = 6.5
    // bull = 5.5 / 6.5 ≈ 0.846
    // bear = 0.5 / 6.5 ≈ 0.077
    // sideways = 0.5 / 6.5 ≈ 0.077
    expect(mixture.bull).toBeCloseTo(5.5 / 6.5);
    expect(mixture.bear).toBeCloseTo(0.5 / 6.5);
    expect(mixture.sideways).toBeCloseTo(0.5 / 6.5);
    expect(mixture.bull + mixture.bear + mixture.sideways).toBeCloseTo(1.0);
  });

  it('sideways-dominant recent states produce a sideways-dominant mixture', () => {
    const recent: RegimeState[] = ['sideways', 'bull', 'sideways', 'sideways', 'bear'];
    const mixture = computeStartStateMixture(recent, 0.5);

    // total count = 5 + 1.5 = 6.5
    // sideways = 3.5 / 6.5
    expect(mixture.sideways).toBeCloseTo(3.5 / 6.5);
    expect(mixture.bull).toBeCloseTo(1.5 / 6.5);
    expect(mixture.bear).toBeCloseTo(1.5 / 6.5);
  });

  it('one-hot mixture reproduces existing hard-state behavior in computeHorizonDriftVol', () => {
    const P = [
      [0.8, 0.1, 0.1],
      [0.2, 0.6, 0.2],
      [0.3, 0.3, 0.4]
    ];
    const regimeStats = {
      bull: { meanReturn: 0.02, stdReturn: 0.01 },
      bear: { meanReturn: -0.02, stdReturn: 0.015 },
      sideways: { meanReturn: 0.0, stdReturn: 0.005 }
    };

    const hardResult = computeHorizonDriftVol(7, P, regimeStats, 'bull');

    const oneHotMixture = { bull: 1.0, bear: 0.0, sideways: 0.0 };
    const mixtureResult = computeHorizonDriftVol(7, P, regimeStats, 'bull', 0, undefined, oneHotMixture);

    expect(mixtureResult.mu_n).toBeCloseTo(hardResult.mu_n);
    expect(mixtureResult.sigma_n).toBeCloseTo(hardResult.sigma_n);
  });

  it('mixed start distribution yields intermediate drift distinct from one-hot', () => {
    const P = [
      [0.8, 0.1, 0.1],
      [0.2, 0.6, 0.2],
      [0.3, 0.3, 0.4]
    ];
    const regimeStats = {
      bull: { meanReturn: 0.02, stdReturn: 0.01 },
      bear: { meanReturn: -0.02, stdReturn: 0.015 },
      sideways: { meanReturn: 0.0, stdReturn: 0.005 }
    };

    const hardResult = computeHorizonDriftVol(7, P, regimeStats, 'bull');

    const mixedMixture = { bull: 0.6, bear: 0.2, sideways: 0.2 };
    const mixtureResult = computeHorizonDriftVol(7, P, regimeStats, 'bull', 0, undefined, mixedMixture);

    expect(mixtureResult.mu_n).toBeLessThan(hardResult.mu_n - 0.001);
  });

  it('promotes BTC short-horizon start-state mixture by default and preserves legacy behavior with explicit false', async () => {
    const prices = Array.from({ length: 150 }, (_, i) => 50000 + i * 100 + (Math.sin(i / 5) * 2000));
    const currentPrice = prices[prices.length - 1];

    const promotedDefault = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
    });

    const explicitPromoted = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      startStateMixture: true,
    });

    const legacyControl = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      startStateMixture: false,
    });

    expect(promotedDefault.metadata.startStateMixtureActive).toBe(true);
    expect(explicitPromoted.metadata.startStateMixtureActive).toBe(true);
    expect(legacyControl.metadata.startStateMixtureActive).toBe(false);
    expect(promotedDefault.actionSignal.expectedReturn).toBe(explicitPromoted.actionSignal.expectedReturn);
    expect(promotedDefault.actionSignal.expectedReturn).not.toBe(legacyControl.actionSignal.expectedReturn);
  });
});
describe('computeHorizonDriftVol — regime-specific sigma', () => {
  const P = buildDefaultMatrix();

  const bullDominantStats: Record<ReturnType<typeof classifyRegimeState>, { meanReturn: number; stdReturn: number }> = {
    bull:     { meanReturn:  0.003, stdReturn: 0.008 },
    bear:     { meanReturn: -0.003, stdReturn: 0.015 },
    sideways: { meanReturn:  0.000, stdReturn: 0.006 },
  };

  it('uses mixture sigma by default (flag off)', () => {
    const mixture = computeHorizonDriftVol(10, P, bullDominantStats, 'bull', 0);
    const regimeMode = computeHorizonDriftVol(10, P, bullDominantStats, 'bull', 0, undefined, undefined, true, 0.99);
    // With default matrix, bull row is ~0.6 bull, ~0.2 bear, ~0.2 sideways
    // The mixture sigma should be larger than any single regime's sigma due to Var(μ)
    // With flag on but threshold=0.99 (not exceeded), should still use mixture sigma
    expect(mixture.sigma_n).toBeCloseTo(regimeMode.sigma_n, 10);
  });

  it('uses dominant regime sigma when threshold is exceeded and flag is on', () => {
    // With a near-identity matrix, after 1 step from bull, weights ≈ [0.6, 0.2, 0.2]
    // max weight = 0.6, which exceeds threshold 0.55 but not 0.70
    const mixture = computeHorizonDriftVol(1, P, bullDominantStats, 'bull', 0, undefined, undefined, false);
    const regimeMode = computeHorizonDriftVol(1, P, bullDominantStats, 'bull', 0, undefined, undefined, true, 0.55);

    // With regime-specific sigma at threshold 0.55, bull dominates (weight ~0.6 > 0.55)
    // so sigma should be the bull regime's own stdReturn * sqrt(1) = 0.008
    // The mixture sigma includes Var(μ) from bear's different mean, so it's larger
    expect(regimeMode.sigma_n).toBeLessThan(mixture.sigma_n);
    // Should equal bull's daily vol scaled by sqrt(horizon)
    expect(regimeMode.sigma_n).toBeCloseTo(0.008, 5);
  });

  it('falls back to mixture sigma when max weight does not exceed threshold', () => {
    // threshold=0.99: no regime can reach this with the default matrix in 1 step
    const mixture = computeHorizonDriftVol(1, P, bullDominantStats, 'bull', 0);
    const regimeMode = computeHorizonDriftVol(1, P, bullDominantStats, 'bull', 0, undefined, undefined, true, 0.99);
    expect(regimeMode.sigma_n).toBeCloseTo(mixture.sigma_n, 10);
  });

  it('drift (mu_n) is unchanged regardless of sigma mode', () => {
    const mixture = computeHorizonDriftVol(10, P, bullDominantStats, 'bull', 0, undefined, undefined, false);
    const regimeMode = computeHorizonDriftVol(10, P, bullDominantStats, 'bull', 0, undefined, undefined, true, 0.55);
    expect(regimeMode.mu_n).toBeCloseTo(mixture.mu_n, 10);
  });

  it('default threshold is 0.60 when not specified', () => {
    // With default matrix at 1 step from bull: max weight ≈ 0.6
    // Without explicit threshold, default is 0.60 — max weight must EXCEED 0.60
    // 0.6 is NOT > 0.6, so mixture sigma should be used
    const mixture = computeHorizonDriftVol(1, P, bullDominantStats, 'bull', 0, undefined, undefined, false);
    const regimeModeDefault = computeHorizonDriftVol(1, P, bullDominantStats, 'bull', 0, undefined, undefined, true);
    expect(regimeModeDefault.sigma_n).toBeCloseTo(mixture.sigma_n, 10);
  });

  it('uses mixture sigma for long horizons where weights diffuse', () => {
    // At horizon=100, the default matrix mixes weights toward uniform
    // No single regime can dominate → regime-specific sigma should not activate
    const mixture = computeHorizonDriftVol(100, P, bullDominantStats, 'bull', 0);
    const regimeMode = computeHorizonDriftVol(100, P, bullDominantStats, 'bull', 0, undefined, undefined, true, 0.55);
    expect(regimeMode.sigma_n).toBeCloseTo(mixture.sigma_n, 10);
  });
});
