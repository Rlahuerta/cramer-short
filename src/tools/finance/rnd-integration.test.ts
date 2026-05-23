import { describe, test, expect } from 'bun:test';

/** Error function approximation for test data generation. */
function erf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1.0 / (1.0 + p * absX);
  const y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX));
  return sign * y;
}
import {
  transformQToP,
  transformQToPWithShift,
  fitLognormalFromStrikes,
  lognormalToRegimeProbabilities,
  nudgeTransitionMatrix,
  DEFAULT_MPR_CAP,
} from './rnd-integration.js';

describe('transformQToP', () => {
  test('identity when lambda is zero', () => {
    const q = 0.3;
    const p = transformQToP(q, 0.05, 0.05, 0.5, 30);
    expect(p).toBeCloseTo(q, 6);
  });

  test('increases for bullish (positive risk premium)', () => {
    const q = 0.3;
    const p = transformQToP(q, 0.40, 0.05, 0.5, 30);
    expect(p).toBeGreaterThan(q);
  });

  test('increases for bearish survival (distribution shifts right)', () => {
    const q = 0.8;
    const p = transformQToP(q, 0.40, 0.05, 0.5, 30);
    expect(p).toBeGreaterThan(q);
  });

  test('clips extremes safely', () => {
    const pLow = transformQToP(0.0001, 0.40, 0.05, 0.5, 30);
    const pHigh = transformQToP(0.9999, 0.40, 0.05, 0.5, 30);
    expect(pLow).toBeGreaterThanOrEqual(0);
    expect(pLow).toBeLessThanOrEqual(1);
    expect(pHigh).toBeGreaterThanOrEqual(0);
    expect(pHigh).toBeLessThanOrEqual(1);
  });

  test('boundary zero and one', () => {
    expect(transformQToP(0, 0.1, 0.05, 0.3, 7)).toBe(0);
    expect(transformQToP(1, 0.1, 0.05, 0.3, 7)).toBe(1);
  });

  test('caps pathological MPR (default cap = 1.5)', () => {
    // raw MPR = (3.0 - 0.05) / 0.3 ≈ 9.83 → would push P-prob to ~1.0
    const uncappedShift = 9.83 * Math.sqrt(30 / 365);
    const p = transformQToP(0.30, 3.0, 0.05, 0.3, 30);
    // With the 1.5 cap we expect the shift to be ≪ uncappedShift → result far below ~1
    expect(p).toBeLessThan(0.97);
    expect(uncappedShift).toBeGreaterThan(2.0); // sanity check on the test setup
  });

  test('mprCap parameter narrows the shift', () => {
    const wide = transformQToP(0.30, 0.50, 0.05, 0.30, 30, 5.0);
    const tight = transformQToP(0.30, 0.50, 0.05, 0.30, 30, 0.1);
    expect(wide).toBeGreaterThan(tight); // larger cap allows larger upward shift
  });

  test('default cap constant is exported and equals 1.5', () => {
    expect(DEFAULT_MPR_CAP).toBe(1.5);
  });

  test('transformQToPWithShift surfaces capped MPR provenance', () => {
    const out = transformQToPWithShift(0.30, 3.0, 0.05, 0.3, 30);
    expect(out.mprRaw).toBeGreaterThan(2.0);
    expect(out.mprUsed).toBeCloseTo(1.5, 6);
    expect(out.zShift).toBeCloseTo(1.5 * Math.sqrt(30 / 365), 6);
  });
});

describe('fitLognormalFromStrikes', () => {
  test('recovers known parameters on synthetic data', () => {
    const muTrue = Math.log(50000);
    const sigmaTrue = 0.2;
    const strikes = [40000, 45000, 50000, 55000, 60000];
    const yesPrices = strikes.map((k) => {
      const d = (Math.log(k) - muTrue) / sigmaTrue;
      return 1.0 - ((1 + erf(d / Math.sqrt(2))) / 2);
    });

    const { muLn, sigmaLn } = fitLognormalFromStrikes(strikes, yesPrices, 50000);
    expect(muLn).toBeCloseTo(muTrue, 1);
    expect(sigmaLn).toBeCloseTo(sigmaTrue, 1);
  });

  test('handles arbitrage violations', () => {
    const strikes = [90000, 95000, 100000];
    const yesPrices = [0.40, 0.42, 0.30];
    const { muLn, sigmaLn } = fitLognormalFromStrikes(strikes, yesPrices, 95000);
    expect(sigmaLn).toBeGreaterThan(0);
    for (const k of strikes) {
      const d = (Math.log(k) - muLn) / sigmaLn;
      const p = 1.0 - ((1 + erf(d / Math.sqrt(2))) / 2);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  test('falls back for single strike', () => {
    const { muLn, sigmaLn } = fitLognormalFromStrikes([100000], [0.5], 95000);
    expect(sigmaLn).toBeCloseTo(0.3, 6);
    expect(Number.isFinite(muLn)).toBe(true);
  });
});

describe('lognormalToRegimeProbabilities', () => {
  test('probabilities sum to one', () => {
    const probs = lognormalToRegimeProbabilities(Math.log(50000), 0.2, 50000);
    const total = probs.bull + probs.bear + probs.sideways;
    expect(total).toBeCloseTo(1.0, 6);
  });

  test('concentrated bullish distribution', () => {
    const probs = lognormalToRegimeProbabilities(Math.log(100000), 0.05, 50000);
    expect(probs.bull).toBeGreaterThan(0.90);
    expect(probs.bear).toBeLessThan(0.05);
  });

  test('concentrated bearish distribution', () => {
    const probs = lognormalToRegimeProbabilities(Math.log(30000), 0.05, 50000);
    expect(probs.bear).toBeGreaterThan(0.90);
    expect(probs.bull).toBeLessThan(0.05);
  });

  test('all probabilities are positive', () => {
    const probs = lognormalToRegimeProbabilities(Math.log(50000), 0.2, 50000);
    expect(probs.bull).toBeGreaterThanOrEqual(0.01);
    expect(probs.bear).toBeGreaterThanOrEqual(0.01);
    expect(probs.sideways).toBeGreaterThanOrEqual(0.01);
  });
});

describe('nudgeTransitionMatrix', () => {
  test('preserves row sums', () => {
    const P = [
      [0.7, 0.2, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.2, 0.7],
    ];
    const target = { bull: 0.5, bear: 0.3, sideways: 0.2 };
    const Pnudged = nudgeTransitionMatrix(P, 'bull', target, 7, 80);
    for (const row of Pnudged) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 6);
    }
  });

  test('terminal distribution moves closer to target', () => {
    const P = [
      [0.7, 0.2, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.2, 0.7],
    ];
    const target = { bull: 0.60, bear: 0.25, sideways: 0.15 };

    const Pnudged = nudgeTransitionMatrix(P, 'bull', target, 7, 100);

    function matMul(A: number[][], B: number[][]): number[][] {
      const n = A.length;
      const C: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          for (let k = 0; k < n; k++) {
            C[i][j] += A[i][k] * B[k][j];
          }
        }
      }
      return C;
    }

    function matPow(M: number[][], n: number): number[][] {
      if (n <= 1) return M.map((r) => [...r]);
      const half = matPow(M, Math.floor(n / 2));
      const full = matMul(half, half);
      return n % 2 === 0 ? full : matMul(M, full);
    }

    const PhOrig = matPow(P, 7);
    const PhNudged = matPow(Pnudged, 7);

    const targetArr = [0.60, 0.25, 0.15];
    const errOrig = PhOrig[0].reduce((sum, v, i) => sum + (v - targetArr[i]) ** 2, 0);
    const errNudged = PhNudged[0].reduce((sum, v, i) => sum + (v - targetArr[i]) ** 2, 0);

    expect(errNudged).toBeLessThan(errOrig);
  });

  test('strength scales with quality score', () => {
    const P = [
      [0.7, 0.2, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.2, 0.7],
    ];
    const target = { bull: 0.9, bear: 0.05, sideways: 0.05 };

    const Plow = nudgeTransitionMatrix(P, 'bull', target, 7, 20);
    const Phigh = nudgeTransitionMatrix(P, 'bull', target, 7, 100);

    // Higher quality should produce a more different matrix
    let diffLow = 0;
    let diffHigh = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        diffLow += Math.abs(Plow[i][j] - P[i][j]);
        diffHigh += Math.abs(Phigh[i][j] - P[i][j]);
      }
    }
    expect(diffHigh).toBeGreaterThanOrEqual(diffLow);
  });

  test('identity when target matches current terminal', () => {
    const P = [
      [0.7, 0.2, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.2, 0.7],
    ];

    function matMul(A: number[][], B: number[][]): number[][] {
      const n = A.length;
      const C: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          for (let k = 0; k < n; k++) {
            C[i][j] += A[i][k] * B[k][j];
          }
        }
      }
      return C;
    }

    function matPow(M: number[][], n: number): number[][] {
      if (n <= 1) return M.map((r) => [...r]);
      const half = matPow(M, Math.floor(n / 2));
      const full = matMul(half, half);
      return n % 2 === 0 ? full : matMul(M, full);
    }

    const Ph = matPow(P, 7);
    const target = { bull: Ph[0][0], bear: Ph[0][1], sideways: Ph[0][2] };
    const Pnudged = nudgeTransitionMatrix(P, 'bull', target, 7, 50);

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(Pnudged[i][j]).toBeCloseTo(P[i][j], 6);
      }
    }
  });

  test('zero quality produces no nudge', () => {
    const P = [
      [0.7, 0.2, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.2, 0.7],
    ];
    const target = { bull: 0.9, bear: 0.05, sideways: 0.05 };
    const Pnudged = nudgeTransitionMatrix(P, 'bull', target, 7, 0);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(Pnudged[i][j]).toBeCloseTo(P[i][j], 12);
      }
    }
  });
});
