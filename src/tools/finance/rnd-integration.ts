/**
 * Risk-Neutral Density integration — TypeScript mirror of research/models/rnd.py.
 *
 * Extracts forward-looking regime probabilities from Polymarket strike markets,
 * transforms Q-measure to P-measure, and maps to Markov regime buckets.
 */

import { normCDF, normPPF } from '@/utils/stats.js';

/** Default cap on |Market Price of Risk| to prevent runaway shifts when
 *  historical drift estimates are noisy (e.g. crypto bull-run windows
 *  producing μ ≈ 200% / σ ≈ 60% → MPR ≈ 3 → silly P-prob shifts). */
export const DEFAULT_MPR_CAP = 1.5;

/**
 * Convert risk-neutral probability to physical probability via Girsanov shift.
 *
 *   Prob^P(S_T > K) = Phi( Phi^{-1}(Prob^Q(S_T > K)) + λ · sqrt(T) )
 *
 * where the Market Price of Risk is λ = (μ − r_f) / σ.
 *
 * Inputs are **annualised**:
 *   - `historicalDrift`  μ      (e.g. 0.40 for 40 % annual)
 *   - `riskFreeRate`     r_f    (e.g. 0.05 for 5 % annual)
 *   - `volatility`       σ      (e.g. 0.50 for 50 % annual)
 *   - `daysToExpiry`     T      (calendar days; converted to years via /365)
 *
 * `mprCap` clamps |λ| to a finite range (default {@link DEFAULT_MPR_CAP}).
 * The cap is necessary because crypto / momentum windows can produce
 * pathological MPR estimates that translate every Q-prob into ~0 or ~1.
 *
 * Returns a probability in [0, 1].  Boundary inputs (0 / 1) pass through.
 */
export function transformQToP(
  qProb: number,
  historicalDrift: number,
  riskFreeRate: number,
  volatility: number,
  daysToExpiry: number,
  mprCap: number = DEFAULT_MPR_CAP,
): number {
  if (qProb <= 0 || qProb >= 1) {
    return Math.max(0, Math.min(1, qProb));
  }

  const qClipped = Math.max(0.001, Math.min(0.999, qProb));
  const T = Math.max(daysToExpiry, 1) / 365.0;
  const rawMpr = (historicalDrift - riskFreeRate) / Math.max(volatility, 1e-6);
  const cap = Math.max(mprCap, 0);
  const lambdaMpr = Math.max(-cap, Math.min(cap, rawMpr));

  const zQ = normPPF(qClipped);
  const zP = zQ + lambdaMpr * Math.sqrt(T);

  return normCDF(zP);
}

/**
 * Diagnostic variant of {@link transformQToP} that also returns the applied
 * Z-score shift `λ · sqrt(T)`.  Useful when surfacing provenance metadata.
 */
export function transformQToPWithShift(
  qProb: number,
  historicalDrift: number,
  riskFreeRate: number,
  volatility: number,
  daysToExpiry: number,
  mprCap: number = DEFAULT_MPR_CAP,
): { pProb: number; zShift: number; mprUsed: number; mprRaw: number } {
  const T = Math.max(daysToExpiry, 1) / 365.0;
  const rawMpr = (historicalDrift - riskFreeRate) / Math.max(volatility, 1e-6);
  const cap = Math.max(mprCap, 0);
  const mprUsed = Math.max(-cap, Math.min(cap, rawMpr));
  const zShift = mprUsed * Math.sqrt(T);
  return {
    pProb: transformQToP(qProb, historicalDrift, riskFreeRate, volatility, daysToExpiry, mprCap),
    zShift,
    mprUsed,
    mprRaw: rawMpr,
  };
}

/**
 * Fit a Log-Normal distribution to physical survival probabilities.
 *
 * Uses Nelder-Mead least squares on P(S_T > K).
 * Returns { muLn, sigmaLn }.
 *
 * Falls back to drift estimate if < 2 strikes.
 */
export function fitLognormalFromStrikes(
  strikes: number[],
  yesPrices: number[],
  currentPrice: number,
): { muLn: number; sigmaLn: number } {
  if (strikes.length < 2) {
    return { muLn: Math.log(currentPrice), sigmaLn: 0.3 };
  }

  const K = strikes;
  const pObs = yesPrices;

  function survival(muLn: number, sigmaLn: number): number[] {
    if (sigmaLn <= 0) {
      return K.map(() => (muLn > Math.log(K[K.length - 1]) ? 1.0 : 0.0));
    }
    return K.map((k) => {
      const d = (Math.log(k) - muLn) / sigmaLn;
      return 1.0 - normCDF(d);
    });
  }

  function objective(params: number[]): number {
    const [muLn, sigmaLn] = params;
    if (sigmaLn <= 0) return 1e6;
    const pred = survival(muLn, sigmaLn);
    let sse = 0;
    for (let i = 0; i < pred.length; i++) {
      const diff = pred[i] - pObs[i];
      sse += diff * diff;
    }
    return sse;
  }

  // Warm start from median and IQR
  const logK = K.map((k) => Math.log(k));
  logK.sort((a, b) => a - b);
  const med = logK[Math.floor(logK.length / 2)];
  const q1 = logK[Math.floor(logK.length * 0.25)];
  const q3 = logK[Math.floor(logK.length * 0.75)];
  const sigma0 = Math.max((q3 - q1) / 1.349, 0.05);

  // Simple Nelder-Mead (amoeba) implementation
  let simplex = [
    [med, sigma0],
    [med * 1.05, sigma0 * 1.1],
    [med * 0.95, sigma0 * 0.9],
  ];

  for (let iter = 0; iter < 200; iter++) {
    // Evaluate
    const values = simplex.map((p) => objective(p));
    // Sort by value
    const indexed = values.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => a.v - b.v);

    const best = simplex[indexed[0].i];
    const worst = simplex[indexed[2].i];
    const second = simplex[indexed[1].i];

    // Centroid of best + second
    const centroid = [
      (best[0] + second[0]) / 2,
      (best[1] + second[1]) / 2,
    ];

    // Reflect
    const reflected = [
      centroid[0] + 1.0 * (centroid[0] - worst[0]),
      centroid[1] + 1.0 * (centroid[1] - worst[1]),
    ];
    const reflectedVal = objective(reflected);

    if (reflectedVal < values[indexed[1].i]) {
      if (reflectedVal < values[indexed[0].i]) {
        // Expand
        const expanded = [
          centroid[0] + 2.0 * (centroid[0] - worst[0]),
          centroid[1] + 2.0 * (centroid[1] - worst[1]),
        ];
        const expandedVal = objective(expanded);
        simplex[indexed[2].i] = expandedVal < reflectedVal ? expanded : reflected;
      } else {
        simplex[indexed[2].i] = reflected;
      }
    } else {
      // Contract
      const contracted = [
        centroid[0] + 0.5 * (worst[0] - centroid[0]),
        centroid[1] + 0.5 * (worst[1] - centroid[1]),
      ];
      const contractedVal = objective(contracted);
      if (contractedVal < values[indexed[2].i]) {
        simplex[indexed[2].i] = contracted;
      } else {
        // Shrink toward best
        simplex[indexed[1].i] = [
          best[0] + 0.5 * (second[0] - best[0]),
          best[1] + 0.5 * (second[1] - best[1]),
        ];
        simplex[indexed[2].i] = [
          best[0] + 0.5 * (worst[0] - best[0]),
          best[1] + 0.5 * (worst[1] - best[1]),
        ];
      }
    }
  }

  const values = simplex.map((p) => objective(p));
  const bestIdx = values.indexOf(Math.min(...values));
  const [muLn, sigmaLn] = simplex[bestIdx];

  return { muLn, sigmaLn: Math.max(sigmaLn, 1e-4) };
}

/**
 * Map fitted Log-Normal to bull/bear/sideways probabilities.
 */
export function lognormalToRegimeProbabilities(
  muLn: number,
  sigmaLn: number,
  currentPrice: number,
  bullThreshold: number = 0.01,
  bearThreshold: number = -0.01,
): Record<string, number> {
  const bullPrice = currentPrice * (1 + bullThreshold);
  const bearPrice = currentPrice * (1 + bearThreshold);

  function cdf(price: number): number {
    if (sigmaLn <= 0 || price <= 0) {
      return Math.log(price) >= muLn ? 1.0 : 0.0;
    }
    const d = (Math.log(price) - muLn) / sigmaLn;
    return normCDF(d);
  }

  const probBear = cdf(bearPrice);
  const probBull = 1.0 - cdf(bullPrice);
  const probSideways = Math.max(0.0, 1.0 - probBear - probBull);

  return {
    bull: Math.max(0.01, probBull),
    bear: Math.max(0.01, probBear),
    sideways: Math.max(0.01, probSideways),
  };
}

/**
 * Nudge transition matrix toward a target terminal distribution.
 *
 * Nudge strength = 0.5 * (qualityScore / 100), capped at 0.5.
 */
export function nudgeTransitionMatrix(
  P: number[][],
  currentRegime: string,
  targetTerminalDist: Record<string, number>,
  horizon: number,
  qualityScore: number,
): number[][] {
  const nudgeStrength = Math.min(0.5 * (qualityScore / 100.0), 0.5);
  if (nudgeStrength <= 0 || horizon <= 0) {
    return P.map((row) => [...row]);
  }

  const regimeOrder = ['bull', 'bear', 'sideways'];
  const currentIdx = regimeOrder.indexOf(currentRegime);
  if (currentIdx < 0) {
    return P.map((row) => [...row]);
  }

  const nStates = P.length;
  const Pnudged = P.map((row) => [...row]);

  // Compute P^horizon
  function matMul(A: number[][], B: number[][]): number[][] {
    const n = A.length;
    const m = B[0].length;
    const p = B.length;
    const C: number[][] = Array.from({ length: n }, () => Array(m).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < p; k++) {
          sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
      }
    }
    return C;
  }

  function matPow(M: number[][], n: number): number[][] {
    if (n <= 0) {
      const size = M.length;
      return Array.from({ length: size }, (_, i) =>
        Array.from({ length: size }, (_, j) => (i === j ? 1.0 : 0.0)),
      );
    }
    if (n === 1) return M.map((row) => [...row]);
    if (n % 2 === 0) {
      const half = matPow(M, n / 2);
      return matMul(half, half);
    }
    return matMul(M, matPow(M, n - 1));
  }

  const Ph = matPow(P, horizon);
  const currentTerminal = Ph[currentIdx];

  const targetArr = regimeOrder.map((r) => targetTerminalDist[r] ?? 0);
  const targetSum = targetArr.reduce((a, b) => a + b, 0);
  const targetNorm = targetSum > 0 ? targetArr.map((v) => v / targetSum) : targetArr;

  const delta = targetNorm.map((t, i) => t - currentTerminal[i]);

  const row = [...Pnudged[currentIdx]];
  for (let j = 0; j < nStates; j++) {
    row[j] += nudgeStrength * delta[j];
  }

  // Ensure non-negative and row-stochastic
  const clamped = row.map((v) => Math.max(v, 0));
  const rowSum = clamped.reduce((a, b) => a + b, 0);
  if (rowSum > 0) {
    Pnudged[currentIdx] = clamped.map((v) => v / rowSum);
  } else {
    Pnudged[currentIdx] = Array(nStates).fill(1.0 / nStates);
  }

  return Pnudged;
}
