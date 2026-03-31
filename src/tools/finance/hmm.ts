/**
 * Gaussian Hidden Markov Model (HMM) for financial regime detection.
 *
 * Replaces threshold-based observable Markov chains with probabilistic
 * state inference. States are latent — the model learns emission distributions
 * (what returns look like in each state) and transition probabilities jointly
 * via the Baum-Welch (EM) algorithm.
 *
 * Advantages over observable Markov:
 * - No hard threshold artifacts (+1.01% "bull" vs +0.99% "sideways")
 * - Learns from data instead of fixed rules
 * - Provides probabilistic state beliefs, not hard assignments
 * - Naturally captures volatility clustering
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HMMParams {
  /** Number of hidden states */
  nStates: number;
  /** Initial state probabilities π[i] = P(state_0 = i) */
  pi: number[];
  /** Transition matrix A[i][j] = P(state_t+1 = j | state_t = i) */
  A: number[][];
  /** Emission means μ[i] for each state (Gaussian) */
  means: number[];
  /** Emission standard deviations σ[i] for each state (Gaussian) */
  stds: number[];
}

export interface HMMFitResult {
  params: HMMParams;
  logLikelihood: number;
  iterations: number;
  converged: boolean;
}

export interface HMMPrediction {
  /** Most likely current state (Viterbi) */
  currentState: number;
  /** Posterior probability of each state at each time step: gamma[t][i] */
  stateProbabilities: number[][];
  /** Posterior probability of each state at the LAST time step */
  currentStateProbabilities: number[];
  /** n-step ahead state probability forecast */
  forecastProbabilities: number[];
  /** Expected return (weighted by state probabilities and emission means) */
  expectedReturn: number;
  /** Expected volatility (weighted by state probabilities and emission stds) */
  expectedVolatility: number;
}

// ---------------------------------------------------------------------------
// Gaussian PDF
// ---------------------------------------------------------------------------

function gaussianPdf(x: number, mean: number, std: number): number {
  if (std < 1e-10) return x === mean ? 1e10 : 1e-300;
  const z = (x - mean) / std;
  return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
}

// ---------------------------------------------------------------------------
// Initialize HMM parameters using K-means-style heuristic
// ---------------------------------------------------------------------------

export function initializeHMM(observations: number[], nStates: number): HMMParams {
  const sorted = [...observations].sort((a, b) => a - b);
  const n = sorted.length;

  // Initialize means by quantile splitting
  const means: number[] = [];
  for (let i = 0; i < nStates; i++) {
    const idx = Math.floor((i + 0.5) * n / nStates);
    means.push(sorted[Math.min(idx, n - 1)]);
  }
  // Sort means so state 0 = bearish (lowest returns), state N-1 = bullish
  means.sort((a, b) => a - b);

  // Initialize stds as the std of each quantile segment
  const stds: number[] = [];
  for (let i = 0; i < nStates; i++) {
    const lo = Math.floor(i * n / nStates);
    const hi = Math.floor((i + 1) * n / nStates);
    const segment = sorted.slice(lo, hi);
    const segMean = segment.reduce((s, v) => s + v, 0) / segment.length;
    const segVar = segment.reduce((s, v) => s + (v - segMean) ** 2, 0) / segment.length;
    stds.push(Math.max(Math.sqrt(segVar), 1e-6));
  }

  // Uniform initial probabilities
  const pi = Array(nStates).fill(1 / nStates);

  // Slightly diagonal-dominant transition matrix (regime persistence)
  const A: number[][] = Array.from({ length: nStates }, (_, i) =>
    Array.from({ length: nStates }, (_, j) =>
      i === j ? 0.7 : 0.3 / (nStates - 1),
    ),
  );

  return { nStates, pi, A, means, stds };
}

// ---------------------------------------------------------------------------
// Forward algorithm: compute α[t][i] = P(obs_1..t, state_t = i)
// Uses log-scaling for numerical stability.
// ---------------------------------------------------------------------------

export function forward(obs: number[], params: HMMParams): {
  alpha: number[][];
  scales: number[];
  logLikelihood: number;
} {
  const { nStates, pi, A, means, stds } = params;
  const T = obs.length;
  const alpha: number[][] = Array.from({ length: T }, () => Array(nStates).fill(0));
  const scales: number[] = Array(T).fill(0);

  // t = 0
  for (let i = 0; i < nStates; i++) {
    alpha[0][i] = pi[i] * gaussianPdf(obs[0], means[i], stds[i]);
  }
  scales[0] = alpha[0].reduce((s, v) => s + v, 0);
  if (scales[0] < 1e-300) scales[0] = 1e-300;
  for (let i = 0; i < nStates; i++) alpha[0][i] /= scales[0];

  // t = 1..T-1
  for (let t = 1; t < T; t++) {
    for (let j = 0; j < nStates; j++) {
      let sum = 0;
      for (let i = 0; i < nStates; i++) {
        sum += alpha[t - 1][i] * A[i][j];
      }
      alpha[t][j] = sum * gaussianPdf(obs[t], means[j], stds[j]);
    }
    scales[t] = alpha[t].reduce((s, v) => s + v, 0);
    if (scales[t] < 1e-300) scales[t] = 1e-300;
    for (let j = 0; j < nStates; j++) alpha[t][j] /= scales[t];
  }

  const logLikelihood = scales.reduce((s, c) => s + Math.log(c), 0);
  return { alpha, scales, logLikelihood };
}

// ---------------------------------------------------------------------------
// Backward algorithm: compute β[t][i] = P(obs_t+1..T | state_t = i)
// ---------------------------------------------------------------------------

export function backward(obs: number[], params: HMMParams, scales: number[]): number[][] {
  const { nStates, A, means, stds } = params;
  const T = obs.length;
  const beta: number[][] = Array.from({ length: T }, () => Array(nStates).fill(0));

  // t = T-1
  for (let i = 0; i < nStates; i++) beta[T - 1][i] = 1;

  // t = T-2..0
  for (let t = T - 2; t >= 0; t--) {
    for (let i = 0; i < nStates; i++) {
      let sum = 0;
      for (let j = 0; j < nStates; j++) {
        sum += A[i][j] * gaussianPdf(obs[t + 1], means[j], stds[j]) * beta[t + 1][j];
      }
      beta[t][i] = sum / scales[t + 1];
    }
  }

  return beta;
}

// ---------------------------------------------------------------------------
// Baum-Welch (EM) algorithm: fit HMM parameters to observations
// ---------------------------------------------------------------------------

export function baumWelch(
  observations: number[],
  nStates = 3,
  maxIterations = 100,
  tolerance = 1e-4,
  minStd = 1e-4,
): HMMFitResult {
  const T = observations.length;
  if (T < 10) {
    // Not enough data — return default params
    const params = initializeHMM(observations, nStates);
    return { params, logLikelihood: -Infinity, iterations: 0, converged: false };
  }

  let params = initializeHMM(observations, nStates);
  let prevLL = -Infinity;
  let iterations = 0;
  let converged = false;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;

    // E-step: forward-backward
    const { alpha, scales, logLikelihood } = forward(observations, params);
    const beta = backward(observations, params, scales);

    // Convergence check
    if (Math.abs(logLikelihood - prevLL) < tolerance) {
      converged = true;
      break;
    }
    prevLL = logLikelihood;

    // Compute gamma[t][i] = P(state_t = i | obs)
    const gamma: number[][] = Array.from({ length: T }, () => Array(nStates).fill(0));
    for (let t = 0; t < T; t++) {
      let sum = 0;
      for (let i = 0; i < nStates; i++) {
        gamma[t][i] = alpha[t][i] * beta[t][i];
        sum += gamma[t][i];
      }
      if (sum < 1e-300) sum = 1e-300;
      for (let i = 0; i < nStates; i++) gamma[t][i] /= sum;
    }

    // Compute xi[t][i][j] = P(state_t = i, state_t+1 = j | obs)
    const xi: number[][][] = Array.from({ length: T - 1 }, () =>
      Array.from({ length: nStates }, () => Array(nStates).fill(0)),
    );
    for (let t = 0; t < T - 1; t++) {
      let sum = 0;
      for (let i = 0; i < nStates; i++) {
        for (let j = 0; j < nStates; j++) {
          xi[t][i][j] =
            alpha[t][i] *
            params.A[i][j] *
            gaussianPdf(observations[t + 1], params.means[j], params.stds[j]) *
            beta[t + 1][j];
          sum += xi[t][i][j];
        }
      }
      if (sum < 1e-300) sum = 1e-300;
      for (let i = 0; i < nStates; i++) {
        for (let j = 0; j < nStates; j++) {
          xi[t][i][j] /= sum;
        }
      }
    }

    // M-step: update parameters
    const newPi = gamma[0].slice();
    const newA: number[][] = Array.from({ length: nStates }, () => Array(nStates).fill(0));
    const newMeans: number[] = Array(nStates).fill(0);
    const newStds: number[] = Array(nStates).fill(0);

    for (let i = 0; i < nStates; i++) {
      // Transition matrix
      let gammaSum = 0;
      for (let t = 0; t < T - 1; t++) gammaSum += gamma[t][i];
      if (gammaSum < 1e-300) gammaSum = 1e-300;

      for (let j = 0; j < nStates; j++) {
        let xiSum = 0;
        for (let t = 0; t < T - 1; t++) xiSum += xi[t][i][j];
        newA[i][j] = xiSum / gammaSum;
      }

      // Emission parameters
      let totalGamma = 0;
      for (let t = 0; t < T; t++) totalGamma += gamma[t][i];
      if (totalGamma < 1e-300) totalGamma = 1e-300;

      // Mean
      let weightedSum = 0;
      for (let t = 0; t < T; t++) weightedSum += gamma[t][i] * observations[t];
      newMeans[i] = weightedSum / totalGamma;

      // Std
      let weightedVarSum = 0;
      for (let t = 0; t < T; t++) {
        weightedVarSum += gamma[t][i] * (observations[t] - newMeans[i]) ** 2;
      }
      newStds[i] = Math.max(Math.sqrt(weightedVarSum / totalGamma), minStd);
    }

    // Ensure means are sorted (state 0 = bearish, state N-1 = bullish)
    const stateOrder = newMeans.map((m, i) => ({ m, i })).sort((a, b) => a.m - b.m);
    const sortedMeans = stateOrder.map(s => s.m);
    const sortedStds = stateOrder.map(s => newStds[s.i]);
    const sortedPi = stateOrder.map(s => newPi[s.i]);
    const sortedA: number[][] = Array.from({ length: nStates }, (_, i) =>
      Array.from({ length: nStates }, (_, j) => newA[stateOrder[i].i][stateOrder[j].i]),
    );

    params = {
      nStates,
      pi: sortedPi,
      A: sortedA,
      means: sortedMeans,
      stds: sortedStds,
    };
  }

  return { params, logLikelihood: prevLL, iterations, converged };
}

// ---------------------------------------------------------------------------
// Viterbi decoding: find most likely state sequence
// ---------------------------------------------------------------------------

export function viterbi(obs: number[], params: HMMParams): number[] {
  const { nStates, pi, A, means, stds } = params;
  const T = obs.length;

  // Use log probabilities for numerical stability
  const delta: number[][] = Array.from({ length: T }, () => Array(nStates).fill(-Infinity));
  const psi: number[][] = Array.from({ length: T }, () => Array(nStates).fill(0));

  // t = 0
  for (let i = 0; i < nStates; i++) {
    delta[0][i] = Math.log(Math.max(pi[i], 1e-300)) + Math.log(Math.max(gaussianPdf(obs[0], means[i], stds[i]), 1e-300));
  }

  // t = 1..T-1
  for (let t = 1; t < T; t++) {
    for (let j = 0; j < nStates; j++) {
      let bestVal = -Infinity;
      let bestIdx = 0;
      for (let i = 0; i < nStates; i++) {
        const val = delta[t - 1][i] + Math.log(Math.max(A[i][j], 1e-300));
        if (val > bestVal) {
          bestVal = val;
          bestIdx = i;
        }
      }
      delta[t][j] = bestVal + Math.log(Math.max(gaussianPdf(obs[t], means[j], stds[j]), 1e-300));
      psi[t][j] = bestIdx;
    }
  }

  // Backtrace
  const path: number[] = Array(T).fill(0);
  let bestFinal = -Infinity;
  for (let i = 0; i < nStates; i++) {
    if (delta[T - 1][i] > bestFinal) {
      bestFinal = delta[T - 1][i];
      path[T - 1] = i;
    }
  }
  for (let t = T - 2; t >= 0; t--) {
    path[t] = psi[t + 1][path[t + 1]];
  }

  return path;
}

// ---------------------------------------------------------------------------
// Predict: compute posterior state probabilities and n-step forecast
// ---------------------------------------------------------------------------

export function predict(
  observations: number[],
  params: HMMParams,
  forecastHorizon: number,
): HMMPrediction {
  const { nStates, A, means, stds } = params;

  // Forward pass to get current state posteriors
  const { alpha, scales } = forward(observations, params);
  const beta = backward(observations, params, scales);
  const T = observations.length;

  // Compute gamma (posterior state probabilities)
  const gamma: number[][] = Array.from({ length: T }, () => Array(nStates).fill(0));
  for (let t = 0; t < T; t++) {
    let sum = 0;
    for (let i = 0; i < nStates; i++) {
      gamma[t][i] = alpha[t][i] * beta[t][i];
      sum += gamma[t][i];
    }
    if (sum < 1e-300) sum = 1e-300;
    for (let i = 0; i < nStates; i++) gamma[t][i] /= sum;
  }

  const currentStateProbabilities = gamma[T - 1];

  // Most likely state via Viterbi
  const path = viterbi(observations, params);
  const currentState = path[T - 1];

  // n-step forecast: multiply current state distribution by A^n
  const An = matPow(A, forecastHorizon);
  const forecastProbabilities: number[] = Array(nStates).fill(0);
  for (let j = 0; j < nStates; j++) {
    for (let i = 0; i < nStates; i++) {
      forecastProbabilities[j] += currentStateProbabilities[i] * An[i][j];
    }
  }

  // Expected return and volatility (forecast-weighted)
  const expectedReturn = forecastProbabilities.reduce(
    (s, p, i) => s + p * means[i], 0,
  );
  const expectedVolatility = Math.sqrt(
    forecastProbabilities.reduce(
      (s, p, i) => s + p * (stds[i] ** 2 + means[i] ** 2), 0,
    ) - expectedReturn ** 2,
  );

  return {
    currentState,
    stateProbabilities: gamma,
    currentStateProbabilities,
    forecastProbabilities,
    expectedReturn,
    expectedVolatility,
  };
}

// ---------------------------------------------------------------------------
// Matrix power (reused for n-step transition forecast)
// ---------------------------------------------------------------------------

function matPow(M: number[][], n: number): number[][] {
  const size = M.length;
  let result = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => (i === j ? 1 : 0)),
  );
  let base = M.map(row => [...row]);

  let power = n;
  while (power > 0) {
    if (power % 2 === 1) {
      result = matMul(result, base);
    }
    base = matMul(base, base);
    power = Math.floor(power / 2);
  }
  return result;
}

function matMul(A: number[][], B: number[][]): number[][] {
  const n = A.length;
  const m = B[0].length;
  const k = B.length;
  const result: number[][] = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      for (let l = 0; l < k; l++) {
        result[i][j] += A[i][l] * B[l][j];
      }
    }
  }
  return result;
}
