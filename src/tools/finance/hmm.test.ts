/**
 * Tests for Gaussian Hidden Markov Model implementation.
 */
import { describe, it, expect } from 'bun:test';
import {
  initializeHMM,
  forward,
  backward,
  baumWelch,
  viterbi,
  predict,
  type HMMParams,
} from './hmm.js';

// ---------------------------------------------------------------------------
// Helper: generate observations from known HMM parameters
// ---------------------------------------------------------------------------

function generateFromHMM(params: HMMParams, length: number, seed = 42): number[] {
  // Simple seeded RNG (xorshift32)
  let rng = seed;
  const random = () => {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (rng >>> 0) / 4294967296;
  };

  // Box-Muller for normal samples
  const normalRandom = (mean: number, std: number): number => {
    const u1 = random();
    const u2 = random();
    const z = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  };

  const { nStates, pi, A, means, stds } = params;
  const observations: number[] = [];

  // Sample initial state from pi
  let state = 0;
  let cumP = 0;
  const r0 = random();
  for (let i = 0; i < nStates; i++) {
    cumP += pi[i];
    if (r0 < cumP) { state = i; break; }
  }

  for (let t = 0; t < length; t++) {
    // Emit observation from current state
    observations.push(normalRandom(means[state], stds[state]));

    // Transition to next state
    const r = random();
    let cum = 0;
    for (let j = 0; j < nStates; j++) {
      cum += A[state][j];
      if (r < cum) { state = j; break; }
    }
  }

  return observations;
}

// A simple 2-state HMM for testing
const TWO_STATE_PARAMS: HMMParams = {
  nStates: 2,
  pi: [0.5, 0.5],
  A: [[0.9, 0.1], [0.1, 0.9]],
  means: [-0.01, 0.01],
  stds: [0.015, 0.012],
};

// A 3-state HMM (bear / sideways / bull)
const THREE_STATE_PARAMS: HMMParams = {
  nStates: 3,
  pi: [0.33, 0.34, 0.33],
  A: [
    [0.8, 0.15, 0.05],
    [0.10, 0.8, 0.10],
    [0.05, 0.15, 0.8],
  ],
  means: [-0.02, 0.0, 0.015],
  stds: [0.025, 0.008, 0.02],
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('initializeHMM', () => {
  it('creates params with correct dimensions', () => {
    const obs = Array.from({ length: 100 }, (_, i) => Math.sin(i / 10) * 0.02);
    const params = initializeHMM(obs, 3);

    expect(params.nStates).toBe(3);
    expect(params.pi).toHaveLength(3);
    expect(params.A).toHaveLength(3);
    expect(params.A[0]).toHaveLength(3);
    expect(params.means).toHaveLength(3);
    expect(params.stds).toHaveLength(3);
  });

  it('means are sorted ascending (bear < bull)', () => {
    const obs = Array.from({ length: 200 }, (_, i) => (i < 100 ? -0.02 : 0.02) + Math.random() * 0.001);
    const params = initializeHMM(obs, 3);

    for (let i = 1; i < params.nStates; i++) {
      expect(params.means[i]).toBeGreaterThanOrEqual(params.means[i - 1]);
    }
  });

  it('pi sums to 1', () => {
    const obs = Array.from({ length: 50 }, () => Math.random() * 0.04 - 0.02);
    const params = initializeHMM(obs, 3);
    const sum = params.pi.reduce((s, v) => s + v, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it('each row of A sums to 1', () => {
    const obs = Array.from({ length: 50 }, () => Math.random() * 0.04 - 0.02);
    const params = initializeHMM(obs, 3);
    for (const row of params.A) {
      const sum = row.reduce((s, v) => s + v, 0);
      expect(sum).toBeCloseTo(1, 10);
    }
  });
});

describe('forward algorithm', () => {
  it('alpha rows sum to 1 after scaling', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 50);
    const { alpha } = forward(obs, TWO_STATE_PARAMS);

    for (const row of alpha) {
      const sum = row.reduce((s, v) => s + v, 0);
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('returns finite logLikelihood', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 100);
    const { logLikelihood } = forward(obs, TWO_STATE_PARAMS);
    expect(Number.isFinite(logLikelihood)).toBe(true);
  });

  it('logLikelihood is finite', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 100);
    const { logLikelihood } = forward(obs, TWO_STATE_PARAMS);
    expect(Number.isFinite(logLikelihood)).toBe(true);
  });
});

describe('backward algorithm', () => {
  it('returns matrix with correct dimensions', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 50);
    const { scales } = forward(obs, TWO_STATE_PARAMS);
    const beta = backward(obs, TWO_STATE_PARAMS, scales);

    expect(beta).toHaveLength(50);
    expect(beta[0]).toHaveLength(2);
  });

  it('last row is all ones', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 50);
    const { scales } = forward(obs, TWO_STATE_PARAMS);
    const beta = backward(obs, TWO_STATE_PARAMS, scales);

    for (const v of beta[49]) {
      expect(v).toBeCloseTo(1, 10);
    }
  });
});

describe('baumWelch', () => {
  it('converges on data from 2-state HMM', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 500);
    const result = baumWelch(obs, 2, 100, 1e-4);

    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(100);
    expect(Number.isFinite(result.logLikelihood)).toBe(true);
  });

  it('recovers approximate means from 2-state data', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 1000);
    const result = baumWelch(obs, 2, 100);

    // Means should be roughly -0.01 and +0.01
    expect(result.params.means[0]).toBeLessThan(0);
    expect(result.params.means[1]).toBeGreaterThan(0);
  });

  it('recovers approximate means from 3-state data', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 1000);
    const result = baumWelch(obs, 3, 100);

    // State 0 should be bearish (negative mean)
    expect(result.params.means[0]).toBeLessThan(result.params.means[2]);
  });

  it('transition matrix rows sum to 1', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 300);
    const result = baumWelch(obs, 2);

    for (const row of result.params.A) {
      const sum = row.reduce((s, v) => s + v, 0);
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('log-likelihood increases monotonically across iterations', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 200);
    // Run with just 5 iterations to test monotonicity
    const r1 = baumWelch(obs, 3, 1);
    const r2 = baumWelch(obs, 3, 5);
    // More iterations should give better (higher) log-likelihood
    expect(r2.logLikelihood).toBeGreaterThanOrEqual(r1.logLikelihood - 0.01);
  });

  it('handles very short sequences gracefully', () => {
    const obs = [0.01, -0.01, 0.02];
    const result = baumWelch(obs, 2);
    expect(result.converged).toBe(false);
    expect(result.params.nStates).toBe(2);
  });
});

describe('viterbi', () => {
  it('returns path of correct length', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 100);
    const path = viterbi(obs, TWO_STATE_PARAMS);
    expect(path).toHaveLength(100);
  });

  it('all states are valid indices', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 100);
    const path = viterbi(obs, THREE_STATE_PARAMS);
    for (const s of path) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThan(3);
    }
  });

  it('correctly identifies bull state for strongly positive observations', () => {
    // All observations strongly positive → should be in bullish state (highest index)
    const obs = Array(50).fill(0.03);
    const path = viterbi(obs, THREE_STATE_PARAMS);

    // Most of the path should be state 2 (bull, highest mean)
    const bullCount = path.filter(s => s === 2).length;
    expect(bullCount).toBeGreaterThan(40);
  });

  it('correctly identifies bear state for strongly negative observations', () => {
    const obs = Array(50).fill(-0.03);
    const path = viterbi(obs, THREE_STATE_PARAMS);

    const bearCount = path.filter(s => s === 0).length;
    expect(bearCount).toBeGreaterThan(40);
  });
});

describe('predict', () => {
  it('returns valid state probabilities summing to 1', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 200);
    const result = predict(obs, THREE_STATE_PARAMS, 30);

    const sum = result.currentStateProbabilities.reduce((s, v) => s + v, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it('forecast probabilities sum to 1', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 200);
    const result = predict(obs, THREE_STATE_PARAMS, 30);

    const sum = result.forecastProbabilities.reduce((s, v) => s + v, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it('expected return is between min and max emission means', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 200);
    const result = predict(obs, THREE_STATE_PARAMS, 30);

    const minMean = Math.min(...THREE_STATE_PARAMS.means);
    const maxMean = Math.max(...THREE_STATE_PARAMS.means);
    expect(result.expectedReturn).toBeGreaterThanOrEqual(minMean - 0.01);
    expect(result.expectedReturn).toBeLessThanOrEqual(maxMean + 0.01);
  });

  it('expected volatility is positive', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 200);
    const result = predict(obs, THREE_STATE_PARAMS, 30);
    expect(result.expectedVolatility).toBeGreaterThan(0);
  });

  it('long horizon forecast converges to stationary distribution', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 200);
    const shortForecast = predict(obs, THREE_STATE_PARAMS, 1);
    const longForecast = predict(obs, THREE_STATE_PARAMS, 1000);

    // Long forecast should be more uniform than short forecast
    const shortMax = Math.max(...shortForecast.forecastProbabilities);
    const longMax = Math.max(...longForecast.forecastProbabilities);
    expect(longMax).toBeLessThanOrEqual(shortMax + 0.01);
  });

  it('bullish observations shift expected return upward', () => {
    const bullObs = Array(100).fill(0.02);
    const bearObs = Array(100).fill(-0.02);
    const bullResult = predict(bullObs, THREE_STATE_PARAMS, 10);
    const bearResult = predict(bearObs, THREE_STATE_PARAMS, 10);
    // Bullish should be higher than bearish
    expect(bullResult.expectedReturn).toBeGreaterThan(bearResult.expectedReturn);
  });

  it('bearish observations produce negative expected return', () => {
    const obs = Array(100).fill(-0.02);
    const result = predict(obs, THREE_STATE_PARAMS, 10);
    expect(result.expectedReturn).toBeLessThan(0);
  });
});

describe('end-to-end: fit and predict', () => {
  it('fit on 2-state data then predict gives reasonable results', () => {
    const obs = generateFromHMM(TWO_STATE_PARAMS, 500);
    const { params } = baumWelch(obs, 2);
    const prediction = predict(obs, params, 14);

    expect(prediction.currentStateProbabilities).toHaveLength(2);
    expect(Number.isFinite(prediction.expectedReturn)).toBe(true);
    expect(Number.isFinite(prediction.expectedVolatility)).toBe(true);
  });

  it('fit on 3-state data then predict separates regimes', () => {
    const obs = generateFromHMM(THREE_STATE_PARAMS, 800);
    const { params } = baumWelch(obs, 3);
    const prediction = predict(obs, params, 30);

    // States should be ordered: bear (negative) < neutral < bull (positive)
    expect(params.means[0]).toBeLessThan(params.means[2]);

    // Volatility should be positive
    expect(prediction.expectedVolatility).toBeGreaterThan(0);
  });

  it('accuracy: HMM directional prediction on trending data > 60%', () => {
    // Generate data with clear trends
    const params: HMMParams = {
      nStates: 2,
      pi: [0.5, 0.5],
      A: [[0.95, 0.05], [0.05, 0.95]], // very persistent states
      means: [-0.015, 0.015],           // clear direction
      stds: [0.008, 0.008],             // low noise
    };

    const obs = generateFromHMM(params, 500, 123);
    const fitted = baumWelch(obs, 2, 100);

    // Walk-forward: predict next day, check direction
    let correct = 0;
    let total = 0;
    const warmup = 100;
    for (let t = warmup; t < obs.length - 1; t++) {
      const history = obs.slice(0, t + 1);
      const pred = predict(history, fitted.params, 1);
      const predictedDir = pred.expectedReturn > 0 ? 1 : -1;
      const actualDir = obs[t + 1] > 0 ? 1 : -1;
      if (predictedDir === actualDir) correct++;
      total++;
    }

    const accuracy = correct / total;
    expect(accuracy).toBeGreaterThan(0.60);
  });
});
