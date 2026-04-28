/**
 * Beta-HMM — Hidden Markov Model with Beta emissions for bounded [0,1] data.
 *
 * Designed for Polymarket prices (probability series) which Gaussian / Student-t
 * emissions cannot represent natively (unbounded support, no U/J shapes).
 *
 * Reference: Voigt (2025), *Predicting Prediction Markets: A Beta-Hidden Markov
 * Modeling Approach* — see references/prediction-markets/BetaHMMpolymarket-18-1.pdf.
 *
 * Algorithm:
 *   - Emission family: Beta(αᵢ, βᵢ) per state i ∈ {0..K-1}.
 *   - Forward / Backward / Baum-Welch identical in shape to Gaussian HMM,
 *     replacing gaussianPdf with betaPdf.
 *   - M-step refits {αᵢ, βᵢ} via *weighted* method-of-moments (closed form)
 *     using the posterior γ[t][i] as weights. MoM is the standard substitute
 *     for the analytically-intractable Beta MLE in HMM EM updates.
 *
 * This module mirrors the Python implementation in research/models/beta_hmm.py.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BetaEmission {
  alpha: number;
  beta: number;
}

export interface BetaHMMParams {
  nStates: number;
  pi: number[];
  A: number[][];
  emissions: BetaEmission[];
}

export interface BetaHMMFitResult {
  params: BetaHMMParams;
  logLikelihood: number;
  iterations: number;
  converged: boolean;
}

// ---------------------------------------------------------------------------
// log-Gamma (Lanczos) — duplicated from markov-distribution.ts to keep this
// module self-contained.
// ---------------------------------------------------------------------------

function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  x -= 1;
  let a = c[0];
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// ---------------------------------------------------------------------------
// Beta PDF
// ---------------------------------------------------------------------------

/** Beta probability density function. Returns 0 outside (0,1) for safety. */
export function betaPdf(x: number, e: BetaEmission): number {
  if (x <= 0 || x >= 1) return 0;
  if (e.alpha <= 0 || e.beta <= 0) return 0;
  const lnB = lgamma(e.alpha) + lgamma(e.beta) - lgamma(e.alpha + e.beta);
  const logPdf = (e.alpha - 1) * Math.log(x) + (e.beta - 1) * Math.log(1 - x) - lnB;
  // Clamp ridiculously large values for numerical safety.
  return Math.exp(Math.max(-700, Math.min(700, logPdf)));
}

// ---------------------------------------------------------------------------
// Method-of-moments fit (weighted)
// ---------------------------------------------------------------------------

/**
 * Weighted method-of-moments fit for Beta(α,β) given samples and per-sample
 * weights (typically the HMM posterior γ[t][i] for state i).
 *
 * MoM equations:
 *   m = Σwx / Σw                     (weighted mean)
 *   v = Σw(x - m)² / Σw              (weighted variance, MLE form)
 *   ν = m(1 - m)/v − 1               (effective sample size)
 *   α = ν · m,   β = ν · (1 - m)
 *
 * Falls back to Beta(1,1) (uniform) when variance is non-positive or weights
 * vanish. Samples are clipped to [ε, 1−ε] so degenerate boundary observations
 * cannot collapse the variance.
 */
export function fitBetaMoM(weights: number[], samples: number[]): BetaEmission {
  if (weights.length !== samples.length) {
    throw new Error('fitBetaMoM: weights and samples must have equal length');
  }
  const eps = 1e-6;
  let wsum = 0;
  let wxsum = 0;
  for (let i = 0; i < samples.length; i++) {
    const w = weights[i];
    if (!Number.isFinite(w) || w <= 0) continue;
    const x = Math.min(1 - eps, Math.max(eps, samples[i]));
    wsum += w;
    wxsum += w * x;
  }
  if (wsum <= 0) return { alpha: 1, beta: 1 };
  const m = wxsum / wsum;

  let wvsum = 0;
  for (let i = 0; i < samples.length; i++) {
    const w = weights[i];
    if (!Number.isFinite(w) || w <= 0) continue;
    const x = Math.min(1 - eps, Math.max(eps, samples[i]));
    wvsum += w * (x - m) * (x - m);
  }
  const v = wvsum / wsum;
  if (!(v > 1e-10) || m <= 0 || m >= 1) return { alpha: 1, beta: 1 };

  const nu = (m * (1 - m)) / v - 1;
  if (!(nu > 0)) return { alpha: 1, beta: 1 };

  const alpha = nu * m;
  const beta = nu * (1 - m);
  if (!Number.isFinite(alpha) || !Number.isFinite(beta) || alpha <= 0 || beta <= 0) {
    return { alpha: 1, beta: 1 };
  }
  return { alpha, beta };
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

/**
 * Initialise a Beta-HMM by quantile-splitting the observations and fitting
 * Beta(α,β) to each bucket via method-of-moments. Transition matrix starts
 * mildly sticky (0.85 self-loop). Initial state π is uniform.
 */
export function initializeBetaHMM(observations: number[], nStates: number): BetaHMMParams {
  if (nStates < 1) throw new Error('initializeBetaHMM: nStates must be >= 1');
  const sorted = [...observations].sort((a, b) => a - b);
  const n = sorted.length;
  const emissions: BetaEmission[] = [];
  for (let i = 0; i < nStates; i++) {
    const lo = Math.floor((i * n) / nStates);
    const hi = Math.floor(((i + 1) * n) / nStates);
    const slice = sorted.slice(lo, Math.max(hi, lo + 1));
    const w = slice.map(() => 1);
    emissions.push(fitBetaMoM(w, slice));
  }
  const stay = 0.85;
  const off = (1 - stay) / Math.max(nStates - 1, 1);
  const A: number[][] = [];
  for (let i = 0; i < nStates; i++) {
    A.push(Array.from({ length: nStates }, (_, j) => (i === j ? stay : off)));
  }
  const pi = Array.from({ length: nStates }, () => 1 / nStates);
  return { nStates, pi, A, emissions };
}

// ---------------------------------------------------------------------------
// Forward / Backward
// ---------------------------------------------------------------------------

export function forwardBeta(obs: number[], p: BetaHMMParams): { alpha: number[][]; scales: number[]; logLik: number } {
  const T = obs.length;
  const K = p.nStates;
  const alpha: number[][] = Array.from({ length: T }, () => Array(K).fill(0));
  const scales: number[] = Array(T).fill(0);

  // t = 0
  let s0 = 0;
  for (let i = 0; i < K; i++) {
    alpha[0][i] = p.pi[i] * Math.max(betaPdf(obs[0], p.emissions[i]), 1e-300);
    s0 += alpha[0][i];
  }
  scales[0] = s0 || 1;
  for (let i = 0; i < K; i++) alpha[0][i] /= scales[0];

  for (let t = 1; t < T; t++) {
    let st = 0;
    for (let j = 0; j < K; j++) {
      let acc = 0;
      for (let i = 0; i < K; i++) acc += alpha[t - 1][i] * p.A[i][j];
      alpha[t][j] = acc * Math.max(betaPdf(obs[t], p.emissions[j]), 1e-300);
      st += alpha[t][j];
    }
    scales[t] = st || 1;
    for (let j = 0; j < K; j++) alpha[t][j] /= scales[t];
  }

  let logLik = 0;
  for (let t = 0; t < T; t++) logLik += Math.log(scales[t]);
  return { alpha, scales, logLik };
}

export function backwardBeta(obs: number[], p: BetaHMMParams, scales: number[]): number[][] {
  const T = obs.length;
  const K = p.nStates;
  const beta: number[][] = Array.from({ length: T }, () => Array(K).fill(0));
  for (let i = 0; i < K; i++) beta[T - 1][i] = 1 / scales[T - 1];
  for (let t = T - 2; t >= 0; t--) {
    for (let i = 0; i < K; i++) {
      let acc = 0;
      for (let j = 0; j < K; j++) {
        acc += p.A[i][j] * Math.max(betaPdf(obs[t + 1], p.emissions[j]), 1e-300) * beta[t + 1][j];
      }
      beta[t][i] = acc / scales[t];
    }
  }
  return beta;
}

// ---------------------------------------------------------------------------
// Baum-Welch
// ---------------------------------------------------------------------------

export function baumWelchBeta(
  obs: number[],
  init: BetaHMMParams,
  maxIter = 50,
  tol = 1e-4,
): BetaHMMFitResult {
  let p = clone(init);
  const T = obs.length;
  const K = p.nStates;
  let prevLL = -Infinity;
  let converged = false;
  let iter = 0;
  let lastLL = -Infinity;

  for (iter = 0; iter < maxIter; iter++) {
    const { alpha, scales, logLik } = forwardBeta(obs, p);
    const beta = backwardBeta(obs, p, scales);
    lastLL = logLik;

    // γ[t][i] = α[t][i] β[t][i] · scales[t]   (because of scaling normalisation)
    const gamma: number[][] = Array.from({ length: T }, () => Array(K).fill(0));
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let i = 0; i < K; i++) {
        gamma[t][i] = alpha[t][i] * beta[t][i] * scales[t];
        s += gamma[t][i];
      }
      if (s > 0) for (let i = 0; i < K; i++) gamma[t][i] /= s;
    }

    // ξ[t][i][j] expected joint state pairs
    const xiSum: number[][] = Array.from({ length: K }, () => Array(K).fill(0));
    for (let t = 0; t < T - 1; t++) {
      let denom = 0;
      const xi_t: number[][] = Array.from({ length: K }, () => Array(K).fill(0));
      for (let i = 0; i < K; i++) {
        for (let j = 0; j < K; j++) {
          xi_t[i][j] = alpha[t][i] * p.A[i][j]
            * Math.max(betaPdf(obs[t + 1], p.emissions[j]), 1e-300)
            * beta[t + 1][j];
          denom += xi_t[i][j];
        }
      }
      if (denom > 0) {
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) xiSum[i][j] += xi_t[i][j] / denom;
      }
    }

    // M-step
    const piNew = gamma[0].slice();
    const piSum = piNew.reduce((s, v) => s + v, 0) || 1;
    for (let i = 0; i < K; i++) piNew[i] /= piSum;

    const Anew: number[][] = Array.from({ length: K }, () => Array(K).fill(0));
    for (let i = 0; i < K; i++) {
      const denom = xiSum[i].reduce((s, v) => s + v, 0);
      if (denom > 0) {
        for (let j = 0; j < K; j++) Anew[i][j] = xiSum[i][j] / denom;
      } else {
        // Stay-state fallback
        Anew[i] = Array.from({ length: K }, (_, j) => (i === j ? 1 : 0));
      }
    }

    const emissionsNew: BetaEmission[] = [];
    for (let i = 0; i < K; i++) {
      const w = gamma.map(row => row[i]);
      emissionsNew.push(fitBetaMoM(w, obs));
    }

    p = { nStates: K, pi: piNew, A: Anew, emissions: emissionsNew };

    if (Math.abs(logLik - prevLL) < tol) {
      converged = true;
      break;
    }
    prevLL = logLik;
  }

  return { params: p, logLikelihood: lastLL, iterations: iter + 1, converged };
}

// ---------------------------------------------------------------------------
// Viterbi (most-likely state path)
// ---------------------------------------------------------------------------

export function viterbiBeta(obs: number[], p: BetaHMMParams): number[] {
  const T = obs.length;
  const K = p.nStates;
  const delta: number[][] = Array.from({ length: T }, () => Array(K).fill(-Infinity));
  const psi: number[][] = Array.from({ length: T }, () => Array(K).fill(0));

  for (let i = 0; i < K; i++) {
    delta[0][i] = Math.log(Math.max(p.pi[i], 1e-300))
      + Math.log(Math.max(betaPdf(obs[0], p.emissions[i]), 1e-300));
  }

  for (let t = 1; t < T; t++) {
    for (let j = 0; j < K; j++) {
      let bestVal = -Infinity;
      let bestI = 0;
      for (let i = 0; i < K; i++) {
        const val = delta[t - 1][i] + Math.log(Math.max(p.A[i][j], 1e-300));
        if (val > bestVal) {
          bestVal = val;
          bestI = i;
        }
      }
      delta[t][j] = bestVal + Math.log(Math.max(betaPdf(obs[t], p.emissions[j]), 1e-300));
      psi[t][j] = bestI;
    }
  }

  const path: number[] = Array(T).fill(0);
  let bestFinal = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < K; i++) {
    if (delta[T - 1][i] > bestVal) {
      bestVal = delta[T - 1][i];
      bestFinal = i;
    }
  }
  path[T - 1] = bestFinal;
  for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1][path[t + 1]];
  return path;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clone(p: BetaHMMParams): BetaHMMParams {
  return {
    nStates: p.nStates,
    pi: p.pi.slice(),
    A: p.A.map(row => row.slice()),
    emissions: p.emissions.map(e => ({ ...e })),
  };
}
