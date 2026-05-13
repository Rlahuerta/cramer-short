import {
  NUM_STATES,
  REGIME_STATES,
  STATE_INDEX,
  type MarkovDistributionPoint,
  type PriceThreshold,
  type RegimeState,
  type TrajectoryPoint,
  type TransitionMatrix,
} from './core.js';
import { matPow } from './transition.js';
import {
  jumpDriftCompensator,
  type JumpEventSpec,
} from '../jump-diffusion.js';

/**
 * Standard normal CDF Φ(x) via Abramowitz & Stegun erf approximation (eq 7.1.26).
 * The A&S formula computes erf(t) which equals Φ(t√2)*2−1, so we
 * rescale the input by 1/√2 to obtain the true standard normal CDF.
 */
export function normalCDF(x: number): number {
  // Rescale: Φ(x) = 0.5*(1 + erf(x/√2))
  const z = x / Math.SQRT2;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = z < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(z));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
  return 0.5 * (1 + sign * y);
}

// ---------------------------------------------------------------------------
// 6. interpolateDistribution
// ---------------------------------------------------------------------------

export interface RegimeStats {
  meanReturn: number;  // daily mean log-return in this regime
  stdReturn:  number;  // daily std of log-return in this regime
}

/**
 * Estimate per-regime empirical return statistics from historical data.
 * Falls back to literature-informed defaults when data is sparse.
 */
/**
 * Winsorize an array: clamp values beyond ±k standard deviations to the boundary.
 * Prevents extreme outliers (geopolitical shocks, flash crashes) from contaminating
 * regime statistics.
 */
export function winsorize(values: number[], k = 3.0): number[] {
  if (values.length < 3) return [...values];
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
  if (std < 1e-12) return [...values];
  const lo = mean - k * std;
  const hi = mean + k * std;
  return values.map(v => Math.max(lo, Math.min(hi, v)));
}

export function estimateRegimeStats(
  returns: number[],
  states: RegimeState[],
  maxDailyDrift?: number,
): Record<RegimeState, RegimeStats> {
  const defaults: Record<RegimeState, RegimeStats> = {
    bull:          { meanReturn:  0.005, stdReturn: 0.010 },
    bear:          { meanReturn: -0.005, stdReturn: 0.012 },
    sideways:      { meanReturn:  0.000, stdReturn: 0.006 },
  };

  const bins: Record<RegimeState, number[]> = {
    bull: [], bear: [], sideways: [],
  };

  for (let i = 0; i < Math.min(returns.length, states.length); i++) {
    bins[states[i]].push(returns[i]);
  }

  const result = { ...defaults };
  for (const [state, vals] of Object.entries(bins) as [RegimeState, number[]][]) {
    if (vals.length >= 5) {
      // Winsorize at 3σ to remove shock outliers before computing stats
      const cleaned = winsorize(vals);
      const mean = cleaned.reduce((s, v) => s + v, 0) / cleaned.length;
      const variance = cleaned.reduce((s, v) => s + (v - mean) ** 2, 0) / cleaned.length;
      let cappedMean = mean;
      // Cap daily drift to prevent shock-period contamination
      if (maxDailyDrift !== undefined && maxDailyDrift > 0) {
        cappedMean = Math.max(-maxDailyDrift, Math.min(maxDailyDrift, mean));
      }
      result[state] = { meanReturn: cappedMean, stdReturn: Math.sqrt(variance) };
    }
  }
  return result;
}

/**
 * Compute the mixing-time weight: how much to trust the Markov regime signal
 * vs. anchor-only at a given horizon.
 *
 * weight = exp(−ρ × n) where ρ = second-largest eigenvalue of P.
 * Near 1 at short horizons (Markov-dominant).
 * Approaches 0 at long horizons (Polymarket anchors dominate).
 */
export function computeMixingWeight(secondEigenvalue: number, horizon: number): number {
  return Math.exp(-secondEigenvalue * horizon);
}

/**
 * Log-normal survival function: P(price > X | current price S₀, drift μ_n, vol σ_n).
 * P(X > target) = 1 − Φ( (ln(target/S₀) − μ_n) / σ_n )
 */
export function logNormalSurvival(
  currentPrice: number,
  targetPrice: number,
  driftN: number,    // n-day log-space drift
  volN: number,      // n-day log-space vol
): number {
  if (volN <= 0) return targetPrice < currentPrice ? 1 : 0;
  const z = (Math.log(targetPrice / currentPrice) - driftN) / volN;
  return 1 - normalCDF(z);
}

/**
 * Regularized incomplete beta function I_x(a, b) via continued fraction.
 * Used to compute Student-t CDF. Lentz's method for convergence.
 */
function regularizedBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  // Use symmetry relation when x > (a+1)/(a+b+2) for better convergence
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - regularizedBeta(1 - x, b, a);
  }

  const lnBeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;

  // Lentz's continued fraction
  let f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  f = d;

  for (let m = 1; m <= 200; m++) {
    // Even step
    let numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + numerator * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + numerator / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    f *= c * d;

    // Odd step
    numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + numerator * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + numerator / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = c * d;
    f *= delta;

    if (Math.abs(delta - 1) < 1e-10) break;
  }

  return front * f;
}

/** Log-gamma via Stirling's approximation (Lanczos coefficients). */
function lgamma(x: number): number {
  const g = 7;
  const coef = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  x -= 1;
  let a = coef[0];
  for (let i = 1; i < g + 2; i++) {
    a += coef[i] / (x + i);
  }
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Student-t CDF: P(T ≤ x) for T ~ t(ν degrees of freedom).
 * Uses regularized incomplete beta function.
 */
export function studentTCDF(x: number, nu: number): number {
  if (nu <= 0) return normalCDF(x); // degenerate: fall back to normal
  const t2 = x * x;
  const betaArg = nu / (nu + t2);
  const ibeta = regularizedBeta(betaArg, nu / 2, 0.5);
  if (x >= 0) {
    return 1 - 0.5 * ibeta;
  } else {
    return 0.5 * ibeta;
  }
}

/**
 * Inverse Student-t CDF via bisection: find x such that CDF(x, nu) = p.
 * Used for drift-based calibration to convert a target P(up) into a drift value.
 */
export function inverseStudentTCDF(p: number, nu: number): number {
  if (p <= 0) return -50;
  if (p >= 1) return 50;
  if (Math.abs(p - 0.5) < 1e-12) return 0;
  let lo = -50, hi = 50;
  for (let iter = 0; iter < 100; iter++) {
    const mid = (lo + hi) / 2;
    const cdf = studentTCDF(mid, nu);
    if (cdf < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-10) break;
  }
  return (lo + hi) / 2;
}

/**
 * Fat-tailed survival function: P(price > X) using Student-t distribution.
 * Same interface as logNormalSurvival but uses Student-t with `nu` degrees
 * of freedom (default 5, typical for daily equity returns).
 *
 * The scaling adjusts the t-distribution standard deviation to match
 * the Gaussian vol parameter: σ_t = σ_n × sqrt((ν-2)/ν) for ν>2.
 */
export function studentTSurvival(
  currentPrice: number,
  targetPrice: number,
  driftN: number,
  volN: number,
  nu = 5,
): number {
  if (volN <= 0) return targetPrice < currentPrice ? 1 : 0;
  if (targetPrice <= 0) return 1; // price can't go below 0; survival = 1
  // Scale vol to match t-distribution variance: Var(t_ν) = ν/(ν-2) for ν>2
  const scaledVol = nu > 2 ? volN * Math.sqrt((nu - 2) / nu) : volN;
  const z = (Math.log(targetPrice / currentPrice) - driftN) / scaledVol;
  // Guard extreme z-scores: regularizedBeta can diverge beyond |z|~50
  if (!Number.isFinite(z)) return z > 0 ? 0 : 1;
  const clamped = Math.max(-50, Math.min(50, z));
  const cdf = studentTCDF(clamped, nu);
  return Number.isFinite(cdf) ? 1 - cdf : (clamped > 0 ? 0 : 1);
}

/**
 * Run a single Monte Carlo random walk through the transition matrix for n steps,
 * starting from `initialStateIdx`. Returns the final n-step regime weight vector.
 */
function singleMarkovWalk(
  P: TransitionMatrix,
  initialStateIdx: number,
  n: number,
): number[] {
  // Use the n-step transition row for the initial state
  const Pn = matPow(P, n);
  return Pn[initialStateIdx];
}

/**
 * Compute the initial probability vector over the current regime using the last K observed states.
 * This replaces a hard start state (one-hot) with a smoothed mixture.
 */
export function computeStartStateMixture(
  recentStates: RegimeState[],
  alpha = 0.5
): Record<RegimeState, number> {
  const counts: Record<RegimeState, number> = { bull: alpha, bear: alpha, sideways: alpha } as Record<RegimeState, number>;
  for (const s of recentStates) {
    if (counts[s] !== undefined) counts[s]++;
  }
  const total = counts.bull + counts.bear + counts.sideways;
  return {
    bull: counts.bull / total,
    bear: counts.bear / total,
    sideways: counts.sideways / total,
  };
}

export function normalizeStateWeightVector(weights: readonly number[]): number[] {
  const sanitized = REGIME_STATES.map((_, index) => {
    const value = weights[index] ?? 0;
    return Number.isFinite(value) && value > 0 ? value : 0;
  });
  const sum = sanitized.reduce((total, value) => total + value, 0);
  if (!(sum > 0)) return Array(NUM_STATES).fill(1 / NUM_STATES);
  return sanitized.map((value) => value / sum);
}

export function computeTerminalStateWeights(
  horizon: number,
  P: TransitionMatrix,
  initialState: RegimeState,
  startMixture?: Record<RegimeState, number>,
): number[] {
  const Pn = matPow(P, horizon);
  if (!startMixture) return [...Pn[STATE_INDEX[initialState]]];

  const weights = [0, 0, 0];
  for (const state of REGIME_STATES) {
    const mixtureWeight = startMixture[state];
    const idx = STATE_INDEX[state];
    for (let j = 0; j < NUM_STATES; j++) {
      weights[j] += mixtureWeight * Pn[idx][j];
    }
  }
  return normalizeStateWeightVector(weights);
}

/**
 * Compute the effective n-step drift and volatility from the Markov chain.
 * Extracted so both interpolateDistribution and calibration logic can reuse it.
 */
export function computeHorizonDriftVol(
  horizon: number,
  P: TransitionMatrix,
  regimeStats: Record<RegimeState, RegimeStats>,
  initialState: RegimeState,
  momentumAdjustment = 0,
  hmmOverride?: { drift: number; vol: number; weight: number },
  startMixture?: Record<RegimeState, number>,
  regimeSpecificSigma?: boolean,
  regimeSpecificSigmaThreshold?: number,
  garchScales?: readonly number[],
  terminalStateWeights?: readonly number[],
): { mu_n: number; sigma_n: number } {
  const stateWeights = terminalStateWeights
    ? normalizeStateWeightVector(terminalStateWeights)
    : computeTerminalStateWeights(horizon, P, initialState, startMixture);

  const mu_obs = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeights[i] * regimeStats[state].meanReturn, 0,
  );
  // Variance of the mixture: E[σ²] + Var(μ).
  // E[σ²] captures within-regime volatility; Var(μ) captures between-regime
  // mean differences — critical when regime weights are mixed and means are well-separated.
  const varOfMeans = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeights[i] * (regimeStats[state].meanReturn - mu_obs) ** 2, 0,
  );
  const mixtureSigmaObs = Math.sqrt(
    REGIME_STATES.reduce(
      (s, state, i) => s + stateWeights[i] * regimeStats[state].stdReturn ** 2, 0,
    ) + varOfMeans,
  );

  // Phase 7: when regime weights are concentrated and flag is enabled,
  // use the dominant regime's own sigma instead of the mixture sigma.
  // The mixture sigma inflates variance via Var(μ) when weights are mixed,
  // but when one regime dominates, that regime's own volatility is more appropriate.
  const maxWeight = Math.max(...stateWeights);
  const threshold = regimeSpecificSigmaThreshold ?? 0.60;
  const dominantIdx = stateWeights.indexOf(maxWeight);
  const dominantSigma = regimeStats[REGIME_STATES[dominantIdx]].stdReturn;
  const useRegimeSigma = regimeSpecificSigma === true && maxWeight > threshold;
  const sigma_obs = useRegimeSigma ? dominantSigma : mixtureSigmaObs;

  let mu_eff: number;
  let sigma_eff: number;
  if (hmmOverride) {
    const w = hmmOverride.weight;
    mu_eff = w * hmmOverride.drift + (1 - w) * mu_obs;
    sigma_eff = w * hmmOverride.vol + (1 - w) * sigma_obs;
  } else {
    mu_eff = mu_obs;
    sigma_eff = sigma_obs;
  }

  let sigma_n = sigma_eff * Math.sqrt(horizon);
  if (garchScales && garchScales.length > 0) {
    let varianceScale = 0;
    for (let d = 0; d < horizon; d++) {
      const k = garchScales[d] ?? 1;
      varianceScale += Number.isFinite(k) && k > 0 ? k * k : 1;
    }
    sigma_n = sigma_eff * Math.sqrt(varianceScale);
  }

  return {
    mu_n: horizon * (mu_eff + momentumAdjustment),
    sigma_n,
  };
}

/**
 * Compute a day-by-day price trajectory for days 1..N.
 *
 * Uses a SINGLE set of Monte Carlo random walks and samples the path at each day,
 * rather than N independent simulations. This ensures:
 * 1. CI widths monotonically increase with horizon
 * 2. ~7× faster than N separate MC runs
 *
 * Returns one TrajectoryPoint per day with expected price, 90% CI, P(up), and regime.
 */
export function computeTrajectory(
  currentPrice: number,
  days: number,
  P: TransitionMatrix,
  regimeStats: Record<RegimeState, { meanReturn: number; stdReturn: number }>,
  initialState: RegimeState,
  momentumAdjustment: number,
  hmmOverride?: { drift: number; vol: number; weight: number },
  nSamples = 1000,
  nu = 5,
  empiricalDailyVol?: number,
  startMixture?: Record<RegimeState, number>,
  /**
   * Polymarket-informed Merton jump events (Idea 2).  When undefined the inner
   * MC loop is **byte-identical** to the pre-jump implementation — no extra
   * RNG draws are consumed.  Supply a non-empty array to enable per-event
   * Bernoulli(λ_e·Δt) jumps with log-jump magnitude N(μ_J,e, σ_J,e²).
   * Drift is compensated by `Σ_e λ_e·(exp(μ_J,e + σ_J,e²/2) − 1)` to keep
   * E[r_t] equal to the pre-jump regime drift.
   */
  jumpSpec?: readonly JumpEventSpec[],
  /**
   * R4 Phase B — optional per-day GARCH(1,1) volatility multipliers.  When
   * provided, `dailyVols[d] *= garchScales[d]` for d in [0, horizonDays).
   * When undefined or empty, behaviour is byte-identical (no extra
   * arithmetic, no extra RNG draws).
   */
  garchScales?: readonly number[],
): TrajectoryPoint[] {
  const initialIdx = STATE_INDEX[initialState];
  const trajectory: TrajectoryPoint[] = [];

  // Pre-compute regime weights at each day via matrix powers
  const regimeWeightsPerDay: number[][] = [];
  for (let d = 1; d <= days; d++) {
    const Pd = matPow(P, d);
    if (startMixture) {
      const weights = [0, 0, 0];
      for (const state of REGIME_STATES) {
        const w = startMixture[state];
        const idx = STATE_INDEX[state];
        for (let j = 0; j < 3; j++) {
          weights[j] += w * Pd[idx][j];
        }
      }
      regimeWeightsPerDay.push(weights);
    } else {
      regimeWeightsPerDay.push(Pd[initialIdx]);
    }
  }

  // Compute per-day mixture drift and vol from regime weights
  const dailyDrifts = new Array(days).fill(0);
  const dailyVols = new Array(days).fill(0);

  for (let d = 0; d < days; d++) {
    const weights = regimeWeightsPerDay[d];

    const muObs = REGIME_STATES.reduce(
      (s, state, i) => s + weights[i] * regimeStats[state].meanReturn, 0,
    );

    const varOfMeans = REGIME_STATES.reduce(
      (s, state, i) => s + weights[i] * (regimeStats[state].meanReturn - muObs) ** 2, 0,
    );
    const expectedVar = REGIME_STATES.reduce(
      (s, state, i) => s + weights[i] * regimeStats[state].stdReturn ** 2, 0,
    );
    let sigmaObs = Math.sqrt(expectedVar + varOfMeans);

    let muDay = muObs + momentumAdjustment;

    // Apply HMM override per-day (HMM drift/vol are daily quantities)
    if (hmmOverride) {
      const w = hmmOverride.weight;
      muDay = w * hmmOverride.drift + (1 - w) * muDay;
      sigmaObs = w * hmmOverride.vol + (1 - w) * sigmaObs;
    }

    // Use empirical vol as floor when provided
    if (empiricalDailyVol) {
      sigmaObs = Math.max(sigmaObs, empiricalDailyVol);
    }

    dailyDrifts[d] = muDay;
    dailyVols[d] = sigmaObs;
  }

  // R4 Phase B — multiplicatively apply GARCH per-day vol scalars when
  // provided.  `garchScales` is a length-`days` array in unconditional-σ
  // units.  Undefined/empty ⇒ no-op ⇒ byte-identical to pre-Phase-B.
  if (garchScales && garchScales.length > 0) {
    const n = Math.min(days, garchScales.length);
    for (let d = 0; d < n; d++) {
      const k = garchScales[d];
      if (Number.isFinite(k) && k > 0) {
        dailyVols[d] *= k;
      }
    }
  }

  // Idea 2 — Merton drift compensator.  Computed once (events are time-stationary
  // over the trajectory horizon).  When jumpSpec is undefined or empty,
  // compensator = 0 ⇒ dailyDrifts unchanged ⇒ rest of the loop is byte-identical
  // to the pre-jump implementation.
  const hasJumps = jumpSpec !== undefined && jumpSpec.length > 0;
  if (hasJumps) {
    const compensator = jumpDriftCompensator(jumpSpec!);
    for (let d = 0; d < days; d++) {
      dailyDrifts[d] -= compensator;
    }
  }

  // Run shared Monte Carlo with per-day mixture drift/vol
  const paths: number[][] = [];
  for (let s = 0; s < nSamples; s++) {
    const path = new Array(days);
    let cumLogReturn = 0;
    for (let d = 0; d < days; d++) {
      const u = Math.random();
      const z = inverseStudentTCDF(u, nu);
      const scaledVol = nu > 2 ? dailyVols[d] * Math.sqrt((nu - 2) / nu) : dailyVols[d];
      cumLogReturn += dailyDrifts[d] + z * scaledVol;

      // Jump term.  Only consume RNG when at least one event exists, so the
      // default-off path stays bit-for-bit identical to the legacy run.
      if (hasJumps) {
        for (const e of jumpSpec!) {
          if (Math.random() < e.dailyIntensity) {
            // Box-Muller via two uniforms — keeps the dependency surface
            // identical to the diffusion draw above (no extra libs).
            const u1 = Math.max(1e-12, Math.random());
            const u2 = Math.random();
            const zJ = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            cumLogReturn += e.meanLogJump + zJ * e.stdLogJump;
          }
        }
      }

      path[d] = cumLogReturn;
    }
    paths.push(path);
  }

  for (let d = 1; d <= days; d++) {
    const dayIdx = d - 1;
    const stateWeights = regimeWeightsPerDay[dayIdx];

    // Cumulative drift and vol from daily arrays
    let mu_n = 0;
    let sigmaSq = 0;
    for (let i = 0; i < d; i++) {
      mu_n += dailyDrifts[i];
      sigmaSq += dailyVols[i] ** 2;
    }
    const sigma_n = Math.sqrt(sigmaSq);

    // Expected price from cumulative drift
    const analyticalExpected = currentPrice * Math.exp(mu_n);

    // CI bounds from Monte Carlo paths
    const prices = paths.map(path => currentPrice * Math.exp(path[dayIdx]));
    prices.sort((a, b) => a - b);
    const p5Idx = Math.max(0, Math.floor(nSamples * 0.05) - 1);
    const p95Idx = Math.min(nSamples - 1, Math.ceil(nSamples * 0.95));
    const lowerBound = prices[p5Idx];
    const upperBound = prices[p95Idx];

    // Keep trajectory expectedPrice on the mean path even when an empirical-vol
    // floor widens the MC interval. Otherwise the trajectory can silently switch
    // from mean to median semantics and contradict the terminal expected price.
    const expectedPrice = analyticalExpected;

    // P(up) from Student-t survival at currentPrice
    const pUp = studentTSurvival(currentPrice, currentPrice, mu_n, sigma_n, nu);

    // Cumulative return
    const ret = (expectedPrice - currentPrice) / currentPrice;
    const sign = ret >= 0 ? '+' : '';
    const cumulativeReturn = `${sign}${(ret * 100).toFixed(1)}%`;

    // Most likely regime at this horizon
    let maxWeight = -1;
    let regime: RegimeState = initialState;
    REGIME_STATES.forEach((state, i) => {
      if (stateWeights[i] > maxWeight) {
        maxWeight = stateWeights[i];
        regime = state;
      }
    });

    trajectory.push({
      day: d,
      expectedPrice: Math.round(expectedPrice * 100) / 100,
      lowerBound: Math.round(lowerBound * 100) / 100,
      upperBound: Math.round(upperBound * 100) / 100,
      pUp: Math.round(pUp * 1000) / 1000,
      cumulativeReturn,
      regime,
    });
  }

  return trajectory;
}

/**
 * Build probability distribution across price levels, blending Markov estimates
 * with Polymarket anchors using the mixing-time weight.
 *
 * Confidence intervals are computed via Monte Carlo simulation (N=1000 walks),
 * taking the 5th/95th percentile of the resulting probability distribution.
 */
export function interpolateDistribution(
  currentPrice: number,
  horizon: number,
  P: TransitionMatrix,
  regimeStats: Record<RegimeState, RegimeStats>,
  initialState: RegimeState,
  anchors: PriceThreshold[],
  secondEigenvalue: number,
  numLevels = 20,
  monteCarloSamples = 1000,
  ciWidthMultiplier = 1.0,
  momentumAdjustment = 0,
  hmmOverride?: { drift: number; vol: number; weight: number },
  dailyVol?: number,
  startMixture?: Record<RegimeState, number>,
  nu = 5,
  regimeSpecificSigma?: boolean,
  regimeSpecificSigmaThreshold?: number,
  /** Number of historical returns used to estimate drift/vol; controls MC perturbation scale */
  sampleSize?: number,
  /** Optional per-day GARCH volatility multipliers for distribution/CI variance. */
  garchScales?: readonly number[],
  /** Optional horizon-level regime weights override for the final forecast state mix. */
  terminalStateWeights?: readonly number[],
): MarkovDistributionPoint[] {
  // Adaptive grid: scale with volatility so CI covers ≥3σ for all assets.
  // Fixed 1.5%/step only covers ±14% total — fine for SPY (~1%/day) but
  // too narrow for TSLA (~3%/day) or BTC (~4%/day) where a 14-day 2σ move is 22-30%.
  const vol = dailyVol ?? 0.015;
  const volRange = 3.5 * vol * Math.sqrt(horizon);
  // Clamp to [0.15, 0.90] — minPrice must remain positive (>10% of currentPrice)
  const halfRange = Math.max(0.15, Math.min(0.90, volRange));
  let minPrice = currentPrice * (1 - halfRange);
  let maxPrice = currentPrice * (1 + halfRange);

  // Extend grid range to include all Polymarket anchors (fixes sparse-anchor bug)
  for (const a of anchors) {
    if (a.price < minPrice) minPrice = a.price * 0.95;
    if (a.price > maxPrice) maxPrice = a.price * 1.05;
  }

  const prices: number[] = [];
  for (let i = 0; i <= numLevels; i++) {
    prices.push(minPrice * Math.pow(maxPrice / minPrice, i / numLevels));
  }

  // Merge anchor prices into the grid so they are never missed
  for (const a of anchors) {
    const closestDist = prices.reduce(
      (best, p) => Math.min(best, Math.abs(p - a.price) / a.price), Infinity,
    );
    if (closestDist > 0.005) prices.push(a.price);
  }
  prices.sort((a, b) => a - b);

  const mixWeight = computeMixingWeight(secondEigenvalue, horizon);

  // Compute regime-weighted drift and vol via shared helper
  const { mu_n, sigma_n } = computeHorizonDriftVol(
    horizon, P, regimeStats, initialState, momentumAdjustment, hmmOverride, startMixture,
    regimeSpecificSigma, regimeSpecificSigmaThreshold, garchScales, terminalStateWeights,
  );

  // Nearest anchor lookup helper with distance-based dampening.
  // Anchors far from current price are less trustworthy (often illiquid/speculative).
  // Apply exponential decay: weight = exp(-k * distance²) where distance = |price - current| / current.
  // At 20% away: weight ≈ 0.67. At 40%: weight ≈ 0.20. At 60%+: weight ≈ 0.03.
  const findAnchor = (price: number) => {
    const TOLERANCE_PCT = 0.02;
    const raw = anchors.find(a => Math.abs(a.price - price) / price < TOLERANCE_PCT);
    if (!raw) return undefined;
    // Compute distance-decay factor
    const distFromCurrent = Math.abs(raw.price - currentPrice) / currentPrice;
    const DISTANCE_DECAY_K = 5.0; // controls how fast far anchors are dampened
    const distanceWeight = Math.exp(-DISTANCE_DECAY_K * distFromCurrent * distFromCurrent);
    return { ...raw, distanceWeight };
  };

  // Monte Carlo: perturb drift/vol within sampling uncertainty.
  // Standard error of the drift estimator is σ/sqrt(N). Without N, the previous
  // ±10% σ_n perturbation was ad-hoc and made the CI essentially independent of
  // sample size. With N supplied, scale the perturbation amplitude by 1/sqrt(N)
  // (clipped to the legacy ±10% σ_n band when N is small or absent so existing
  // CI behaviour is preserved on small-history calls). For typical N≈250
  // (1y daily history) this shrinks the band to ~0.06 σ — reflecting that the
  // drift point estimate is materially better with more data.
  const N = sampleSize && sampleSize > 0 ? sampleSize : undefined;
  const driftScale = N ? Math.min(0.20, 1 / Math.sqrt(N)) : 0.20; // matches old ±0.5 × 0.20 band
  const volLowerScale = N ? Math.max(0.85, 1 - driftScale * 0.5) : 0.90;
  const volUpperScale = N ? Math.min(1.15, 1 + driftScale * 0.5) : 1.10;
  const rng = (): number => Math.random();
  const ciSamples: Map<number, number[]> = new Map(prices.map(p => [p, []]));

  for (let s = 0; s < monteCarloSamples; s++) {
    // Perturb drift and vol within sampling uncertainty
    const perturbedMu  = mu_n    + (rng() - 0.5) * sigma_n * driftScale;
    const perturbedVol = sigma_n * (volLowerScale + rng() * (volUpperScale - volLowerScale));
    for (const price of prices) {
      const p = studentTSurvival(currentPrice, price, perturbedMu, perturbedVol, nu);
      ciSamples.get(price)!.push(p);
    }
  }

  // Build distribution points
  const rawPoints = prices.map(price => {
    const anchor = findAnchor(price);
    const markovEst = studentTSurvival(currentPrice, price, mu_n, sigma_n, nu);

    let probability: number;
    let source: 'polymarket' | 'markov' | 'blend';

    if (anchor && anchor.trustScore === 'high') {
      // Scale anchor influence by distance from current price
      const anchorW = (1 - mixWeight) * anchor.distanceWeight;
      probability = (1 - anchorW) * markovEst + anchorW * anchor.probability;
      source = anchorW < 0.05 ? 'markov' : anchorW > 0.5 ? 'polymarket' : 'blend';
    } else if (anchor && anchor.trustScore === 'low') {
      // Low-trust anchors: half nominal influence, further scaled by distance
      const anchorW = (1 - mixWeight) * 0.5 * anchor.distanceWeight;
      probability = (1 - anchorW) * markovEst + anchorW * anchor.probability;
      source = 'blend';
    } else {
      probability = markovEst;
      source = 'markov';
    }

    const samples = ciSamples.get(price)!.sort((a, b) => a - b);
    const lo = samples[Math.floor(0.05 * samples.length)];
    const hi = samples[Math.floor(0.95 * samples.length)];

    // Apply CI widening multiplier (used when structural break detected)
    const halfWidth = (hi - lo) / 2;
    const center = (hi + lo) / 2;
    const widenedLo = Math.max(0, center - halfWidth * ciWidthMultiplier);
    const widenedHi = Math.min(1, center + halfWidth * ciWidthMultiplier);

    return { price, probability, lowerBound: widenedLo, upperBound: widenedHi, source };
  });

  // Enforce monotonicity: P(price > X) must be non-increasing in X
  for (let i = rawPoints.length - 2; i >= 0; i--) {
    if (rawPoints[i].probability < rawPoints[i + 1].probability) {
      rawPoints[i].probability = rawPoints[i + 1].probability;
    }
  }

  return rawPoints;
}
