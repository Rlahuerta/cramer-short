/**
 * Mirrors `research/models/hawkes.py`.
 *
 * Hawkes self-exciting point process.
 *
 * λ(t) = μ + Σ_{t_i < t} α · exp(−β · (t − t_i))
 *
 * Stability requires α/β < 1 (branching ratio).
 *
 * Reference: Cestari et al. 2023 (arXiv 2312.16190); Ogata 1981 thinning.
 */

export interface HawkesParams {
  mu: number;
  alpha: number;
  beta: number;
}

export interface HawkesFit extends HawkesParams {
  logLikelihood: number;
  isStable: boolean;
}

export class HawkesIntensity {
  readonly mu: number;
  readonly alpha: number;
  readonly beta: number;

  constructor(params: HawkesParams) {
    if (!(params.mu >= 0) || !Number.isFinite(params.mu))
      throw new Error("Hawkes: mu must be >= 0");
    if (params.alpha < 0) throw new Error("Hawkes: alpha must be >= 0");
    if (!(params.beta > 0)) throw new Error("Hawkes: beta must be > 0");
    this.mu = params.mu;
    this.alpha = params.alpha;
    this.beta = params.beta;
  }

  /**
   * Right-continuous intensity λ(t+) using ti <= t — i.e. a jump at exactly t
   * contributes its full excitation α to the intensity.  This matches the
   * standard convention "intensity right after the jump".
   */
  intensity(t: number, history: readonly number[]): number {
    let sum = 0;
    for (const ti of history) {
      if (ti <= t) sum += Math.exp(-this.beta * (t - ti));
    }
    return this.mu + this.alpha * sum;
  }

  branchingRatio(): number {
    return this.alpha / this.beta;
  }

  isStable(): boolean {
    return this.branchingRatio() < 1.0;
  }

  /**
   * Log-likelihood of an event sequence on [0, T]:
   *   logL = Σ log λ(t_i) − ∫₀ᵀ λ(s) ds
   *
   * The integral admits a closed form for the exponential kernel:
   *   ∫₀ᵀ λ(s) ds = μ·T + Σ_i (α/β) · (1 − e^(−β·(T−t_i)))
   */
  logLikelihood(events: readonly number[], horizon: number): number {
    if (horizon <= 0) return 0;
    let logSum = 0;
    // Accumulate Σ_{t_j < t_i} e^(-β(t_i - t_j)) recursively; classic O(N) trick.
    let recursive = 0;
    let prev = 0;
    for (let i = 0; i < events.length; i++) {
      const ti = events[i];
      if (i > 0) recursive = (recursive + 1) * Math.exp(-this.beta * (ti - prev));
      const lam = this.mu + this.alpha * recursive;
      if (lam <= 0) return -Infinity;
      logSum += Math.log(lam);
      prev = ti;
    }
    let compensator = this.mu * horizon;
    const ratio = this.alpha / this.beta;
    for (const ti of events) {
      if (ti < horizon) compensator += ratio * (1 - Math.exp(-this.beta * (horizon - ti)));
    }
    return logSum - compensator;
  }
}

/**
 * Ogata (1981) thinning algorithm to simulate a Hawkes process on (0, T].
 * `rng` returns uniform [0, 1).
 */
export function simulateHawkes(
  params: HawkesParams,
  T: number,
  rng: () => number,
): number[] {
  const h = new HawkesIntensity(params);
  const events: number[] = [];
  let t = 0;
  while (t < T) {
    const lamBar = h.intensity(t, events) + 1e-12; // upper bound at t
    const u = rng();
    const w = -Math.log(u) / lamBar;
    t = t + w;
    if (t >= T) break;
    const d = rng();
    const lamT = h.intensity(t, events);
    if (d * lamBar <= lamT) events.push(t);
  }
  return events;
}

interface FitOptions {
  initialMu?: number;
  initialAlpha?: number;
  initialBeta?: number;
  maxIter?: number;
  tol?: number;
}

/**
 * Maximum-likelihood fit via coordinate-wise grid + golden-section refinement.
 *
 * Not as efficient as L-BFGS but stable, deterministic, and dependency-free
 * — adequate for the dataset sizes seen in our backtests (≤ a few thousand
 * events).
 */
export function fitHawkesMLE(
  events: readonly number[],
  horizon: number,
  opts: FitOptions = {},
): HawkesFit {
  const initialMu = opts.initialMu ?? Math.max(1e-3, events.length / Math.max(horizon, 1e-6));
  const initialAlpha = opts.initialAlpha ?? 0.1;
  const initialBeta = opts.initialBeta ?? 1.0;
  const maxIter = opts.maxIter ?? 50;
  const tol = opts.tol ?? 1e-5;

  let mu = initialMu;
  let alpha = initialAlpha;
  let beta = initialBeta;

  const ll = (mu_: number, alpha_: number, beta_: number): number => {
    if (mu_ <= 0 || alpha_ < 0 || beta_ <= 0) return -Infinity;
    if (alpha_ / beta_ >= 0.999) return -Infinity; // stability constraint
    return new HawkesIntensity({ mu: mu_, alpha: alpha_, beta: beta_ }).logLikelihood(
      events,
      horizon,
    );
  };

  // Golden-section search on a single coordinate.
  const golden = (
    lo: number,
    hi: number,
    fn: (x: number) => number,
    iters = 40,
  ): number => {
    const phi = (Math.sqrt(5) - 1) / 2;
    let a = lo;
    let b = hi;
    let c = b - phi * (b - a);
    let d = a + phi * (b - a);
    for (let i = 0; i < iters; i++) {
      if (fn(c) > fn(d)) b = d;
      else a = c;
      c = b - phi * (b - a);
      d = a + phi * (b - a);
    }
    return (a + b) / 2;
  };

  let prevLL = ll(mu, alpha, beta);
  for (let iter = 0; iter < maxIter; iter++) {
    mu = golden(1e-6, Math.max(initialMu * 10, events.length / horizon + 1), (m) =>
      ll(m, alpha, beta),
    );
    beta = golden(1e-3, 50, (b) => ll(mu, alpha, b));
    alpha = golden(0, 0.999 * beta, (a) => ll(mu, a, beta));
    const nextLL = ll(mu, alpha, beta);
    if (Math.abs(nextLL - prevLL) < tol) break;
    prevLL = nextLL;
  }

  const final = new HawkesIntensity({ mu, alpha, beta });
  return {
    mu,
    alpha,
    beta,
    logLikelihood: final.logLikelihood(events, horizon),
    isStable: final.isStable(),
  };
}
