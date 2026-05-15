/**
 * P3a — GARCH(1,1) interim volatility helper.
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §10.3
 *
 * Replaces the constant `dailyVol` in trajectory MC with a one-lag
 * GARCH(1,1) update that captures volatility clustering — the Bloch
 * critique of constant-σ GBM.
 *
 *   h_t  = ω + α · z²_{t-1} · h_{t-1} + β · h_{t-1}
 *   σ_t  = √h_t
 *
 * This is *not* a full MLE estimator. It uses fixed industry priors
 * (α = 0.10, β = 0.85) and matches the unconditional variance to the
 * sample variance — a pragmatic shortcut that captures persistence
 * without an iterative optimiser.
 *
 * For full MLE, use the Python `arch` library via `research/models/garch.py`.
 */

export interface Garch11Params {
  /** Constant term ω > 0. */
  omega: number;
  /** Innovation weight α ∈ [0, 1]. */
  alpha: number;
  /** Persistence weight β ∈ [0, 1]. */
  beta: number;
  /** Initial conditional variance h_0. */
  h0: number;
}

/** Standard equity-style GARCH(1,1) priors (Hansen & Lunde 2005 survey). */
export const GARCH_DEFAULTS = {
  alpha: 0.10,
  beta: 0.85,
} as const;

/**
 * Moment-matching estimator. Holds α and β fixed at industry priors and
 * solves for ω so that the unconditional variance ω / (1 − α − β)
 * equals the sample variance of `returns`.
 *
 * Throws if `returns.length < 5`.
 */
export function fitGarch11(
  returns: number[],
  alpha: number = GARCH_DEFAULTS.alpha,
  beta: number = GARCH_DEFAULTS.beta,
): Garch11Params {
  if (returns.length < 5) {
    throw new Error(`fitGarch11 requires ≥ 5 observations, got ${returns.length}`);
  }
  if (alpha + beta >= 1) {
    throw new Error(`fitGarch11 requires α + β < 1 for stationarity, got ${alpha + beta}`);
  }
  // Sample variance about zero — for daily returns, the mean is ~0 and
  // including it would be noise.
  let sse = 0;
  for (const r of returns) sse += r * r;
  const sampleVar = sse / returns.length;
  const omega = sampleVar * (1 - alpha - beta);
  return { omega, alpha, beta, h0: sampleVar };
}

/** One step of the GARCH(1,1) recursion. */
export function garchStep(prevH: number, prevZ: number, p: Garch11Params): number {
  return p.omega + p.alpha * prevZ * prevZ * prevH + p.beta * prevH;
}

/**
 * Multi-step σ forecast using z = 0 expectation (i.e., E[z²] = 1 substitution).
 *
 * Standard k-step-ahead variance forecast for GARCH(1,1):
 *
 *   E[h_{t+k}] = σ²_∞ + (α + β)^{k-1} · (h_t − σ²_∞)
 *
 * Returns σ_t = √h_t for t = 1, …, horizonDays.
 */
export function garchForecast(p: Garch11Params, horizonDays: number): number[] {
  if (horizonDays <= 0) return [];
  const persistence = p.alpha + p.beta;
  const uncondVar = p.omega / Math.max(1e-12, 1 - persistence);
  const out: number[] = [];
  // Step 1: substitute z² with its expectation (= 1).
  let h = garchStep(p.h0, 1, p);
  out.push(Math.sqrt(Math.max(0, h)));
  for (let t = 1; t < horizonDays; t++) {
    // E[h_{t+1}] = ω + (α + β) · h_t  (with E[z²] = 1)
    h = p.omega + persistence * h;
    out.push(Math.sqrt(Math.max(0, h)));
  }
  // Sanity: long-horizon should reproduce uncondVar within rounding.
  void uncondVar;
  return out;
}
