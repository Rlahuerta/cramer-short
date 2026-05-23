/**
 * Convergence-Time Confidence Signal.
 *
 * Reference: Voigt 2025 (Polymarket Beta-HMM paper).
 *
 * For a Polymarket price series, measure the time-to-first-crossing of the
 * outer bands (p > 1 − ε) or (p < ε). Fast convergence ⇒ market consensus ⇒
 * boost forecast confidence. Slow / no convergence ⇒ persistent uncertainty
 * ⇒ damp confidence. Voigt also documents a YES/NO asymmetry: NO contracts
 * converge ~40% faster, so the effective horizon for NO is days / 0.6.
 *
 * Output is a multiplicative factor in [0.85, 1.20] applied to existing
 * confidence bands.
 */

export interface ConvergenceResult {
  converged: boolean;
  /** Days from the first observation to the first crossing; null if never. */
  daysToConverge: number | null;
  /** Which band was crossed: 'yes' (p > 1-ε) or 'no' (p < ε). */
  direction: 'yes' | 'no' | null;
}

/** Voigt 2025: NO converges ~40% faster than YES on Polymarket. */
const NO_SPEEDUP = 0.6;
const FAST_THRESHOLD_DAYS = 7;
const SLOW_THRESHOLD_DAYS = 30;
const FAST_BOOST = 0.15;
const SLOW_DAMP = 0.10;

export function convergenceTime(prices: readonly number[], epsilon = 0.05): ConvergenceResult {
  if (prices.length === 0) return { converged: false, daysToConverge: null, direction: null };
  const upper = 1 - epsilon;
  const lower = epsilon;
  for (let i = 0; i < prices.length; i++) {
    const p = prices[i] ?? 0.5;
    if (p > upper) return { converged: true, daysToConverge: i, direction: 'yes' };
    if (p < lower) return { converged: true, daysToConverge: i, direction: 'no' };
  }
  return { converged: false, daysToConverge: null, direction: null };
}

/**
 * Map a {@link ConvergenceResult} to a multiplicative confidence factor.
 *
 *   - Not converged       ⇒ 1.0 (no adjustment, just persistent uncertainty)
 *   - daysToConverge ≤ 7  ⇒ linearly decaying boost, max +15% at 1d
 *   - daysToConverge ≥ 30 ⇒ down to −10% damp (saturates at 30d+)
 *   - Linear interpolation in (7, 30), continuing from the 7d fast-window factor
 *   - NO direction is normalised by NO_SPEEDUP=0.6 before the lookup
 */
export function convergenceTimeFactor(r: ConvergenceResult): number {
  if (!r.converged || r.daysToConverge === null) return 1.0;
  const days = r.direction === 'no' ? r.daysToConverge * NO_SPEEDUP : r.daysToConverge;
  const fastWindowFloor = 1 + FAST_BOOST / FAST_THRESHOLD_DAYS;
  if (days <= 1) return 1 + FAST_BOOST;
  if (days <= FAST_THRESHOLD_DAYS) {
    const tFast = (days - 1) / Math.max(1, FAST_THRESHOLD_DAYS - 1);
    return (1 + FAST_BOOST) + tFast * (fastWindowFloor - (1 + FAST_BOOST));
  }
  if (days >= SLOW_THRESHOLD_DAYS) return 1 - SLOW_DAMP;
  // Linear interp between the day-7 fast-window floor and the 30d damped value.
  const t = (days - FAST_THRESHOLD_DAYS) / (SLOW_THRESHOLD_DAYS - FAST_THRESHOLD_DAYS);
  return fastWindowFloor + t * ((1 - SLOW_DAMP) - fastWindowFloor);
}
