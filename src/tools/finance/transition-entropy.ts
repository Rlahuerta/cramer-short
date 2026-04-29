/**
 * R5 Idea #14 — Markov transition-entropy CI modulator.
 *
 * Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #14),
 * arXiv:2511.05621 (Chen et al. 2025) — entropy production rates as a
 * leading indicator of regime instability.
 *
 * Hypothesis: when the row-entropy of the empirical Markov transition
 * matrix spikes (high uncertainty about *which regime is next*), the
 * forecast CI should widen.  When the matrix becomes near-deterministic
 * (low entropy), the CI can tighten.
 *
 * We compute the stationary-weighted average row entropy:
 *
 *   H = -Σ_i π_i Σ_j P_ij log P_ij        // nats
 *   H_max = log(K)                         // K = num states
 *   H_norm = H / H_max ∈ [0, 1]
 *
 * Then the caller keeps a rolling mean/std over the last `windowSize`
 * H_norm values, computes z = (H_norm − μ) / σ, and applies:
 *
 *   ciScale = clamp(1 + κ × (-z), 0.7, 1.4)
 *
 * (negative z → high recent uncertainty → wider CI).  The caller multiplies
 * the half-width of the predictive interval by `ciScale`.
 *
 * Pure functions only — no I/O, no global state.  Caller manages the
 * rolling window (typically inside `walkForward`).
 */

export interface TransitionEntropyResult {
  /** Stationary-weighted row entropy in nats. */
  entropyNats: number;
  /** H normalized to [0, 1] by log(K). */
  entropyNorm: number;
  /** Number of states K. */
  K: number;
}

/**
 * Compute a stationary distribution by power-iteration on `P^T`.
 * Falls back to uniform if `P` is degenerate.
 */
export function approximateStationary(P: readonly (readonly number[])[]): number[] {
  const K = P.length;
  if (K === 0) return [];
  let pi = new Array(K).fill(1 / K);
  for (let iter = 0; iter < 100; iter++) {
    const next = new Array(K).fill(0);
    for (let j = 0; j < K; j++) {
      let s = 0;
      for (let i = 0; i < K; i++) s += pi[i] * (P[i]?.[j] ?? 0);
      next[j] = s;
    }
    let total = 0;
    for (const x of next) total += x;
    if (!(total > 0)) return new Array(K).fill(1 / K);
    for (let j = 0; j < K; j++) next[j] /= total;
    let delta = 0;
    for (let j = 0; j < K; j++) delta += Math.abs(next[j] - pi[j]);
    pi = next;
    if (delta < 1e-9) break;
  }
  return pi;
}

export function computeTransitionEntropy(
  P: readonly (readonly number[])[],
): TransitionEntropyResult {
  const K = P.length;
  if (K === 0) return { entropyNats: 0, entropyNorm: 0, K: 0 };
  const pi = approximateStationary(P);
  let H = 0;
  for (let i = 0; i < K; i++) {
    const row = P[i] ?? [];
    let rowH = 0;
    for (let j = 0; j < K; j++) {
      const p = row[j] ?? 0;
      if (p > 0) rowH -= p * Math.log(p);
    }
    H += pi[i] * rowH;
  }
  const Hmax = Math.log(Math.max(2, K));
  const norm = Hmax > 0 ? H / Hmax : 0;
  return { entropyNats: H, entropyNorm: Math.max(0, Math.min(1, norm)), K };
}

/**
 * Online rolling z-score state — keep one instance per backtest fold.
 * Reservoir of last `windowSize` H_norm values.
 */
export class EntropyZScoreTracker {
  private buf: number[] = [];
  constructor(private readonly windowSize: number = 60) {
    if (windowSize < 5) throw new Error('EntropyZScoreTracker windowSize must be ≥ 5');
  }

  push(value: number): void {
    this.buf.push(value);
    if (this.buf.length > this.windowSize) this.buf.shift();
  }

  /** Returns null until at least 5 values observed (insufficient for std). */
  zScore(value: number): number | null {
    if (this.buf.length < 5) return null;
    let mean = 0;
    for (const v of this.buf) mean += v;
    mean /= this.buf.length;
    let sse = 0;
    for (const v of this.buf) sse += (v - mean) ** 2;
    const std = Math.sqrt(sse / this.buf.length);
    if (!(std > 1e-9)) return 0;
    return (value - mean) / std;
  }

  size(): number { return this.buf.length; }
}

/**
 * Convert a transition-entropy z-score to a CI width scalar.
 *
 * @param zNorm   — current entropy z-score (positive = unusually uncertain)
 * @param kappa   — sensitivity (default 0.15)
 * @param bounds  — clamp range (default [0.7, 1.4])
 *
 * Behaviour:
 *   z = +2 (very high uncertainty)  ⇒ ciScale = 1 + 0.15·(-(-2)) ⇒ 1.30
 *
 *   Wait — actually reverse: high z means *more* uncertainty, so we want
 *   *wider* CI ⇒ ciScale > 1.  The formula is `1 + κ·z` (positive z widens).
 *
 * Wait the spec says `clamp(1 + κ × (-zH), 0.7, 1.4)` — but the original
 * R5 doc derivation has zH defined such that *negative* zH means high
 * uncertainty (because they took log-likelihood, not entropy).  We use
 * the simpler convention: z_entropy positive ⇒ high uncertainty ⇒ widen.
 * So our formula is: `ciScale = clamp(1 + κ × z, 0.7, 1.4)`.
 */
export function entropyZToCiScale(
  zNorm: number,
  kappa: number = 0.15,
  bounds: [number, number] = [0.7, 1.4],
): number {
  const raw = 1 + kappa * zNorm;
  return Math.max(bounds[0], Math.min(bounds[1], raw));
}
