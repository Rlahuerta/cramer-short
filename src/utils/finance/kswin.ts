/**
 * R4 Phase C — KSWIN-style variance-aware drift detector.
 *
 * Kolmogorov–Smirnov Windowing (Raab, Heusinger & Schleif 2020, ECML-PKDD).
 * Distinct from ADWIN by being **distribution-aware** rather than mean-aware:
 * KS is sensitive to changes in *spread* (variance), so it fires on regime
 * vol-shifts that ADWIN — which tracks the running mean — typically misses
 * (e.g., BTC quiet-then-bursty patterns).
 *
 * Algorithm:
 *   1. Walk a sliding pair of windows (`reference`, `recent`) from the most
 *      recent observation backward through history.
 *   2. At each split point, compute the two-sample KS statistic
 *        D = max_x |F_ref(x) − F_recent(x)|
 *      where F_⋅ are empirical CDFs.
 *   3. Reject H₀ (no drift) when
 *        D > c(α) · √((n + m) / (n · m))     (Smirnov critical value)
 *      with the standard table value c(0.005) ≈ 1.731.
 *   4. The earliest rejection point (closest to "now") becomes the trim
 *      boundary — older observations are dropped.
 *
 * Operates on `|log-return|` by default (variance proxy).  Caller is free to
 * pass any 1-D series.
 */

/** One-sided critical multipliers for the two-sample KS statistic. */
const KS_CRITICAL: Readonly<Record<string, number>> = {
  '0.10': 1.224,
  '0.05': 1.358,
  '0.025': 1.480,
  '0.01': 1.628,
  '0.005': 1.731,
  '0.001': 1.949,
};

export interface KswinOptions {
  /** Reference (older) window size.  Default 50. */
  referenceWindow?: number;
  /** Recent window size.  Default 50. */
  recentWindow?: number;
  /** Significance level α.  Default 0.005 (matches paper). */
  alpha?: number;
  /** Minimum number of trailing samples to retain.  Default 60. */
  minKeep?: number;
}

export interface KswinResult {
  /** Was a drift point detected? */
  drift: boolean;
  /** Largest D statistic observed (across all candidate split points). */
  maxD: number;
  /** Critical D used for the chosen α. */
  criticalD: number;
  /** Number of trailing samples to keep (when `drift`, ≥ minKeep). */
  keepCount: number;
}

/** Two-sample KS statistic on already-finalised arrays (sorts internally). */
export function kolmogorovSmirnovD(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const sa = [...a].sort((x, y) => x - y);
  const sb = [...b].sort((x, y) => x - y);
  let i = 0;
  let j = 0;
  let d = 0;
  while (i < sa.length && j < sb.length) {
    const x = Math.min(sa[i], sb[j]);
    while (i < sa.length && sa[i] <= x) i++;
    while (j < sb.length && sb[j] <= x) j++;
    const cdfA = i / sa.length;
    const cdfB = j / sb.length;
    const gap = Math.abs(cdfA - cdfB);
    if (gap > d) d = gap;
  }
  return d;
}

/**
 * Detects the most recent drift point in `series` (newer observations on the
 * right).  When drift is detected, returns the suffix length to keep —
 * everything older than the detected change-point is considered non-stationary.
 */
export function detectKswinDrift(
  series: readonly number[],
  opts: KswinOptions = {},
): KswinResult {
  const r = opts.referenceWindow ?? 50;
  const w = opts.recentWindow ?? 50;
  const alpha = opts.alpha ?? 0.005;
  const minKeep = opts.minKeep ?? 60;
  const n = series.length;

  const cAlpha = KS_CRITICAL[alpha.toString()] ?? KS_CRITICAL['0.005'];
  const criticalD = cAlpha * Math.sqrt((r + w) / (r * w));

  if (n < r + w) {
    return { drift: false, maxD: 0, criticalD, keepCount: n };
  }

  // Walk split points from "newest" backward: at each split s, compare
  //   reference = series[s - r .. s)         (older r samples)
  //   recent    = series[s .. s + w)         (newer w samples)
  // Slide by 1 each step.  We want the *latest* split (largest s) at which
  // drift is detected, because everything older than that split is unsafe.
  let maxD = 0;
  let driftSplit = -1;
  for (let s = n - w; s >= r; s--) {
    const ref = series.slice(s - r, s);
    const rec = series.slice(s, s + w);
    const d = kolmogorovSmirnovD(ref, rec);
    if (d > maxD) maxD = d;
    if (d > criticalD && driftSplit < 0) {
      driftSplit = s;
      break; // first (latest) detected drift wins.
    }
  }

  if (driftSplit < 0) {
    return { drift: false, maxD, criticalD, keepCount: n };
  }

  // Keep everything from the drift split onward — it represents the recent
  // stationary regime.  Enforce the minKeep floor.
  const keep = Math.max(minKeep, n - driftSplit);
  return { drift: true, maxD, criticalD, keepCount: Math.min(n, keep) };
}
