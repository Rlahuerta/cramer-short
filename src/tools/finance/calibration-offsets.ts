/**
 * Domain × horizon Polymarket recalibration.
 *
 * Reference: Bayesian hierarchical decomposition of prediction-market
 * miscalibration explains 87.3% of variance through three terms:
 *   - α_i  (domain intercept)
 *   - β_i · log(T)  (horizon × domain interaction)
 *   - γ    (trade-size effect — not modelled here)
 *
 * Magnitudes per the meta-analysis: politics is most miscalibrated
 * (α ≈ +0.15 logit, underconfident on YES); sports is well-calibrated;
 * crypto and macro fall in between.
 *
 * Apply this **upstream** of {@link transformQToP} so the pipeline is:
 *   raw polymarket price
 *     → recalibratePolymarketPrice (domain × horizon)
 *     → transformQToP (Sharpe-driven Girsanov)
 *     → fitLognormalFromStrikes / nudgeTransitionMatrix
 */

import { normCDF, normPPF } from '@/utils/stats.js';

export type Domain = 'politics' | 'sports' | 'crypto' | 'macro' | 'unknown';

export interface CalibrationOffset {
  /** Logit-space intercept (additive shift on Φ⁻¹(q)) */
  alpha: number;
  /** Slope coefficient on log1p(daysToExpiry); 0 ⇒ horizon-independent */
  betaPerLogT: number;
}

export const DOMAIN_OFFSETS: Record<Domain, CalibrationOffset> = {
  politics: { alpha: 0.15, betaPerLogT: 0.05 },
  sports:   { alpha: 0.00, betaPerLogT: 0.00 },
  crypto:   { alpha: 0.05, betaPerLogT: 0.03 },
  macro:    { alpha: 0.02, betaPerLogT: 0.04 },
  unknown:  { alpha: 0.00, betaPerLogT: 0.00 },
};

/**
 * Recalibrate a raw Polymarket YES probability for a given domain and horizon.
 *
 *   z_recal = (1 + β · log1p(T_days)) · Φ⁻¹(q) + α
 *   p_recal = Φ(z_recal)
 *
 * Boundary inputs (q ≤ 0 or q ≥ 1) pass through unchanged.
 */
export function recalibratePolymarketPrice(
  qProb: number,
  domain: Domain,
  daysToExpiry: number,
): number {
  if (qProb <= 0) return 0;
  if (qProb >= 1) return 1;
  const offset = DOMAIN_OFFSETS[domain] ?? DOMAIN_OFFSETS.unknown;
  if (offset.alpha === 0 && offset.betaPerLogT === 0) return qProb;
  const days = Math.max(daysToExpiry, 1);
  const slope = 1 + offset.betaPerLogT * Math.log1p(days);
  const z = normPPF(qProb);
  const zRecal = slope * z + offset.alpha;
  return normCDF(zRecal);
}
