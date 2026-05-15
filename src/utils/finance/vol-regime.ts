/**
 * P3b — VIX-based volatility regime classifier.
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §9
 *
 * Provides a synchronous, no-fitting alternative to running a secondary
 * HMM on (return, IV-change) joint dynamics. Uses VIX level as a regime
 * proxy and applies a per-step leverage-effect vol multiplier inside
 * the trajectory MC.
 *
 * Asset gating (§9.3): leverage effect applies only to **equity** and
 * **gold** — not to crypto (positive return-vol correlation in bull
 * markets) or commodities (regime-dependent).
 */

export type VolRegime = 'sticky_strike' | 'transitional' | 'sticky_implied_tree';

const _VIX_TRANSITIONAL = 15;
const _VIX_FEAR = 25;

/**
 * Classify the current vol regime from a VIX level.
 *
 *  - VIX < 15  → 'sticky_strike'
 *  - 15 ≤ VIX < 25 → 'transitional'
 *  - VIX ≥ 25  → 'sticky_implied_tree' (leverage-effect regime)
 */
export function getVolatilityRegime(vix: number): VolRegime {
  if (!Number.isFinite(vix) || vix < _VIX_TRANSITIONAL) return 'sticky_strike';
  if (vix < _VIX_FEAR) return 'transitional';
  return 'sticky_implied_tree';
}

/**
 * Per-step vol multiplier for the leverage effect in the trajectory MC.
 *
 *   regime=sticky_implied_tree, z<0 (down draw) → ×1.4 (vol amplifies)
 *   regime=sticky_implied_tree, z>0 (up draw)   → ×0.8 (vol mutes)
 *   else                                       → 1.0 (neutral)
 *
 * Gated on `assetClass ∈ {equity, gold}`. Other asset classes are
 * empirically not consistent with the leverage-effect pattern (§9.3).
 */
export function leverageVolMultiplier(
  regime: VolRegime,
  z: number,
  assetClass: string,
): number {
  if (regime !== 'sticky_implied_tree') return 1;
  if (assetClass !== 'equity' && assetClass !== 'gold') return 1;
  if (!Number.isFinite(z) || z === 0) return 1;
  return z < 0 ? 1.4 : 0.8;
}
