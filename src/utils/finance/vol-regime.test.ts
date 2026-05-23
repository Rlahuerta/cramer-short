/**
 * P3b — VIX-based volatility regime classifier (TDD).
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §9.2
 *
 *   VIX < 15  → 'sticky_strike'         (moderate vol, no leverage effect)
 *   15 ≤ VIX < 25 → 'transitional'      (mild leverage effect)
 *   VIX ≥ 25  → 'sticky_implied_tree'  (strong leverage, vol spikes on down)
 *
 * Pure helpers under test:
 *   - getVolatilityRegime(vix)
 *   - leverageVolMultiplier(regime, z, assetClass)
 *
 * Asset gating: leverage effect applies only to 'equity' / 'gold'.
 */

import { describe, expect, it } from 'bun:test';
import { getVolatilityRegime, leverageVolMultiplier, type VolRegime } from './vol-regime.js';

describe('getVolatilityRegime', () => {
  it('classifies VIX < 15 as sticky_strike', () => {
    expect(getVolatilityRegime(10)).toBe('sticky_strike');
    expect(getVolatilityRegime(14.99)).toBe('sticky_strike');
  });

  it('classifies 15 ≤ VIX < 25 as transitional', () => {
    expect(getVolatilityRegime(15)).toBe('transitional');
    expect(getVolatilityRegime(20)).toBe('transitional');
    expect(getVolatilityRegime(24.99)).toBe('transitional');
  });

  it('classifies VIX ≥ 25 as sticky_implied_tree', () => {
    expect(getVolatilityRegime(25)).toBe('sticky_implied_tree');
    expect(getVolatilityRegime(40)).toBe('sticky_implied_tree');
    expect(getVolatilityRegime(80)).toBe('sticky_implied_tree');
  });

  it('handles boundary edge cases', () => {
    expect(getVolatilityRegime(0)).toBe('sticky_strike');
    expect(getVolatilityRegime(-5)).toBe('sticky_strike');
  });
});

describe('leverageVolMultiplier', () => {
  it('amplifies vol on down moves in sticky_implied_tree (equity)', () => {
    // z < 0 (down move) ⇒ ×1.4
    expect(leverageVolMultiplier('sticky_implied_tree', -1.5, 'equity')).toBeCloseTo(1.4, 6);
  });

  it('mutes vol on up moves in sticky_implied_tree (equity)', () => {
    // z > 0 ⇒ ×0.8
    expect(leverageVolMultiplier('sticky_implied_tree', 1.5, 'equity')).toBeCloseTo(0.8, 6);
  });

  it('returns 1.0 in sticky_strike regime', () => {
    expect(leverageVolMultiplier('sticky_strike', -2, 'equity')).toBe(1);
    expect(leverageVolMultiplier('sticky_strike', 2, 'equity')).toBe(1);
  });

  it('returns 1.0 in transitional regime (mild — left to caller)', () => {
    // Spec says "mild" — keep multiplier neutral; callers can apply softer dampening
    expect(leverageVolMultiplier('transitional', -2, 'equity')).toBe(1);
  });

  it('returns 1.0 for non-equity/non-gold assets even in fear regime', () => {
    expect(leverageVolMultiplier('sticky_implied_tree', -2, 'crypto')).toBe(1);
    expect(leverageVolMultiplier('sticky_implied_tree', 2, 'crypto')).toBe(1);
    expect(leverageVolMultiplier('sticky_implied_tree', -2, 'commodity')).toBe(1);
  });

  it('applies leverage effect for gold (mild)', () => {
    // Gold should get the same shape as equity (spec §9.3)
    expect(leverageVolMultiplier('sticky_implied_tree', -1, 'gold')).toBeCloseTo(1.4, 6);
    expect(leverageVolMultiplier('sticky_implied_tree', 1, 'gold')).toBeCloseTo(0.8, 6);
  });

  it('z = 0 returns neutral 1.0', () => {
    expect(leverageVolMultiplier('sticky_implied_tree', 0, 'equity')).toBe(1);
  });
});
