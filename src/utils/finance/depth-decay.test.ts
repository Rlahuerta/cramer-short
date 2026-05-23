/**
 * W3 Idea 1a — Dubach (2026) depth-decay haircut.
 *
 * Reference: arXiv 2604.24366 §4 — empirical depth-vs-time-to-close slope of
 * roughly +0.55 on log seconds-to-close (i.e. depth shrinks materially as a
 * Polymarket contract approaches resolution). This is independent of the
 * existing information-value boost in `computeExpiryBoost`: that one captures
 * martingale collapse; this one corrects a stale liquidity proxy.
 *
 * The haircut is a multiplicative factor on the *liquidity* component of the
 * quality weight only. It must not go above 1.0 (no inflation) and must
 * never collapse a market entirely (floor 0.5).
 */
import { describe, it, expect } from 'bun:test';
import {
  depthDecayHaircut,
  computeMarketQualityWeight,
  type MarketInput,
} from './ensemble.js';

const baseMarket: MarketInput = {
  question: 'Will BTC be above $100k on Friday?',
  probability: 0.45,
  volume24hUsd: 250_000,
  ageDays: 30,
  signalTier: 'macro',
  deltaYes: 0.06,
  deltaNo: -0.04,
};

describe('depthDecayHaircut', () => {
  it('returns 1.0 when daysToExpiry is undefined or non-finite', () => {
    expect(depthDecayHaircut(undefined)).toBe(1.0);
    expect(depthDecayHaircut(NaN)).toBe(1.0);
    expect(depthDecayHaircut(Number.POSITIVE_INFINITY)).toBe(1.0);
  });

  it('returns 1.0 at the 30-day reference horizon (no penalty)', () => {
    expect(depthDecayHaircut(30)).toBe(1.0);
  });

  it('returns 1.0 for any horizon ≥ 30 days', () => {
    expect(depthDecayHaircut(60)).toBe(1.0);
    expect(depthDecayHaircut(180)).toBe(1.0);
  });

  it('is materially below 1.0 well before resolution (≤ 14 days)', () => {
    const h14 = depthDecayHaircut(14);
    expect(h14).toBeLessThan(0.85);
    expect(h14).toBeGreaterThan(0.5);
  });

  it('is at the 0.5 floor for very near-expiry windows (≤ 7 days)', () => {
    // (7/30)^0.55 ≈ 0.45 → floored at 0.5
    expect(depthDecayHaircut(7)).toBe(0.5);
    expect(depthDecayHaircut(3)).toBe(0.5);
    expect(depthDecayHaircut(1)).toBe(0.5);
  });

  it('is monotonically non-decreasing in daysToExpiry', () => {
    const samples = [0.5, 1, 2, 3, 5, 7, 10, 14, 20, 25, 30, 60];
    const haircuts = samples.map(depthDecayHaircut);
    for (let i = 1; i < haircuts.length; i++) {
      expect(haircuts[i]).toBeGreaterThanOrEqual(haircuts[i - 1]!);
    }
  });

  it('is floored at 0.5 — never collapses a market entirely', () => {
    expect(depthDecayHaircut(0.1)).toBeGreaterThanOrEqual(0.5);
    expect(depthDecayHaircut(0)).toBeGreaterThanOrEqual(0.5);
    expect(depthDecayHaircut(-1)).toBeGreaterThanOrEqual(0.5);
  });

  it('matches the Dubach 0.55 log-slope where the floor does not apply (day 14)', () => {
    // (14/30)^0.55 ≈ 0.665 — above the floor, slope dominates.
    const h14 = depthDecayHaircut(14);
    expect(h14).toBeGreaterThan(0.6);
    expect(h14).toBeLessThan(0.75);
  });
});

describe('computeMarketQualityWeight — wired with depth-decay haircut', () => {
  it('penalises a near-expiry market with the same volume as a 30-day market', () => {
    const wFar = computeMarketQualityWeight({ ...baseMarket, daysToExpiry: 30 });
    const wNear = computeMarketQualityWeight({ ...baseMarket, daysToExpiry: 5 });
    // The information-value boost (computeExpiryBoost) inflates wNear by 1.20,
    // but the depth-decay haircut on the liquidity component must dominate or
    // at least materially offset it. Net effect: near-expiry weight must be
    // strictly below the 30-day weight × the existing 1.20 boost ratio.
    const expectedWithoutHaircut = wFar * 1.20;
    expect(wNear).toBeLessThan(expectedWithoutHaircut);
  });

  it('does not inflate weights above 1.0', () => {
    const w = computeMarketQualityWeight({ ...baseMarket, daysToExpiry: 1, volume24hUsd: 10_000_000 });
    expect(w).toBeLessThanOrEqual(1.0);
  });

  it('preserves backward compat — undefined daysToExpiry produces identical weight to before', () => {
    const w = computeMarketQualityWeight({ ...baseMarket, daysToExpiry: undefined });
    // No expiry boost, no depth haircut → matches the pre-W3 formula.
    // wAge = 1, wLiq = log10(250001)/6 ≈ 0.898, tau = 0.90
    const expected = 1 * (Math.log10(250_001) / 6) * 0.90;
    expect(w).toBeCloseTo(expected, 5);
  });
});
