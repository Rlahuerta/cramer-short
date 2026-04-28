/**
 * Phase 1 (P1) ensemble improvements:
 *   P1a — adjustYesBiasV2: U-shaped longshot-bias correction
 *   P1b — daysToExpiry + computeExpiryBoost quality factor
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §2, §3
 */

import { describe, it, expect } from 'bun:test';
import {
  adjustYesBiasV2,
  computeExpiryBoost,
  computeMarketQualityWeight,
  type MarketInput,
} from './ensemble.js';

// ---------------------------------------------------------------------------
// P1a — adjustYesBiasV2 (longshot regime)
// ---------------------------------------------------------------------------

describe('adjustYesBiasV2 — extreme longshot regime', () => {
  it('p=0.03 (deep longshot) → 30% multiplicative discount', () => {
    expect(adjustYesBiasV2(0.03)).toBeCloseTo(0.03 * 0.7, 6);
  });

  it('p=0.04 (still deep longshot) → 30% discount', () => {
    expect(adjustYesBiasV2(0.04)).toBeCloseTo(0.04 * 0.7, 6);
  });
});

describe('adjustYesBiasV2 — moderate longshot regime', () => {
  it('p=0.05 (boundary) → 30% discount (continuous from below)', () => {
    expect(adjustYesBiasV2(0.05)).toBeCloseTo(0.05 * 0.7, 6);
  });

  it('p=0.10 (mid longshot) → ~17.5% discount (linear interpolation)', () => {
    // t = (0.10-0.05)/0.10 = 0.5; mult = 0.70 + 0.5*(0.95-0.70) = 0.825
    expect(adjustYesBiasV2(0.10)).toBeCloseTo(0.10 * 0.825, 6);
  });

  it('p=0.15 (boundary) → 5% multiplicative discount', () => {
    // t = 1.0; mult = 0.95
    expect(adjustYesBiasV2(0.15)).toBeCloseTo(0.15 * 0.95, 6);
  });
});

describe('adjustYesBiasV2 — mid range (matches legacy adjustYesBias)', () => {
  it('p=0.30 → unchanged (below 0.5)', () => {
    expect(adjustYesBiasV2(0.30)).toBeCloseTo(0.30, 6);
  });

  it('p=0.55 → -3.5pp legacy shift', () => {
    expect(adjustYesBiasV2(0.55)).toBeCloseTo(0.515, 6);
  });

  it('p=0.70 → -3.5pp legacy shift', () => {
    expect(adjustYesBiasV2(0.70)).toBeCloseTo(0.665, 6);
  });

  it('p=0.85 → boundary into favourite regime, -3.5pp shift', () => {
    expect(adjustYesBiasV2(0.85)).toBeCloseTo(0.815, 6);
  });
});

describe('adjustYesBiasV2 — strong favourite regime', () => {
  it('p=0.90 → -2.5pp shift', () => {
    expect(adjustYesBiasV2(0.90)).toBeCloseTo(0.875, 6);
  });

  it('p=0.92 → -2.5pp shift', () => {
    expect(adjustYesBiasV2(0.92)).toBeCloseTo(0.895, 6);
  });
});

describe('adjustYesBiasV2 — clamping', () => {
  it('p=0 → clamped to 0.001', () => {
    expect(adjustYesBiasV2(0)).toBeCloseTo(0.001, 6);
  });

  it('p=1 → clamped to 0.999', () => {
    expect(adjustYesBiasV2(1)).toBeCloseTo(0.999, 6);
  });

  it('p=-0.5 → clamped to 0.001', () => {
    expect(adjustYesBiasV2(-0.5)).toBeCloseTo(0.001, 6);
  });

  it('p=1.5 → clamped to 0.999', () => {
    expect(adjustYesBiasV2(1.5)).toBeCloseTo(0.999, 6);
  });
});

describe('adjustYesBiasV2 — true improvement vs legacy adjustYesBias', () => {
  it('longshot p=0.05: V2 discounts (3.5%), legacy leaves unchanged (5%)', () => {
    const v2 = adjustYesBiasV2(0.05);
    expect(v2).toBeLessThan(0.05);
    expect(v2).toBeCloseTo(0.035, 6);
  });

  it('legacy treats p=0.40 unchanged; V2 also unchanged in mid range', () => {
    expect(adjustYesBiasV2(0.40)).toBeCloseTo(0.40, 6);
  });
});

// ---------------------------------------------------------------------------
// P1b — computeExpiryBoost
// ---------------------------------------------------------------------------

describe('computeExpiryBoost — schedule', () => {
  it('1 day to expiry → boost 1.50 (high certainty)', () => {
    expect(computeExpiryBoost(1)).toBeCloseTo(1.50, 6);
  });

  it('0 days (expiring today) → boost 1.50', () => {
    expect(computeExpiryBoost(0)).toBeCloseTo(1.50, 6);
  });

  it('5 days → boost 1.20', () => {
    expect(computeExpiryBoost(5)).toBeCloseTo(1.20, 6);
  });

  it('7 days (boundary) → boost 1.20', () => {
    expect(computeExpiryBoost(7)).toBeCloseTo(1.20, 6);
  });

  it('15 days → boost 1.00 (neutral)', () => {
    expect(computeExpiryBoost(15)).toBeCloseTo(1.00, 6);
  });

  it('30 days (boundary) → boost 1.00', () => {
    expect(computeExpiryBoost(30)).toBeCloseTo(1.00, 6);
  });

  it('60 days → boost 0.85', () => {
    expect(computeExpiryBoost(60)).toBeCloseTo(0.85, 6);
  });

  it('90 days (boundary) → boost 0.85', () => {
    expect(computeExpiryBoost(90)).toBeCloseTo(0.85, 6);
  });

  it('180 days (far-dated) → boost 0.70', () => {
    expect(computeExpiryBoost(180)).toBeCloseTo(0.70, 6);
  });
});

// ---------------------------------------------------------------------------
// P1b — computeMarketQualityWeight integration
// ---------------------------------------------------------------------------

describe('computeMarketQualityWeight — daysToExpiry integration', () => {
  // Use lower volume so wLiq < 1 — avoids clamping when boost is applied.
  const baseInput: MarketInput = {
    question: 'Q',
    probability: 0.5,
    volume24hUsd: 10_000,
    ageDays: 30,
    signalTier: 'geopolitical',
    deltaYes: 0.05,
    deltaNo: -0.03,
  };

  it('omitting daysToExpiry → no change vs legacy behaviour', () => {
    const wOld = computeMarketQualityWeight(baseInput);
    const wExplicit = computeMarketQualityWeight({ ...baseInput, daysToExpiry: undefined });
    expect(wExplicit).toBeCloseTo(wOld, 6);
  });

  it('daysToExpiry=1 → information-value boost (1.5×) net of W3 depth-decay haircut (0.5)', () => {
    // W3 Idea 1a (Dubach 2026): the 1.5× near-expiry information-value boost
    // is now offset by a 0.5 haircut on the liquidity component when depth is
    // expected to have evaporated. Net effect: near-expiry quality drops to
    // ~0.75× of the 30-day reference. Intentional behaviour change — a
    // public-feed Polymarket signal at <7d to resolution is *less* trustworthy,
    // not more, despite the martingale collapse, because liquidity disappears.
    const wNeutral = computeMarketQualityWeight({ ...baseInput, daysToExpiry: 30 });
    const wNear = computeMarketQualityWeight({ ...baseInput, daysToExpiry: 1 });
    expect(wNear).toBeLessThan(wNeutral);
    expect(wNear / wNeutral).toBeCloseTo(0.75, 4);
  });

  it('daysToExpiry=180 → quality discounted by 0.7×', () => {
    const wNeutral = computeMarketQualityWeight({ ...baseInput, daysToExpiry: 30 });
    const wFar = computeMarketQualityWeight({ ...baseInput, daysToExpiry: 180 });
    expect(wFar).toBeLessThan(wNeutral);
    expect(wFar / wNeutral).toBeCloseTo(0.7, 4);
  });

  it('weight remains clamped to [0, 1] even with boost', () => {
    const wMaxBoost = computeMarketQualityWeight({
      ...baseInput,
      daysToExpiry: 1,
      signalTier: 'macro', // τ=0.90, near 1.0 with boost
    });
    expect(wMaxBoost).toBeLessThanOrEqual(1);
    expect(wMaxBoost).toBeGreaterThanOrEqual(0);
  });
});
