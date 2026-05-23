import { describe, expect, test } from 'bun:test';
import {
  DOMAIN_OFFSETS,
  recalibratePolymarketPrice,
  type Domain,
} from './calibration-offsets.js';

describe('recalibratePolymarketPrice', () => {
  test('identity on unknown domain', () => {
    for (const q of [0.05, 0.2, 0.5, 0.7, 0.95]) {
      expect(recalibratePolymarketPrice(q, 'unknown', 30)).toBeCloseTo(q, 12);
    }
  });

  test('identity on sports domain (alpha=0, beta=0)', () => {
    expect(recalibratePolymarketPrice(0.4, 'sports', 30)).toBeCloseTo(0.4, 12);
  });

  test('boundary inputs (0 / 1) pass through', () => {
    expect(recalibratePolymarketPrice(0, 'politics', 30)).toBe(0);
    expect(recalibratePolymarketPrice(1, 'politics', 30)).toBe(1);
  });

  test('output always in (0, 1) for finite inputs', () => {
    for (const d of Object.keys(DOMAIN_OFFSETS) as Domain[]) {
      for (const q of [0.001, 0.05, 0.5, 0.95, 0.999]) {
        const p = recalibratePolymarketPrice(q, d, 30);
        expect(p).toBeGreaterThan(0);
        expect(p).toBeLessThan(1);
      }
    }
  });

  test('politics shifts upward when q < 0.5 (paper alpha=+0.15)', () => {
    const p = recalibratePolymarketPrice(0.30, 'politics', 30);
    expect(p).toBeGreaterThan(0.30);
  });

  test('politics shifts upward at q = 0.5 by ≈ Φ(α)', () => {
    const p = recalibratePolymarketPrice(0.50, 'politics', 1);
    // log1p(1)=ln2≈0.693 → slope = 1+0.05·ln2≈1.035
    // z=0 → slope·z+α=0.15 → Φ(0.15)≈0.5596
    expect(p).toBeGreaterThan(0.55);
    expect(p).toBeLessThan(0.57);
  });

  test('crypto shift smaller than politics shift at q < 0.5', () => {
    const polP = recalibratePolymarketPrice(0.30, 'politics', 30);
    const cryptoP = recalibratePolymarketPrice(0.30, 'crypto', 30);
    expect(polP - 0.30).toBeGreaterThan(cryptoP - 0.30);
  });

  test('horizon amplifies recalibration when q > 0.5 (politics)', () => {
    // At q=0.7, z>0; slope>1 amplifies upward, α=+0.15 also pushes upward.
    // Both effects compound, so longer T ⇒ stronger upward shift.
    const short = recalibratePolymarketPrice(0.70, 'politics', 1);
    const long = recalibratePolymarketPrice(0.70, 'politics', 365);
    expect(long).toBeGreaterThan(short);
  });

  test('macro applies smaller alpha than politics', () => {
    const polit = recalibratePolymarketPrice(0.30, 'politics', 30);
    const macro = recalibratePolymarketPrice(0.30, 'macro', 30);
    expect(polit).toBeGreaterThan(macro);
  });

  test('clips daysToExpiry < 1 to 1 (no negative log)', () => {
    expect(() => recalibratePolymarketPrice(0.5, 'politics', 0)).not.toThrow();
    const p = recalibratePolymarketPrice(0.5, 'politics', 0);
    expect(Number.isFinite(p)).toBe(true);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(1);
  });

  test('does not flip ordering: q1<q2 ⇒ recal(q1)<recal(q2)', () => {
    const a = recalibratePolymarketPrice(0.20, 'politics', 30);
    const b = recalibratePolymarketPrice(0.40, 'politics', 30);
    const c = recalibratePolymarketPrice(0.60, 'politics', 30);
    expect(a).toBeLessThan(b);
    expect(b).toBeLessThan(c);
  });
});
