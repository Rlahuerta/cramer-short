import { describe, expect, test } from 'bun:test';
import {
  computePriceVelocityPpH,
  computeMaxHourlyJump,
  parseClobPriceHistory,
  type ClobPricePoint,
} from './polymarket-clob.js';

describe('parseClobPriceHistory', () => {
  test('parses well-formed CLOB response', () => {
    const raw = { history: [{ t: 1000, p: 0.30 }, { t: 4600, p: 0.32 }] };
    const out = parseClobPriceHistory(raw);
    expect(out).toEqual([
      { tSec: 1000, p: 0.30 },
      { tSec: 4600, p: 0.32 },
    ]);
  });

  test('rejects non-finite probabilities', () => {
    const raw = { history: [{ t: 1, p: NaN }, { t: 2, p: 0.5 }, { t: 3, p: 2.0 }] };
    const out = parseClobPriceHistory(raw);
    expect(out).toEqual([{ tSec: 2, p: 0.5 }]);
  });

  test('returns empty on malformed input', () => {
    expect(parseClobPriceHistory(null)).toEqual([]);
    expect(parseClobPriceHistory({})).toEqual([]);
    expect(parseClobPriceHistory({ history: 'oops' })).toEqual([]);
  });

  test('sorts by timestamp ascending', () => {
    const raw = { history: [{ t: 30, p: 0.4 }, { t: 10, p: 0.2 }, { t: 20, p: 0.3 }] };
    const out = parseClobPriceHistory(raw);
    expect(out.map((x) => x.tSec)).toEqual([10, 20, 30]);
  });
});

describe('computePriceVelocityPpH', () => {
  test('returns 0 for empty / single-point series', () => {
    expect(computePriceVelocityPpH([])).toBe(0);
    expect(computePriceVelocityPpH([{ tSec: 0, p: 0.5 }])).toBe(0);
  });

  test('linear ramp +1pp per hour returns ~1.0', () => {
    const pts: ClobPricePoint[] = [];
    for (let h = 0; h < 6; h += 1) {
      pts.push({ tSec: h * 3600, p: 0.30 + h * 0.01 });
    }
    expect(computePriceVelocityPpH(pts)).toBeCloseTo(1.0, 3);
  });

  test('linear ramp -2pp per hour returns ~-2.0', () => {
    const pts: ClobPricePoint[] = [];
    for (let h = 0; h < 6; h += 1) {
      pts.push({ tSec: h * 3600, p: 0.50 - h * 0.02 });
    }
    expect(computePriceVelocityPpH(pts)).toBeCloseTo(-2.0, 3);
  });

  test('flat series returns ~0', () => {
    const pts: ClobPricePoint[] = [];
    for (let h = 0; h < 6; h += 1) {
      pts.push({ tSec: h * 3600, p: 0.40 });
    }
    expect(computePriceVelocityPpH(pts)).toBeCloseTo(0, 6);
  });

  test('only uses the last lookbackHours window', () => {
    // 24h history but lookbackHours=3 → only last 3 points
    const pts: ClobPricePoint[] = [];
    for (let h = 0; h < 24; h += 1) pts.push({ tSec: h * 3600, p: 0.30 + h * 0.01 });
    // Last 3 hours linear ramp +1pp/hr
    const v = computePriceVelocityPpH(pts, 3);
    expect(v).toBeCloseTo(1.0, 3);
  });
});

describe('computeMaxHourlyJump', () => {
  test('returns 0 for short series', () => {
    expect(computeMaxHourlyJump([])).toBe(0);
    expect(computeMaxHourlyJump([{ tSec: 0, p: 0.5 }])).toBe(0);
  });

  test('returns the max abs hourly delta over window', () => {
    const pts: ClobPricePoint[] = [
      { tSec: 0, p: 0.30 },
      { tSec: 3600, p: 0.32 },     // +0.02
      { tSec: 7200, p: 0.45 },     // +0.13 ← max
      { tSec: 10_800, p: 0.40 },   // -0.05
    ];
    expect(computeMaxHourlyJump(pts)).toBeCloseTo(0.13, 6);
  });

  test('excludes data older than windowHours', () => {
    const pts: ClobPricePoint[] = [
      { tSec: 0, p: 0.10 },
      { tSec: 3600, p: 0.50 },         // huge jump but old
      { tSec: 100 * 3600, p: 0.50 },
      { tSec: 101 * 3600, p: 0.55 },   // +0.05 in the window
    ];
    expect(computeMaxHourlyJump(pts, 24)).toBeCloseTo(0.05, 6);
  });
});
