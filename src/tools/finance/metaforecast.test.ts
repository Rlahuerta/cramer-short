/**
 * P2b — metaforecast.org cross-platform fusion (TDD, RED first).
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §6
 *
 * Pure helpers under test:
 *   - parseMetaforecastResponse — schema-tolerant deserialiser
 *   - findBestMetaforecastMatch  — fuzzy match by question keywords
 *   - computeCrossPlatformDelta  — |p_poly − p_meta|
 *   - shouldFlagCrossPlatform    — threshold check (Δp > 10pp ⇒ true)
 *
 * The HTTP fetcher (fetchMetaforecastQuestions) is exported but not unit-tested
 * here — it is a thin curl wrapper.
 */

import { describe, expect, it } from 'bun:test';
import {
  computeCrossPlatformDelta,
  findBestMetaforecastMatch,
  parseMetaforecastResponse,
  shouldFlagCrossPlatform,
} from './metaforecast.js';

describe('parseMetaforecastResponse', () => {
  it('extracts question, probability, platform, stars from valid records', () => {
    const raw = [
      {
        title: 'Will the Fed cut rates in March 2026?',
        options: [{ name: 'Yes', probability: 0.42 }],
        platform: 'metaculus',
        qualityindicators: { stars: 3 },
        url: 'https://metaculus.com/q/12345',
      },
    ];
    const parsed = parseMetaforecastResponse(raw);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].title).toBe('Will the Fed cut rates in March 2026?');
    expect(parsed[0].probability).toBeCloseTo(0.42, 6);
    expect(parsed[0].platform).toBe('metaculus');
    expect(parsed[0].stars).toBe(3);
  });

  it('skips malformed records silently', () => {
    const raw = [
      { title: 'No probability here', options: [] },
      { title: 'Missing options entirely', platform: 'manifold' },
      null,
      undefined,
      { title: 'Valid', options: [{ name: 'Yes', probability: 0.5 }], platform: 'kalshi' },
    ];
    const parsed = parseMetaforecastResponse(raw as any);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].title).toBe('Valid');
  });

  it('clamps probabilities into [0, 1]', () => {
    const raw = [
      { title: 'A', options: [{ name: 'Yes', probability: 1.5 }], platform: 'p' },
      { title: 'B', options: [{ name: 'Yes', probability: -0.2 }], platform: 'p' },
    ];
    const parsed = parseMetaforecastResponse(raw);
    expect(parsed[0].probability).toBe(1);
    expect(parsed[1].probability).toBe(0);
  });

  it('returns empty array for non-array input', () => {
    expect(parseMetaforecastResponse(null as any)).toEqual([]);
    expect(parseMetaforecastResponse({} as any)).toEqual([]);
    expect(parseMetaforecastResponse('not array' as any)).toEqual([]);
  });
});

describe('findBestMetaforecastMatch', () => {
  const candidates = [
    { title: 'Will the Fed cut rates in March 2026?', probability: 0.42, platform: 'metaculus', stars: 3 },
    { title: 'Will Trump win 2024?', probability: 0.55, platform: 'manifold', stars: 2 },
    { title: 'Will Bitcoin reach 100k by year end?', probability: 0.30, platform: 'kalshi', stars: 4 },
  ];

  it('returns the best keyword match', () => {
    const match = findBestMetaforecastMatch('Will Bitcoin hit $100k?', candidates);
    expect(match?.title).toContain('Bitcoin');
  });

  it('returns null when no candidate shares enough keywords', () => {
    const match = findBestMetaforecastMatch('Will SpaceX land on Mars?', candidates);
    expect(match).toBeNull();
  });

  it('returns null for empty candidate list', () => {
    expect(findBestMetaforecastMatch('Anything', [])).toBeNull();
  });

  it('prefers higher star quality on ties', () => {
    const ties = [
      { title: 'Will rates change soon?', probability: 0.5, platform: 'a', stars: 1 },
      { title: 'Will rates change soon?', probability: 0.5, platform: 'b', stars: 4 },
    ];
    const match = findBestMetaforecastMatch('Will rates change soon?', ties);
    expect(match?.platform).toBe('b');
  });
});

describe('computeCrossPlatformDelta', () => {
  it('returns absolute difference', () => {
    expect(computeCrossPlatformDelta(0.40, 0.25)).toBeCloseTo(0.15, 6);
    expect(computeCrossPlatformDelta(0.20, 0.50)).toBeCloseTo(0.30, 6);
    expect(computeCrossPlatformDelta(0.50, 0.50)).toBe(0);
  });

  it('keeps disagreements in raw probability space', () => {
    expect(computeCrossPlatformDelta(0.91, 0.09)).toBeCloseTo(0.82, 6);
  });
});

describe('shouldFlagCrossPlatform', () => {
  it('flags when delta > 10pp', () => {
    expect(shouldFlagCrossPlatform(0.11)).toBe(true);
    expect(shouldFlagCrossPlatform(0.50)).toBe(true);
  });

  it('does not flag at or below 10pp', () => {
    expect(shouldFlagCrossPlatform(0.10)).toBe(false);
    expect(shouldFlagCrossPlatform(0.05)).toBe(false);
    expect(shouldFlagCrossPlatform(0)).toBe(false);
  });

  it('does not take absolute value internally', () => {
    expect(shouldFlagCrossPlatform(-0.20)).toBe(false);
  });
});
