import { describe, it, expect } from 'bun:test';
import { normalizeSentiment, STATE_INDEX } from './core.js';
import { adjustTransitionMatrix, buildDefaultMatrix } from './transition.js';
import { computeMarkovDistribution, markovDistributionTool } from '../markov-distribution.js';
import { MS_PER_DAY } from '../../../utils/time.js';

const FIXED_NOW_MS = Date.parse('2025-04-02T12:00:00.000Z');

describe('normalizeSentiment', () => {
  it('passes through valid decimal inputs (0–1 scale)', () => {
    const result = normalizeSentiment({ bullish: 0.71, bearish: 0.29 });
    expect(result).toBeDefined();
    expect(result!.bullish).toBeCloseTo(0.71, 5);
    expect(result!.bearish).toBeCloseTo(0.29, 5);
  });

  it('normalizes percent-style inputs (71/29 → 0.71/0.29)', () => {
    const result = normalizeSentiment({ bullish: 71, bearish: 29 });
    expect(result).toBeDefined();
    expect(result!.bullish).toBeCloseTo(0.71, 5);
    expect(result!.bearish).toBeCloseTo(0.29, 5);
  });

  it('rejects mixed decimal/percent scales', () => {
    expect(normalizeSentiment({ bullish: 71, bearish: 0.3 })).toBeUndefined();
    expect(normalizeSentiment({ bullish: 0.7, bearish: 30 })).toBeUndefined();
  });

  it('returns undefined for negative values', () => {
    expect(normalizeSentiment({ bullish: -0.1, bearish: 0.5 })).toBeUndefined();
    expect(normalizeSentiment({ bullish: 0.5, bearish: -10 })).toBeUndefined();
  });

  it('returns undefined for values > 100 (out of range)', () => {
    expect(normalizeSentiment({ bullish: 150, bearish: 30 })).toBeUndefined();
    expect(normalizeSentiment({ bullish: 50, bearish: 101 })).toBeUndefined();
  });

  it('returns undefined for non-number types', () => {
    expect(normalizeSentiment({ bullish: '71', bearish: 29 })).toBeUndefined();
    expect(normalizeSentiment({ bullish: 71, bearish: '29' })).toBeUndefined();
    expect(normalizeSentiment({ bullish: NaN, bearish: 29 })).toBeUndefined();
    expect(normalizeSentiment({ bullish: 71, bearish: Infinity })).toBeUndefined();
  });

  it('returns undefined for null, undefined, array, or malformed objects', () => {
    expect(normalizeSentiment(null)).toBeUndefined();
    expect(normalizeSentiment(undefined)).toBeUndefined();
    expect(normalizeSentiment([71, 29])).toBeUndefined();
    expect(normalizeSentiment({})).toBeUndefined();
    expect(normalizeSentiment({ bull: 71, bear: 29 })).toBeUndefined();
  });

  it('handles exact boundary values: 0, 1, and 100', () => {
    const at1 = normalizeSentiment({ bullish: 1, bearish: 0 });
    expect(at1).toBeDefined();
    expect(at1!.bullish).toBe(1);
    expect(at1!.bearish).toBe(0);

    const at100 = normalizeSentiment({ bullish: 100, bearish: 0 });
    expect(at100).toBeDefined();
    expect(at100!.bullish).toBe(1);
    expect(at100!.bearish).toBe(0);

    const at0 = normalizeSentiment({ bullish: 0, bearish: 0 });
    expect(at0).toBeDefined();
    expect(at0!.bullish).toBe(0);
    expect(at0!.bearish).toBe(0);
  });

  it('keeps normalized results in [0, 1]', () => {
    const result = normalizeSentiment({ bullish: 99.9, bearish: 0 });
    expect(result).toBeDefined();
    expect(result!.bullish).toBeLessThanOrEqual(1);
    expect(result!.bearish).toBeGreaterThanOrEqual(0);
  });

  it('sentiment with percent inputs adjusts adjustTransitionMatrix correctly', () => {
    const baseMatrix = buildDefaultMatrix();
    const result = normalizeSentiment({ bullish: 80, bearish: 20 });
    expect(result).toBeDefined();
    const adjusted = adjustTransitionMatrix(baseMatrix, result!);
    const bullIdx = STATE_INDEX['bull'];
    expect(adjusted[bullIdx][bullIdx]).toBeGreaterThan(baseMatrix[bullIdx][bullIdx]);
  });

  it('computeMarkovDistribution rejects invalid mixed sentiment scales', async () => {
    const prices = Array.from({ length: 90 }, (_, i) => 100 + i * 0.2 + Math.sin(i) * 2);
    expect(computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 5,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      sentiment: { bullish: 71, bearish: 0.3 } as { bullish: number; bearish: number },
    })).rejects.toThrow('Invalid sentiment input');
  });

  it('markovDistributionTool accepts percent-style sentiment on the real tool path', async () => {
    const canonicalPrices = Array.from({ length: 91 }, (_, i) => {
      let p = 100;
      for (let j = 0; j <= i; j++) {
        p *= 1 + Math.sin(j * 0.15) * 0.006;
      }
      return Math.round(p * 100) / 100;
    });
    const canonicalCurrentPrice = canonicalPrices[canonicalPrices.length - 1];
    const canonicalAnchors = [0.97, 1.0, 1.03].map((mult, idx) => ({
      question: `Will FMT_TEST be above $${Math.round(canonicalCurrentPrice * mult)} on April 9?`,
      probability: [0.72, 0.5, 0.28][idx],
      volume: 5000,
      createdAt: FIXED_NOW_MS - MS_PER_DAY * 5,
    }));

    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 7,
      historicalPrices: canonicalPrices,
      polymarketMarkets: canonicalAnchors,
      sentiment: { bullish: 71, bearish: 29 },
    });

    const parsed = JSON.parse(output) as {
      data: {
        _tool: string;
        canonical: { diagnostics: { canEmitCanonical: boolean } };
      };
    };

    expect(parsed.data._tool).toBe('markov_distribution');
    expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
  });
});
