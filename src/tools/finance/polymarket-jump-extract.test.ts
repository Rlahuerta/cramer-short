/**
 * Unit tests for extractJumpEventMarkets (Idea 2 / B1).
 */
import { describe, expect, test } from 'bun:test';
import { extractJumpEventMarkets, type PolymarketMarketResult } from './polymarket.js';

const NOW = new Date('2025-06-01T00:00:00Z');
const HORIZON = new Date('2025-07-01T00:00:00Z'); // +30 days

function mkt(over: Partial<PolymarketMarketResult>): PolymarketMarketResult {
  return {
    marketId: 'm1',
    question: 'Will X happen?',
    probability: 0.3,
    volume24h: 10_000,
    ageDays: 7,
    endDate: '2025-06-15T00:00:00Z',
    ...over,
  };
}

describe('extractJumpEventMarkets', () => {
  test('passes a clean market', () => {
    const out = extractJumpEventMarkets([mkt({})], { horizonDate: HORIZON, now: NOW });
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe('m1');
    expect(out[0].probability).toBe(0.3);
    expect(out[0].daysToSettlement).toBe(14);
  });

  test('drops degenerate probabilities', () => {
    const out = extractJumpEventMarkets(
      [mkt({ probability: 0 }), mkt({ probability: 1 }), mkt({ probability: 0.5 })],
      { horizonDate: HORIZON, now: NOW },
    );
    expect(out).toHaveLength(1);
    expect(out[0].probability).toBe(0.5);
  });

  test('drops illiquid markets below volume floor', () => {
    const out = extractJumpEventMarkets(
      [mkt({ volume24h: 100 })],
      { horizonDate: HORIZON, now: NOW, minVolume24h: 5_000 },
    );
    expect(out).toHaveLength(0);
  });

  test('default minVolume24h floor is 5000 — rejects 4999 without explicit option', () => {
    // The 5_000 threshold is the trust gate; callers should not need to specify it
    const out = extractJumpEventMarkets(
      [mkt({ volume24h: 4_999 })],
      { horizonDate: HORIZON, now: NOW },
    );
    expect(out).toHaveLength(0);
  });

  test('default minVolume24h floor allows exactly 5000', () => {
    const out = extractJumpEventMarkets(
      [mkt({ volume24h: 5_000 })],
      { horizonDate: HORIZON, now: NOW },
    );
    expect(out).toHaveLength(1);
  });

  test('drops too-young markets', () => {
    const out = extractJumpEventMarkets(
      [mkt({ ageDays: 1 }), mkt({ ageDays: undefined })],
      { horizonDate: HORIZON, now: NOW, minAgeDays: 2 },
    );
    expect(out).toHaveLength(0);
  });

  test('drops markets resolving past horizon', () => {
    const out = extractJumpEventMarkets(
      [mkt({ endDate: '2025-08-01T00:00:00Z' })],
      { horizonDate: HORIZON, now: NOW },
    );
    expect(out).toHaveLength(0);
  });

  test('floors daysToSettlement at 1 for already-settled markets', () => {
    const out = extractJumpEventMarkets(
      [mkt({ endDate: '2025-05-30T00:00:00Z' })], // before now
      { horizonDate: HORIZON, now: NOW },
    );
    expect(out).toHaveLength(1);
    expect(out[0].daysToSettlement).toBe(1);
  });

  test('falls back to question when marketId is absent', () => {
    const out = extractJumpEventMarkets(
      [mkt({ marketId: undefined, question: 'fallback-id' })],
      { horizonDate: HORIZON, now: NOW },
    );
    expect(out[0].id).toBe('fallback-id');
  });
});
