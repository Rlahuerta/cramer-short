import { describe, expect, test } from 'bun:test';
import { extractJumpEventMarkets, type PolymarketMarketResult } from './polymarket.js';

const NOW = new Date('2026-04-15T12:00:00Z');
const HORIZON = new Date('2026-07-15T12:00:00Z');

function makeMarket(over: Partial<PolymarketMarketResult> = {}): PolymarketMarketResult {
  return {
    question: 'Will X happen?',
    probability: 0.30,
    yesPrice: 0.30,
    noPrice: 0.70,
    volume: 100_000,
    volume24h: 50_000,
    liquidity: 25_000,
    endDate: '2026-05-15T12:00:00Z',
    closed: false,
    active: true,
    url: 'https://polymarket.com/x',
    marketId: 'mkt-1',
    ageDays: 21,
    ...over,
  } as PolymarketMarketResult;
}

describe('extractJumpEventMarkets — P1c near-expiry filter', () => {
  test('drops markets settling in less than 24h', () => {
    const m = makeMarket({
      endDate: new Date(NOW.getTime() + 12 * 3_600_000).toISOString(),
    });
    const out = extractJumpEventMarkets([m], { horizonDate: HORIZON, now: NOW });
    expect(out.length).toBe(0);
  });

  test('drops markets that already settled', () => {
    const m = makeMarket({
      endDate: new Date(NOW.getTime() - 24 * 3_600_000).toISOString(),
    });
    const out = extractJumpEventMarkets([m], { horizonDate: HORIZON, now: NOW });
    expect(out.length).toBe(0);
  });

  test('keeps markets settling in 1 day or more', () => {
    const m = makeMarket({
      endDate: new Date(NOW.getTime() + 36 * 3_600_000).toISOString(),
    });
    const out = extractJumpEventMarkets([m], { horizonDate: HORIZON, now: NOW });
    expect(out.length).toBe(1);
    expect(out[0]!.daysToSettlement).toBeGreaterThanOrEqual(1);
  });

  test('keeps mid-horizon markets unchanged (regression)', () => {
    const m = makeMarket();
    const out = extractJumpEventMarkets([m], { horizonDate: HORIZON, now: NOW });
    expect(out.length).toBe(1);
    expect(out[0]!.id).toBe('mkt-1');
  });
});
