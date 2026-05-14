import { describe, it, expect } from 'bun:test';
import { getAssetProfile } from './asset-profile.js';
import { normalizeHistoricalPriceTicker } from './live-policies.js';
import { computeMarkovDistribution, normalizeAnchorPricesForETF } from '../markov-distribution.js';
import type { PriceThreshold } from '../markov-distribution.js';
import { MS_PER_DAY } from '../../../utils/time.js';

describe('getAssetProfile', () => {
  it('classifies SPY as ETF', () => {
    expect(getAssetProfile('SPY').type).toBe('etf');
  });

  it('classifies QQQ as ETF', () => {
    expect(getAssetProfile('QQQ').type).toBe('etf');
  });

  it('classifies GLD as commodity', () => {
    expect(getAssetProfile('GLD').type).toBe('commodity');
  });

  it('classifies SLV as commodity', () => {
    expect(getAssetProfile('SLV').type).toBe('commodity');
  });

  it('classifies CL as commodity', () => {
    expect(getAssetProfile('CL').type).toBe('commodity');
  });

  it('classifies NG as commodity', () => {
    expect(getAssetProfile('NG').type).toBe('commodity');
  });

  it('classifies GC as commodity', () => {
    expect(getAssetProfile('GC').type).toBe('commodity');
  });

  it('classifies USO as commodity', () => {
    expect(getAssetProfile('USO').type).toBe('commodity');
  });

  it('classifies GOLD as equity', () => {
    expect(getAssetProfile('GOLD').type).toBe('equity');
  });

  it('classifies XAUUSD as commodity', () => {
    expect(getAssetProfile('XAUUSD').type).toBe('commodity');
  });

  it('classifies AAPL as equity', () => {
    expect(getAssetProfile('AAPL').type).toBe('equity');
  });

  it('classifies TSLA as equity', () => {
    expect(getAssetProfile('TSLA').type).toBe('equity');
  });

  it('classifies BTC-USD as crypto', () => {
    expect(getAssetProfile('BTC-USD').type).toBe('crypto');
  });

  it('classifies ETH-USD as crypto', () => {
    expect(getAssetProfile('ETH-USD').type).toBe('crypto');
  });

  it('case insensitive', () => {
    expect(getAssetProfile('spy').type).toBe('etf');
    expect(getAssetProfile('btc-usd').type).toBe('crypto');
  });

  it('ETFs have lower kappa multiplier (more trust)', () => {
    const etf = getAssetProfile('SPY');
    const crypto = getAssetProfile('BTC-USD');
    expect(etf.kappaMultiplier).toBeLessThan(crypto.kappaMultiplier);
  });

  it('crypto has lower HMM weight multiplier', () => {
    const etf = getAssetProfile('SPY');
    const crypto = getAssetProfile('BTC-USD');
    expect(crypto.hmmWeightMultiplier).toBeLessThan(etf.hmmWeightMultiplier);
  });

  it('crypto has fatter tails (lower Student-t nu)', () => {
    const etf = getAssetProfile('SPY');
    const crypto = getAssetProfile('BTC-USD');
    expect(crypto.studentTNu).toBeLessThan(etf.studentTNu);
  });

  it('unknown ticker defaults to equity', () => {
    expect(getAssetProfile('UNKNOWN_TICKER').type).toBe('equity');
  });
});
describe('commodity asset profile', () => {
  it('has maxDailyDrift defined', () => {
    const profile = getAssetProfile('CL');
    expect(profile.maxDailyDrift).toBeDefined();
    expect(profile.maxDailyDrift!).toBeGreaterThan(0);
    expect(profile.maxDailyDrift!).toBeLessThanOrEqual(0.015);
  });

  it('all profiles have maxDailyDrift defined', () => {
    for (const ticker of ['SPY', 'AAPL', 'BTC-USD', 'CL']) {
      const profile = getAssetProfile(ticker);
      expect(profile.maxDailyDrift).toBeDefined();
      expect(profile.maxDailyDrift!).toBeGreaterThan(0);
    }
  });

  it('crypto has the highest maxDailyDrift', () => {
    const crypto = getAssetProfile('BTC-USD');
    const commodity = getAssetProfile('CL');
    const etf = getAssetProfile('SPY');
    expect(crypto.maxDailyDrift!).toBeGreaterThan(commodity.maxDailyDrift!);
    expect(commodity.maxDailyDrift!).toBeGreaterThan(etf.maxDailyDrift!);
  });
});
describe('normalizeHistoricalPriceTicker', () => {
  it('maps oil proxy tickers to USO', () => {
    expect(normalizeHistoricalPriceTicker('OIL')).toBe('USO');
    expect(normalizeHistoricalPriceTicker('WTICOUSD')).toBe('USO');
    expect(normalizeHistoricalPriceTicker('CRUDE')).toBe('USO');
  });

  it('passes through regular tickers unchanged', () => {
    expect(normalizeHistoricalPriceTicker('AAPL')).toBe('AAPL');
    expect(normalizeHistoricalPriceTicker('BTC')).toBe('BTC');
    expect(normalizeHistoricalPriceTicker('GLD')).toBe('GLD');
  });

  it('trims and uppercases input', () => {
    expect(normalizeHistoricalPriceTicker('  oil  ')).toBe('USO');
    expect(normalizeHistoricalPriceTicker('wtiCOusd')).toBe('USO');
  });
});
describe('normalizeAnchorPricesForETF', () => {
  const makeAnchor = (price: number): PriceThreshold => ({
    price,
    rawProbability: 0.5,
    probability: 0.475,
    trustScore: 'high',
    source: 'polymarket',
  });

  it('converts gold futures anchors ($5,500) to GLD-scale (~$485)', () => {
    const anchors = [makeAnchor(5000), makeAnchor(5500), makeAnchor(6000)];
    const result = normalizeAnchorPricesForETF(anchors, 415, 'GLD');
    // Should scale down by ~415/5500 ≈ 0.075
    for (const a of result) {
      expect(a.price).toBeLessThan(1000);
      expect(a.price).toBeGreaterThan(300);
    }
  });

  it('preserves anchor order after conversion', () => {
    const anchors = [makeAnchor(4000), makeAnchor(5000), makeAnchor(6000)];
    const result = normalizeAnchorPricesForETF(anchors, 415, 'GLD');
    for (let i = 1; i < result.length; i++) {
      expect(result[i].price).toBeGreaterThan(result[i - 1].price);
    }
  });

  it('does not convert when anchors are already in ETF range', () => {
    const anchors = [makeAnchor(400), makeAnchor(420), makeAnchor(450)];
    const result = normalizeAnchorPricesForETF(anchors, 415, 'GLD');
    expect(result[0].price).toBe(400);
    expect(result[1].price).toBe(420);
    expect(result[2].price).toBe(450);
  });

  it('does not convert for non-commodity tickers', () => {
    const anchors = [makeAnchor(5000), makeAnchor(5500)];
    const result = normalizeAnchorPricesForETF(anchors, 150, 'AAPL');
    expect(result[0].price).toBe(5000);
    expect(result[1].price).toBe(5500);
  });

  it('works for silver ETF (SLV)', () => {
    // Silver at ~$30/oz, SLV at ~$28. Polymarket might say "$35 silver"
    // If anchors are >3x current, conversion kicks in
    const anchors = [makeAnchor(100), makeAnchor(120)];
    const result = normalizeAnchorPricesForETF(anchors, 28, 'SLV');
    // 100 > 28*3=84, so conversion applies
    for (const a of result) {
      expect(a.price).toBeLessThan(50);
    }
  });

  it('works for oil ETF (USO)', () => {
    const anchors = [makeAnchor(300), makeAnchor(400)];
    const result = normalizeAnchorPricesForETF(anchors, 80, 'USO');
    // 300 > 80*3=240, so conversion applies
    for (const a of result) {
      expect(a.price).toBeLessThan(120);
    }
  });

  it('returns empty array for empty input', () => {
    expect(normalizeAnchorPricesForETF([], 415, 'GLD')).toEqual([]);
  });

  it('preserves trustScore and probability after conversion', () => {
    const anchors = [makeAnchor(5500)];
    anchors[0].trustScore = 'high';
    anchors[0].probability = 0.42;
    const result = normalizeAnchorPricesForETF(anchors, 415, 'GLD');
    expect(result[0].trustScore).toBe('high');
    expect(result[0].probability).toBe(0.42);
  });

  it('uses correct median for even-length anchor arrays', () => {
    // 4 anchors: [4200, 4700, 5500, 6000] → median = (4700+5500)/2 = 5100
    // conversionFactor = 430 / 5100 = 0.08431
    const anchors = [makeAnchor(4200), makeAnchor(4700), makeAnchor(5500), makeAnchor(6000)];
    const result = normalizeAnchorPricesForETF(anchors, 430, 'GLD');
    // With median=5100: $4700 → 4700*(430/5100) ≈ $396.47
    // With wrong median=5500 (old code): $4700 → 4700*(430/5500) ≈ $367.27
    const converted4700 = result.find(a => a.price > 390 && a.price < 405);
    expect(converted4700).toBeDefined();
    // $5500 should NOT map to exactly current price (that was the bug)
    const converted5500 = result.find(a => Math.abs(a.price - 430) < 5);
    expect(converted5500).toBeUndefined(); // No anchor should land exactly at current price
  });

  it('keeps GLD 1d/2d/3d manual anchor grids in ETF scale after normalization', async () => {
    const historicalPrices = Array.from({ length: 160 }, (_, i) =>
      300 * Math.exp(i * 0.0004 + Math.sin(i * 0.11) * 0.002),
    );
    const currentPrice = historicalPrices[historicalPrices.length - 1];
    const now = new Date('2026-05-06T12:00:00Z').getTime();

    for (const horizon of [1, 2, 3]) {
      const result = await computeMarkovDistribution({
        ticker: 'GLD',
        horizon,
        currentPrice,
        historicalPrices,
        polymarketMarkets: [
          {
            question: `Will gold exceed $4,650 in ${horizon} day${horizon === 1 ? '' : 's'}?`,
            probability: 0.58,
            volume: 15_000,
            createdAt: now - 7 * MS_PER_DAY,
            endDate: new Date(now + horizon * MS_PER_DAY).toISOString(),
          },
        ],
      });

      expect(Math.max(...result.distribution.map((point) => point.price))).toBeLessThan(1_000);
      expect(result.distribution.some((point) => Math.abs(point.price - currentPrice) < 20)).toBe(true);
    }
  });
});
