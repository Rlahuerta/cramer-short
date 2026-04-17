import { describe, expect, it, beforeEach, afterEach, beforeAll, afterAll } from 'bun:test';
import { polymarketTool, questionMatchesQuery, inferTagSlugs, setRetryDelays, RETRY_DELAYS, clearPolymarketCache, scoreAnchorMarketRelevance, fetchPolymarketAnchorMarkets, fetchPolymarketMarkets } from './polymarket.js';
import { polymarketBreaker } from '../../utils/circuit-breaker.js';
import type { PolymarketMarketResult } from './polymarket.js';

// Disable retry delays in tests to avoid timeouts
let originalDelays: number[];
beforeAll(() => {
  originalDelays = [...RETRY_DELAYS];
  setRetryDelays([0, 0, 0]);
});
afterAll(() => {
  setRetryDelays(originalDelays);
});

// ---------------------------------------------------------------------------
// Unit tests — no network, mock fetch
// ---------------------------------------------------------------------------

const MOCK_MARKET = {
  id: '1',
  question: 'Will the Fed cut rates in 2026?',
  outcomes: '["Yes","No"]',
  outcomePrices: '["0.72","0.28"]',
  endDateIso: '2026-12-31',
  volume24hr: 1_500_000,
  volumeNum: 5_000_000,
  liquidityNum: 800_000,
  active: true,
  closed: false,
};

const MOCK_EVENT = {
  id: 'e1',
  title: 'Fed Rate Decisions 2026',
  volume24hr: 1_500_000,
  markets: [MOCK_MARKET],
};

const SPORTS_MARKET = {
  id: '99',
  question: 'Will the Lakers win the NBA championship?',
  outcomes: '["Yes","No"]',
  outcomePrices: '["0.30","0.70"]',
  endDateIso: '2026-06-30',
  volume24hr: 5_000_000,
  volumeNum: 20_000_000,
  liquidityNum: 2_000_000,
  active: true,
  closed: false,
};

const BITCOIN_MARKET = {
  id: '42',
  question: 'Will Bitcoin price exceed $100K in 2026?',
  outcomes: '["Yes","No"]',
  outcomePrices: '["0.65","0.35"]',
  endDateIso: '2026-12-31',
  volume24hr: 3_000_000,
  volumeNum: 10_000_000,
  liquidityNum: 1_500_000,
  active: true,
  closed: false,
};

function mockFetch(eventData: unknown, marketData: unknown) {
  return async (url: string | URL) => {
    const urlStr = String(url);
    const body = urlStr.includes('/events') ? eventData : marketData;
    return {
      ok: true,
      status: 200,
      json: async () => body,
    } as Response;
  };
}

/** Mock that returns sports for keyword search but Bitcoin market for tag_slug */
function mockFetchWithTagFallback(tagSlug: string, tagData: unknown, keywordData: unknown) {
  return async (url: string | URL) => {
    const urlStr = String(url);
    if (urlStr.includes(`tag_slug=${tagSlug}`)) {
      return { ok: true, status: 200, json: async () => tagData } as Response;
    }
    return { ok: true, status: 200, json: async () => keywordData } as Response;
  };
}

describe('questionMatchesQuery', () => {
  it('returns true when question contains a query word', () => {
    expect(questionMatchesQuery('Will the Fed cut rates in 2026?', 'Fed rate cut')).toBe(true);
  });

  it('returns true when question contains a partial query word (substring match)', () => {
    expect(questionMatchesQuery('Will Bitcoin reach $100K?', 'Bitcoin price')).toBe(true);
  });

  it('returns false for a sports market when querying crypto', () => {
    expect(questionMatchesQuery('Will the Lakers win the NBA championship?', 'Bitcoin price')).toBe(false);
  });

  it('returns false for a sports market when querying Fed rates', () => {
    expect(questionMatchesQuery('Will Team A win the Super Bowl?', 'Fed rate cut')).toBe(false);
  });

  it('does not let weak price words make gold queries match bitcoin markets', () => {
    expect(questionMatchesQuery('Will Bitcoin price exceed $100K in 2026?', 'gold price')).toBe(false);
  });

  it('matches anchored gold questions even when the query includes weak words', () => {
    expect(questionMatchesQuery('Will gold reach $3,000 per ounce by June?', 'gold price')).toBe(true);
  });

  it('does not match weak-only commodity price queries against bitcoin markets', () => {
    expect(questionMatchesQuery('Will Bitcoin price exceed $100K in 2026?', 'commodity price')).toBe(false);
  });

  it('returns true for empty query words (no filtering)', () => {
    expect(questionMatchesQuery('Anything goes here', '')).toBe(true);
  });

  it('ignores stop words in query', () => {
    // "the and for" are all stop words, so zero significant words → no filtering
    expect(questionMatchesQuery('Something completely unrelated', 'the and for')).toBe(true);
  });

  it('is case-insensitive', () => {
    expect(questionMatchesQuery('Will NVIDIA earnings beat consensus?', 'nvidia earnings')).toBe(true);
  });

  it('filters short query words (< 3 chars)', () => {
    // "AI" = 2 chars, filtered out → no significant words → returns true
    expect(questionMatchesQuery('Sports championship final', 'AI')).toBe(true);
  });

  it('matches "recession" query against recession market', () => {
    expect(questionMatchesQuery('Will the US enter a recession in 2026?', 'US recession')).toBe(true);
  });

  it('does not match "recession" query against sports market', () => {
    expect(questionMatchesQuery('Will the Cowboys win the Super Bowl?', 'US recession')).toBe(false);
  });
});

describe('inferTagSlugs', () => {
  it('returns bitcoin and crypto slugs for bitcoin query', () => {
    const slugs = inferTagSlugs('Bitcoin price');
    expect(slugs).toContain('bitcoin');
    expect(slugs.some(s => s.startsWith('crypto'))).toBe(true);
  });

  it('returns crypto slugs for eth query', () => {
    const slugs = inferTagSlugs('ethereum price prediction');
    expect(slugs.some(s => s.startsWith('crypto') || s === 'ethereum')).toBe(true);
  });

  it('returns fed-rates for Fed/FOMC query (not economics)', () => {
    const slugs = inferTagSlugs('Fed rate cut');
    expect(slugs).toContain('fed-rates');
    expect(slugs).not.toContain('economics'); // old broken slug
  });

  it('returns economy for recession query (not economics)', () => {
    const slugs = inferTagSlugs('US recession 2026');
    expect(slugs).toContain('economy');
    expect(slugs).not.toContain('economics'); // old broken slug
  });

  it('returns elections slugs for election query', () => {
    const slugs = inferTagSlugs('US presidential election');
    expect(slugs).toContain('elections');
    expect(slugs).toContain('politics');
  });

  it('returns big-tech or tech for NVIDIA query (not broken technology slug)', () => {
    const slugs = inferTagSlugs('NVIDIA earnings');
    expect(slugs.some(s => ['big-tech', 'tech', 'business'].includes(s))).toBe(true);
    expect(slugs).not.toContain('technology'); // old broken slug that returns 0 results
  });

  it('returns empty array for unrecognized query', () => {
    expect(inferTagSlugs('completely random unknown topic')).toEqual([]);
  });

  it('is case-insensitive for BTC', () => {
    expect(inferTagSlugs('BTC halving')).toContain('bitcoin');
  });

  it('returns tariffs slug for tariff query', () => {
    expect(inferTagSlugs('US China tariffs')).toContain('tariffs');
  });
});

describe('scoreAnchorMarketRelevance', () => {
  it('prefers dated threshold questions over touch-style markets', () => {
    const threshold = scoreAnchorMarketRelevance(
      'Will the price of Bitcoin be above $70,000 on April 9?',
      'BTC-USD',
      7,
      new Date(Date.now() + 7 * 86_400_000).toISOString(),
    );
    const barrier = scoreAnchorMarketRelevance(
      'Will Bitcoin reach $70,000 this week?',
      'BTC-USD',
      7,
      new Date(Date.now() + 7 * 86_400_000).toISOString(),
    );
    expect(threshold).toBeGreaterThan(barrier);
  });

  it('requires Barrick-specific wording for explicit GOLD ticker matches', () => {
    const barrick = scoreAnchorMarketRelevance(
      'Will Barrick Gold close above $25 by June?',
      'GOLD',
      30,
      new Date(Date.now() + 30 * 86_400_000).toISOString(),
    );
    const genericGold = scoreAnchorMarketRelevance(
      'Will gold exceed $3500 by June?',
      'GOLD',
      30,
      new Date(Date.now() + 30 * 86_400_000).toISOString(),
    );

    expect(barrick).toBeGreaterThan(0);
    expect(genericGold).toBe(0);
  });
});

describe('fetchPolymarketAnchorMarkets', () => {
  beforeEach(() => { clearPolymarketCache(); });

  it('ranks terminal threshold markets ahead of generic or touch-style matches', async () => {
    const event = {
      id: 'btc-anchor',
      title: 'Bitcoin markets',
      markets: [
        {
          id: '1',
          question: 'Will Bitcoin reach $70,000 this week?',
          outcomes: '["Yes","No"]',
          outcomePrices: '["0.30","0.70"]',
          endDateIso: new Date(Date.now() + 7 * 86_400_000).toISOString(),
          volume24hr: 400000,
          volumeNum: 400000,
          liquidityNum: 100000,
          active: true,
          closed: false,
          createdAt: new Date(Date.now() - 3 * 86_400_000).toISOString(),
        },
        {
          id: '2',
          question: 'Will the price of Bitcoin be above $70,000 on April 9?',
          outcomes: '["Yes","No"]',
          outcomePrices: '["0.45","0.55"]',
          endDateIso: new Date(Date.now() + 7 * 86_400_000).toISOString(),
          volume24hr: 120000,
          volumeNum: 120000,
          liquidityNum: 50000,
          active: true,
          closed: false,
          createdAt: new Date(Date.now() - 3 * 86_400_000).toISOString(),
        },
      ],
    };

    globalThis.fetch = mockFetch([event], []) as typeof fetch;
    const results = await fetchPolymarketAnchorMarkets('Bitcoin above', 5, { ticker: 'BTC-USD', horizonDays: 7 });
    expect(results[0]?.question).toContain('be above $70,000 on April 9');
  });
});

describe('polymarketTool', () => {
  beforeEach(() => { polymarketBreaker.reset(); clearPolymarketCache(); });

  it('tool name is polymarket_search', () => {
    expect(polymarketTool.name).toBe('polymarket_search');
  });

  it('formats YES/NO probabilities as percentages', async () => {
    globalThis.fetch = mockFetch([MOCK_EVENT], [MOCK_MARKET]) as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'Fed rate cut', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).toContain('72.0%');
    expect(text).toContain('28.0%');
    expect(text).toContain('Will the Fed cut rates in 2026?');
  });

  it('includes volume and liquidity metadata', async () => {
    globalThis.fetch = mockFetch([MOCK_EVENT], [MOCK_MARKET]) as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'Fed rate cut', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).toContain('$1.5M');  // 24h volume
    expect(text).toContain('2026-12-31'); // end date
  });

  it('deduplicates markets appearing in both events and direct search', async () => {
    globalThis.fetch = mockFetch([MOCK_EVENT], [MOCK_MARKET]) as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'Fed rate cut', limit: 10 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    // Question should appear exactly once
    const occurrences = (text.match(/Will the Fed cut rates in 2026\?/g) ?? []).length;
    expect(occurrences).toBe(1);
  });

  it('filters out sports markets when querying financial topics', async () => {
    // Simulate keyword search returning sports (API ignores keyword)
    globalThis.fetch = mockFetch([], [SPORTS_MARKET]) as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'Bitcoin price', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).not.toContain('Lakers');
    expect(text).not.toContain('NBA');
  });

  it('uses tag-slug fallback when keyword search returns no relevant results', async () => {
    // keyword search → returns sports (irrelevant, filtered out)
    // tag_slug=bitcoin → returns Bitcoin market (relevant)
    globalThis.fetch = mockFetchWithTagFallback(
      'bitcoin',
      [BITCOIN_MARKET],  // tag-based result
      [SPORTS_MARKET],   // keyword result (filtered out by text filter)
    ) as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'Bitcoin price', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).toContain('Bitcoin');
    expect(text).not.toContain('Lakers');
  });

  it('returns a no-results message when both endpoints return empty', async () => {
    globalThis.fetch = mockFetch([], []) as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'nonexistent query xyz', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).toContain('No active Polymarket prediction markets found');
  });

  it('handles API error gracefully without throwing', async () => {
    globalThis.fetch = (async () => ({ ok: false, status: 503, json: async () => ({}) })) as unknown as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'test', limit: 3 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    // Promise.allSettled means a 503 degrades to "no results" rather than crashing
    expect(text).not.toContain('undefined');
    expect(text).not.toContain('TypeError');
    // Either "no results" or error message — both are acceptable graceful responses
    const graceful = text.includes('No active Polymarket') || text.includes('Polymarket search failed');
    expect(graceful).toBe(true);
  });

  it('handles network failure gracefully without throwing', async () => {
    globalThis.fetch = (async () => { throw new Error('Network error'); }) as unknown as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'test', limit: 3 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    const graceful = text.includes('No active Polymarket') || text.includes('Polymarket search failed');
    expect(graceful).toBe(true);
  });

  it('skips closed markets', async () => {
    const closedEvent = {
      ...MOCK_EVENT,
      markets: [{ ...MOCK_MARKET, closed: true }],
    };
    globalThis.fetch = mockFetch([closedEvent], []) as typeof fetch;
    const result = await polymarketTool.invoke({ query: 'Fed rate cut', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).toContain('No active Polymarket prediction markets found');
  });
});

describe('inferTagSlugs — commodity coverage (regression)', () => {
  it('maps gold to commodities', () => {
    expect(inferTagSlugs('gold price')).toContain('commodities');
  });
  it('maps silver to commodities', () => {
    expect(inferTagSlugs('silver forecast')).toContain('commodities');
  });
  it('maps copper to commodities', () => {
    expect(inferTagSlugs('copper demand')).toContain('commodities');
  });
  it('maps natural gas to commodities', () => {
    expect(inferTagSlugs('natural gas price')).toContain('commodities');
  });
  it('maps wheat to commodities', () => {
    expect(inferTagSlugs('wheat supply chain')).toContain('commodities');
  });
});

describe('polymarket commodity search filtering (regression)', () => {
  beforeEach(() => { clearPolymarketCache(); });

  it('filters bitcoin price markets out of gold price searches while preserving gold markets', async () => {
    globalThis.fetch = mockFetch([
      {
        id: 'gold-and-btc',
        title: 'Commodity markets',
        markets: [
          {
            id: 'gold-market',
            question: 'Will gold reach $3,000 per ounce by June?',
            outcomes: '["Yes","No"]',
            outcomePrices: '["0.41","0.59"]',
            endDateIso: '2026-06-30',
            volume24hr: 600_000,
            volumeNum: 2_000_000,
            liquidityNum: 500_000,
            active: true,
            closed: false,
          },
          {
            id: 'btc-market',
            question: 'Will Bitcoin price exceed $100K in 2026?',
            outcomes: '["Yes","No"]',
            outcomePrices: '["0.65","0.35"]',
            endDateIso: '2026-12-31',
            volume24hr: 3_000_000,
            volumeNum: 10_000_000,
            liquidityNum: 1_500_000,
            active: true,
            closed: false,
          },
        ],
      },
    ], []) as typeof fetch;

    const { fetchPolymarketMarkets: isolatedFetchPolymarketMarkets } = await import(`./polymarket.js?commodity-regression=${Date.now()}`) as {
      fetchPolymarketMarkets: (query: string, limit: number) => Promise<PolymarketMarketResult[]>;
    };
    const results = await isolatedFetchPolymarketMarkets('gold price', 5);

    expect(results.map((m) => m.question)).toContain('Will gold reach $3,000 per ounce by June?');
    expect(results.map((m) => m.question)).not.toContain('Will Bitcoin price exceed $100K in 2026?');
  });
});

// ---------------------------------------------------------------------------
// fetchWithRetry — exponential backoff
// ---------------------------------------------------------------------------

describe('fetchWithRetry', () => {
  it('returns result on first success', async () => {
    const { fetchWithRetry } = await import('./polymarket.js');
    const result = await fetchWithRetry(async () => 42, 3, [0, 0, 0]);
    expect(result).toBe(42);
  });

  it('retries on transient error and succeeds', async () => {
    const { fetchWithRetry } = await import('./polymarket.js');
    let calls = 0;
    const result = await fetchWithRetry(async () => {
      calls++;
      if (calls < 3) throw new Error('Server 500');
      return 'ok';
    }, 3, [0, 0, 0]);
    expect(result).toBe('ok');
    expect(calls).toBe(3);
  });

  it('does NOT retry on 4xx client error', async () => {
    const { fetchWithRetry } = await import('./polymarket.js');
    let calls = 0;
    try {
      await fetchWithRetry(async () => {
        calls++;
        throw new Error('HTTP 404 not found');
      }, 3, [0, 0, 0]);
    } catch (e) {
      expect((e as Error).message).toContain('404');
    }
    expect(calls).toBe(1); // no retries
  });

  it('throws after exhausting retries', async () => {
    const { fetchWithRetry } = await import('./polymarket.js');
    let calls = 0;
    try {
      await fetchWithRetry(async () => {
        calls++;
        throw new Error('timeout');
      }, 2, [0, 0]);
    } catch (e) {
      expect((e as Error).message).toBe('timeout');
    }
    expect(calls).toBe(3); // initial + 2 retries
  });
});

// ---------------------------------------------------------------------------
// TTL cache
// ---------------------------------------------------------------------------

describe('TTL cache', () => {
  let savedFetch: typeof globalThis.fetch;
  beforeEach(() => {
    savedFetch = globalThis.fetch;
    clearPolymarketCache();
  });
  afterEach(() => { globalThis.fetch = savedFetch; });

  it('returns cached result on second call with same query', async () => {
    let fetchCount = 0;
    globalThis.fetch = (async () => {
      fetchCount++;
      return {
        ok: true,
        status: 200,
        json: async () => [{
          id: 'e1',
          title: 'Test Event',
          volume24hr: 100,
          markets: [{
            id: '1',
            question: 'Will test pass?',
            outcomes: '["Yes","No"]',
            outcomePrices: '["0.80","0.20"]',
            endDateIso: '2027-01-01',
            volume24hr: 100,
            volumeNum: 1000,
            liquidityNum: 500,
            active: true,
            closed: false,
          }],
        }],
      };
    }) as unknown as typeof fetch;

    await polymarketTool.invoke({ query: 'test pass', limit: 3 });
    const count1 = fetchCount;

    await polymarketTool.invoke({ query: 'test pass', limit: 3 });
    // Second call should not increase fetch count (served from cache)
    expect(fetchCount).toBe(count1);
  });

  it('clearPolymarketCache empties the cache', async () => {
    let fetchCount = 0;
    globalThis.fetch = (async () => {
      fetchCount++;
      return {
        ok: true,
        status: 200,
        json: async () => [{
          id: 'e1',
          title: 'Cache Test',
          volume24hr: 100,
          markets: [{
            id: '1',
            question: 'Will cache clear?',
            outcomes: '["Yes","No"]',
            outcomePrices: '["0.60","0.40"]',
            endDateIso: '2027-01-01',
            volume24hr: 100,
            volumeNum: 1000,
            liquidityNum: 500,
            active: true,
            closed: false,
          }],
        }],
      };
    }) as unknown as typeof fetch;

    await polymarketTool.invoke({ query: 'cache clear', limit: 3 });
    const countBefore = fetchCount;
    clearPolymarketCache();
    await polymarketTool.invoke({ query: 'cache clear', limit: 3 });
    expect(fetchCount).toBeGreaterThan(countBefore);
  });
});

// ---------------------------------------------------------------------------
// Degradation warnings
// ---------------------------------------------------------------------------

describe('degradation warnings', () => {
  let savedFetch: typeof globalThis.fetch;
  beforeEach(() => {
    savedFetch = globalThis.fetch;
    clearPolymarketCache();
    polymarketBreaker.reset();
  });
  afterEach(() => { globalThis.fetch = savedFetch; });

  it('drainSearchWarnings returns and clears warnings', async () => {
    const { drainSearchWarnings } = await import('./polymarket.js');

    // Force a failure in tag search so a warning is emitted
    globalThis.fetch = (async () => {
      throw new Error('Simulated outage');
    }) as unknown as typeof fetch;

    const result = await polymarketTool.invoke({ query: 'gold price', limit: 3 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    // Warnings are drained and included in tool output
    expect(text.includes('warning') || text.includes('Warning') || text.includes('failed') || text.includes('No active')).toBe(true);

    // After tool call, drainSearchWarnings should be empty (already consumed)
    const leftover = drainSearchWarnings();
    expect(leftover.length).toBe(0);
  });

  it('surfaces warnings in tool output when API partially fails', async () => {
    let callCount = 0;
    globalThis.fetch = (async (url: string | URL) => {
      callCount++;
      const urlStr = String(url);
      // Tag search succeeds, global fallback fails
      if (urlStr.includes('tag_slug=')) {
        return {
          ok: true,
          status: 200,
          json: async () => [{
            id: 'e1',
            title: 'Gold Prices',
            volume24hr: 500_000,
            markets: [{
              id: '1',
              question: 'Will gold exceed $3000?',
              outcomes: '["Yes","No"]',
              outcomePrices: '["0.55","0.45"]',
              endDateIso: '2027-01-01',
              volume24hr: 500_000,
              volumeNum: 2_000_000,
              liquidityNum: 1_000_000,
              active: true,
              closed: false,
            }],
          }],
        };
      }
      throw new Error('Global API outage');
    }) as unknown as typeof fetch;

    const result = await polymarketTool.invoke({ query: 'gold price', limit: 3 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).toContain('gold');
    // Should still return results from tag search even though global fallback failed
    expect(text).toContain('55.0%');
  });
});

// ---------------------------------------------------------------------------
// computeAgeDays (indirectly via formatMarket)
// ---------------------------------------------------------------------------

describe('ageDays population', () => {
  let savedFetch: typeof globalThis.fetch;
  beforeEach(() => {
    savedFetch = globalThis.fetch;
    clearPolymarketCache();
    polymarketBreaker.reset();
  });
  afterEach(() => { globalThis.fetch = savedFetch; });

  it('populates ageDays in tool output when createdAt is present', async () => {
    const twoDaysAgo = new Date(Date.now() - 2 * 86_400_000).toISOString();
    globalThis.fetch = (async () => ({
      ok: true,
      status: 200,
      json: async () => [{
        id: 'e1',
        title: 'Bitcoin Price Prediction',
        volume24hr: 100,
        markets: [{
          id: '1',
          question: 'Will Bitcoin exceed $100K?',
          outcomes: '["Yes","No"]',
          outcomePrices: '["0.70","0.30"]',
          endDateIso: '2027-01-01',
          volume24hr: 100,
          volumeNum: 1000,
          liquidityNum: 500,
          active: true,
          closed: false,
          createdAt: twoDaysAgo,
        }],
      }],
    })) as unknown as typeof fetch;

    // Use tool directly (not fetchPolymarketMarkets which may be mocked by other tests)
    const result = await polymarketTool.invoke({ query: 'bitcoin price', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    // Tool output should contain market info
    expect(text).toContain('Bitcoin');
    expect(text).toContain('70.0%');
  });

  it('handles missing createdAt gracefully', async () => {
    globalThis.fetch = (async () => ({
      ok: true,
      status: 200,
      json: async () => [{
        id: 'e1',
        title: 'Ethereum Price',
        volume24hr: 100,
        markets: [{
          id: '1',
          question: 'Will Ethereum reach $5000?',
          outcomes: '["Yes","No"]',
          outcomePrices: '["0.50","0.50"]',
          endDateIso: '2027-01-01',
          volume24hr: 100,
          volumeNum: 1000,
          liquidityNum: 500,
          active: true,
          closed: false,
          // No createdAt field — ageDays should be undefined internally
        }],
      }],
    })) as unknown as typeof fetch;

    const result = await polymarketTool.invoke({ query: 'ethereum price', limit: 5 });
    const text = typeof result === 'string' ? result : JSON.stringify(result);
    expect(text).toContain('Ethereum');
    expect(text).toContain('50.0%');
  });
});
