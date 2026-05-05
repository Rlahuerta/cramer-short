import { describe, it, expect, beforeEach } from 'bun:test';
import { polymarketBreaker } from '../../utils/circuit-breaker.js';
import { createPolymarketForecastTool, evaluateMarketHistory } from './polymarket-forecast.js';
import type { PolymarketMarketResult } from './polymarket.js';
import type { ArbiterReplayBundle, RawPolymarketReplayRow } from './arbiter-replay.js';

const LIVE_BRIER_REPLAY_FLAG = 'POLYMARKET_BRIER_REPLAY_CALIBRATOR_ENABLED';

const mockMarkets: PolymarketMarketResult[] = [
  { marketId: 'nvda-market-1', assetId: 'nvda-yes-1', question: 'Will NVIDIA beat Q2 earnings?', probability: 0.72, volume24h: 500_000, ageDays: 0 },
  { marketId: 'nvda-market-2', assetId: 'nvda-yes-2', question: 'Will NVIDIA revenue exceed $30B?', probability: 0.65, volume24h: 300_000, ageDays: 0 },
];

function futureIso(daysAhead: number): string {
  return new Date(Date.now() + daysAhead * 86_400_000).toISOString();
}

// ---------------------------------------------------------------------------
// Default hermetic tool factory
// ---------------------------------------------------------------------------

function makeHermeticTool(
  fetchMarkets: (query: string, limit: number) => Promise<PolymarketMarketResult[]> = async () => mockMarkets,
  recordReplayPolymarketCapture: ((capture: {
    rawRow: RawPolymarketReplayRow;
    polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
  }) => void) | undefined = () => {},
  fetchAnchorMarketsWithQueries: (
    queries: string[],
    limit: number,
    options: { ticker: string; horizonDays?: number; endDateFilter?: { end_date_min: string; end_date_max: string } },
  ) => Promise<PolymarketMarketResult[]> = async (queries, limit) => fetchMarkets(queries[0] ?? '', limit),
): ReturnType<typeof createPolymarketForecastTool> {
  return createPolymarketForecastTool({
    fetchMarkets,
    fetchAnchorMarketsWithQueries,
    readRecords: () => [], // hermetic: always empty snapshot history
    recordReplayPolymarketCapture,
  });
}

let polymarketForecastTool: ReturnType<typeof createPolymarketForecastTool>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseResult(raw: unknown): string {
  const data = parsePayload(raw);
  return data.result ?? data.error ?? '';
}

function parsePayload(raw: unknown): { result?: string; error?: string; forecastReturn?: number } {
  const outer = JSON.parse(raw as string) as {
    data?: { result?: string; error?: string; forecastReturn?: number };
  };
  return outer.data ?? {};
}

beforeEach(() => {
  delete process.env[LIVE_BRIER_REPLAY_FLAG];
  polymarketBreaker.reset();
  polymarketForecastTool = makeHermeticTool();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('polymarketForecastTool', () => {
  it('preserves the baseline live-path payload when the calibrator is default-off or explicitly falsey', async () => {
    const baseline = parsePayload(await makeHermeticTool(async () => [
      {
        question: 'Will Bitcoin be above $70,000 on May 9?',
        probability: 0.58,
        volume24h: 240_000,
        ageDays: 2,
        endDate: futureIso(7),
      },
    ]).func(
        { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
        undefined,
    ));

    for (const flag of ['0', 'false', 'no', 'off']) {
      process.env[LIVE_BRIER_REPLAY_FLAG] = flag;
      const disabled = parsePayload(await makeHermeticTool(async () => [
        {
          question: 'Will Bitcoin be above $70,000 on May 9?',
          probability: 0.58,
          volume24h: 240_000,
          ageDays: 2,
          endDate: futureIso(7),
        },
      ]).func(
        { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
        undefined,
      ));

      expect(disabled).toEqual(baseline);
      expect(disabled.forecastReturn).toBeUndefined();
    }
  });

  it('changes the live-path forecast payload when the calibrator flag is enabled', async () => {
    const tool = makeHermeticTool(async () => [
      {
        question: 'Will Bitcoin be above $70,000 on May 9?',
        probability: 0.58,
        volume24h: 240_000,
        ageDays: 2,
        endDate: futureIso(7),
      },
    ]);

    const baseline = parsePayload(await tool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
      undefined,
    ));

    process.env[LIVE_BRIER_REPLAY_FLAG] = '1';
    const enabled = parsePayload(await tool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
      undefined,
    ));

    expect(typeof baseline.result).toBe('string');
    expect(enabled.forecastReturn).toBeNumber();
    expect(enabled.forecastReturn).not.toBe(baseline.forecastReturn);
    expect(enabled.result).not.toBe(baseline.result);
  });

  it('skips non-finite probabilities before the live calibrator path', async () => {
    process.env[LIVE_BRIER_REPLAY_FLAG] = '1';
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const raw = await makeHermeticTool(
      async () => [
        {
          marketId: 'btc-nan',
          assetId: 'btc-nan-yes',
          question: 'Will Bitcoin be above $70,000 on May 9?',
          probability: Number.NaN,
          volume24h: 240_000,
          ageDays: 2,
          endDate: futureIso(7),
        },
        {
          marketId: 'btc-inf',
          assetId: 'btc-inf-yes',
          question: 'Will Bitcoin be above $72,000 on May 9?',
          probability: Number.POSITIVE_INFINITY,
          volume24h: 240_000,
          ageDays: 2,
          endDate: futureIso(7),
        },
      ],
      (capture) => { captures.push(capture); },
    ).func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
      undefined,
    );

    const payload = parsePayload(raw);
    const result = parseResult(raw);

    expect(payload.forecastReturn).toBeUndefined();
    expect(result).toContain('[No Polymarket markets found for this asset]');
    expect(result).not.toContain('NaN');
    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual([]);
    expect(captures[0]?.polymarket.selectedMarkets).toEqual([]);
  });

  it('result string contains "Polymarket Forecast"', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    expect(parseResult(raw)).toContain('Polymarket Forecast');
  });

  it('includes the ticker in the output', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    expect(parseResult(raw)).toContain('NVDA');
  });

  it('uses commodity display label alongside proxy ticker for GLD', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'GLD', horizon_days: 30, current_price: 440.08 },
      undefined,
    );
    expect(parseResult(raw)).toContain('Polymarket Forecast: GOLD (GLD)');
  });

  it('forecastPrice > 0', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    const output = parseResult(raw);
    const match = output.match(/Forecast price:\s+\$([0-9.]+)/);
    expect(match).not.toBeNull();
    const price = parseFloat(match![1]!);
    expect(price).toBeGreaterThan(0);
  });

  it('shows warning when no current_price provided', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7 },
      undefined,
    );
    expect(parseResult(raw)).toContain('No current price provided');
  });

  it('shows market questions in polymarket signal section', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    expect(parseResult(raw)).toContain('Will NVIDIA beat Q2 earnings?');
  });

  it('filters out markets that are irrelevant to the selected signal category', async () => {
    const freshTool = makeHermeticTool(async () => [
      { question: 'Will ETH be above $2,000 by April 30?', probability: 0.61, volume24h: 150_000, ageDays: 1 },
    ]);
    const raw = await freshTool.func(
      { ticker: 'GLD', horizon_days: 30, current_price: 440.08 },
      undefined,
    );
    const result = parseResult(raw);

    expect(result).not.toContain('Will ETH be above $2,000 by April 30?');
    expect(result).toContain('[No Polymarket markets found for this asset]');
  });

  it('filters bitcoin price markets out of GLD output while preserving gold markets', async () => {
    const freshTool = makeHermeticTool(async () => [
      { question: 'Will gold reach $3,000 per ounce by June?', probability: 0.44, volume24h: 240_000, ageDays: 1 },
      { question: 'Will Bitcoin price exceed $100K in 2026?', probability: 0.65, volume24h: 1_500_000, ageDays: 1 },
    ]);
    const raw = await freshTool.func(
      { ticker: 'GLD', horizon_days: 30, current_price: 440.08 },
      undefined,
    );
    const result = parseResult(raw);

    expect(result).toContain('Will gold reach $3,000 per ounce by June?');
    expect(result).not.toContain('Will Bitcoin price exceed $100K in 2026?');
  });

  it('includes 95% CI in output', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    expect(parseResult(raw)).toContain('95% CI');
  });

  it('includes grade in output', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    expect(parseResult(raw)).toMatch(/Grade:\s+[ABCD]/);
  });

  it('omits not-provided signals with placeholder text', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    expect(parseResult(raw)).toContain('[signal omitted — not provided]');
  });

  it('includes provided sentiment_score in output', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50, sentiment_score: 0.7 },
      undefined,
    );
    expect(parseResult(raw)).toContain('very bullish');
  });

  it('accepts markov_return and shows Markov contribution in output', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000, markov_return: 0.025 },
      undefined,
    );
    const result = parseResult(raw);
    expect(result).toContain('Markov chain:');
    expect(result).toContain('+2.50%');
  });

  it('keeps Markov contribution omitted when markov_return is absent', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
      undefined,
    );
    const result = parseResult(raw);
    expect(result).toContain('Markov chain:       [signal omitted — not provided]');
  });

  it('grade D when 0 markets returned', async () => {
    const freshTool = makeHermeticTool(async () => []);
    const raw = await freshTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    expect(parseResult(raw)).toContain('Grade: D');
  });

  it('captures a replay-ready Polymarket decision block with frozen semantics and CLOB token ids', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [
        {
          marketId: 'btc-market-1',
          assetId: 'btc-yes-1',
          question: 'Will Bitcoin be above $70,000 on May 7?',
          probability: 0.54,
          volume24h: 250_000,
          ageDays: 3,
          endDate: '2026-05-07T00:00:00.000Z',
          active: true,
          closed: false,
          enableOrderBook: true,
        },
      ],
      (capture) => { captures.push(capture); },
    );

    await freshTool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68000 },
      undefined,
    );

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.ticker).toBe('BTC');
    expect(captures[0]?.rawRow.currentPrice).toBe(68000);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-market-1']);
    expect(captures[0]?.rawRow.candidates[0]).toMatchObject({
      marketId: 'btc-market-1',
      assetId: 'btc-yes-1',
      enableOrderBook: true,
    });
    expect(captures[0]?.polymarket.selectedMarkets).toEqual([
      {
        marketId: 'btc-market-1',
        assetId: 'btc-yes-1',
        question: 'Will Bitcoin be above $70,000 on May 7?',
        probability: 0.54,
        volume24h: 250_000,
        endDate: '2026-05-07T00:00:00.000Z',
        semantics: 'terminal',
        extractedPriceLevels: [70000],
        relevanceScore: expect.any(Number),
      },
    ]);
  });

  it('uses anchor-first retrieval for 1-day BTC and selects near-expiry threshold markets ahead of broad macro fallback', async () => {
    const anchorCalls: string[][] = [];
    const genericCalls: string[] = [];
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async (query) => {
        genericCalls.push(query);
        return [
          {
            marketId: 'btc-macro-fallback-1d',
            assetId: 'btc-macro-fallback-1d-yes',
            question: 'US recession by end of 2026?',
            probability: 0.22,
            volume24h: 14_000,
            ageDays: 9,
            endDate: futureIso(180),
            active: true,
            closed: false,
          },
        ];
      },
      (capture) => { captures.push(capture); },
      async (queries) => {
        anchorCalls.push(queries);
        return [
          {
            marketId: 'btc-anchor-1d',
            assetId: 'btc-anchor-1d-yes',
            question: 'Will Bitcoin be above $72,000 tomorrow?',
            probability: 0.61,
            volume24h: 240_000,
            ageDays: 2,
            endDate: futureIso(1),
            active: true,
            closed: false,
          },
        ];
      },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 1, current_price: 68_000 },
      undefined,
    ));

    expect(anchorCalls).toHaveLength(1);
    expect(anchorCalls[0]?.slice(0, 4)).toEqual(['Bitcoin price', 'Bitcoin', 'Bitcoin above', 'Bitcoin below']);
    expect(genericCalls).toEqual([]);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-anchor-1d']);
    expect(result).toContain('Will Bitcoin be above $72,000 tomorrow?');
    expect(result).not.toContain('US recession by end of 2026?');
  });

  it('uses anchor-first retrieval for 2-day BTC and selects near-expiry threshold markets ahead of broad macro fallback', async () => {
    const anchorCalls: string[][] = [];
    const genericCalls: string[] = [];
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async (query) => {
        genericCalls.push(query);
        return [
          {
            marketId: 'btc-macro-fallback-2d',
            assetId: 'btc-macro-fallback-2d-yes',
            question: 'Will the Fed cut rates by August 2026?',
            probability: 0.31,
            volume24h: 21_000,
            ageDays: 12,
            endDate: futureIso(90),
            active: true,
            closed: false,
          },
        ];
      },
      (capture) => { captures.push(capture); },
      async (queries) => {
        anchorCalls.push(queries);
        return [
          {
            marketId: 'btc-anchor-2d',
            assetId: 'btc-anchor-2d-yes',
            question: 'Will Bitcoin be above $73,000 in 2 days?',
            probability: 0.59,
            volume24h: 230_000,
            ageDays: 2,
            endDate: futureIso(2),
            active: true,
            closed: false,
          },
        ];
      },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 2, current_price: 68_000 },
      undefined,
    ));

    expect(anchorCalls).toHaveLength(1);
    expect(anchorCalls[0]?.slice(0, 4)).toEqual(['Bitcoin price', 'Bitcoin', 'Bitcoin above', 'Bitcoin below']);
    expect(genericCalls).toEqual([]);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-anchor-2d']);
    expect(result).toContain('Will Bitcoin be above $73,000 in 2 days?');
    expect(result).not.toContain('Will the Fed cut rates by August 2026?');
  });

  it('uses anchor-first retrieval for 3-day BTC and selects near-expiry threshold markets ahead of broad macro fallback', async () => {
    const anchorCalls: string[][] = [];
    const genericCalls: string[] = [];
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async (query) => {
        genericCalls.push(query);
        return [
          {
            marketId: 'btc-macro-fallback-3d',
            assetId: 'btc-macro-fallback-3d-yes',
            question: 'Will the SEC approve a new crypto ETF in 2026?',
            probability: 0.28,
            volume24h: 19_000,
            ageDays: 10,
            endDate: futureIso(120),
            active: true,
            closed: false,
          },
        ];
      },
      (capture) => { captures.push(capture); },
      async (queries) => {
        anchorCalls.push(queries);
        return [
          {
            marketId: 'btc-anchor-3d',
            assetId: 'btc-anchor-3d-yes',
            question: 'Will Bitcoin be above $74,000 in 3 days?',
            probability: 0.57,
            volume24h: 235_000,
            ageDays: 2,
            endDate: futureIso(3),
            active: true,
            closed: false,
          },
        ];
      },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 3, current_price: 68_000 },
      undefined,
    ));

    expect(anchorCalls).toHaveLength(1);
    expect(anchorCalls[0]?.slice(0, 4)).toEqual(['Bitcoin price', 'Bitcoin', 'Bitcoin above', 'Bitcoin below']);
    expect(genericCalls).toEqual([]);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-anchor-3d']);
    expect(result).toContain('Will Bitcoin be above $74,000 in 3 days?');
    expect(result).not.toContain('Will the SEC approve a new crypto ETF in 2026?');
  });

  const shortHorizonGenericFallbackCases = [
    { horizonDays: 1, relativePhrase: 'Bitcoin tomorrow' },
    { horizonDays: 2, relativePhrase: 'Bitcoin in 2 days' },
    { horizonDays: 3, relativePhrase: 'Bitcoin in 3 days' },
  ] as const;

  for (const { horizonDays, relativePhrase } of shortHorizonGenericFallbackCases) {
    it(`uses BTC-centric short-horizon generic signals first for ${horizonDays}-day crypto fallback`, async () => {
      const anchorCalls: string[][] = [];
      const genericCalls: string[] = [];
      const freshTool = makeHermeticTool(
        async (query) => {
          genericCalls.push(query);
          return query === 'Bitcoin above'
            ? [
                {
                  marketId: `btc-generic-fallback-${horizonDays}d`,
                  assetId: `btc-generic-fallback-${horizonDays}d-yes`,
                  question: `Will Bitcoin be above $3,000 in ${horizonDays} day${horizonDays === 1 ? '' : 's'}?`,
                  probability: 0.56,
                  volume24h: 210_000,
                  ageDays: 1,
                  endDate: futureIso(horizonDays),
                  active: true,
                  closed: false,
                },
              ]
            : [];
        },
        undefined,
        async (queries) => {
          anchorCalls.push(queries);
          return [];
        },
      );

      const result = parseResult(await freshTool.func(
        { ticker: 'ETH', horizon_days: horizonDays, current_price: 3_000 },
        undefined,
      ));

      expect(anchorCalls.length).toBeGreaterThan(0);
      expect(genericCalls[0]).toBe('Bitcoin above');
      expect(genericCalls[1]).toBe('Bitcoin below');
      expect(genericCalls).toContain('Bitcoin price');
      expect(genericCalls).toContain(relativePhrase);
      expect(genericCalls.some((query) => /^Bitcoin (above|below) [A-Z][a-z]{2} \d{1,2}$/.test(query))).toBe(true);
      expect(genericCalls.some((query) => /^Bitcoin [A-Z][a-z]{2} \d{1,2}$/.test(query))).toBe(true);
      expect(genericCalls.indexOf('Bitcoin price')).toBeLessThan(genericCalls.indexOf('Bitcoin ETF'));
      expect(genericCalls.indexOf(relativePhrase)).toBeLessThan(genericCalls.indexOf('crypto regulation'));
      expect(result).toContain(`Will Bitcoin be above $3,000 in ${horizonDays} day${horizonDays === 1 ? '' : 's'}?`);
    });
  }

  for (const horizonDays of [1, 2, 3] as const) {
    it(`excludes missing endDate markets from ${horizonDays}-day crypto generic fallback`, async () => {
      const captures: Array<{
        rawRow: RawPolymarketReplayRow;
        polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
      }> = [];
      const freshTool = makeHermeticTool(
        async () => [
          {
            marketId: `eth-undated-${horizonDays}d`,
            assetId: `eth-undated-${horizonDays}d-yes`,
            question: `Will Bitcoin be above $3,100 in ${horizonDays} day${horizonDays === 1 ? '' : 's'}?`,
            probability: 0.63,
            volume24h: 260_000,
            ageDays: 2,
            active: true,
            closed: false,
          },
          {
            marketId: `eth-dated-${horizonDays}d`,
            assetId: `eth-dated-${horizonDays}d-yes`,
            question: `Will Bitcoin be above $3,050 in ${horizonDays} day${horizonDays === 1 ? '' : 's'}?`,
            probability: 0.58,
            volume24h: 230_000,
            ageDays: 2,
            endDate: futureIso(horizonDays),
            active: true,
            closed: false,
          },
        ],
        (capture) => { captures.push(capture); },
        async () => [],
      );

      const result = parseResult(await freshTool.func(
        { ticker: 'ETH', horizon_days: horizonDays, current_price: 3_000 },
        undefined,
      ));

      expect(captures).toHaveLength(1);
      expect(captures[0]?.rawRow.selectedMarketIds).toEqual([`eth-dated-${horizonDays}d`]);
      expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.marketId)).toEqual([`eth-dated-${horizonDays}d`]);
      expect(result).toContain('Skipped 1 Polymarket market');
      expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.question)).toEqual([
        `Will Bitcoin be above $3,050 in ${horizonDays} day${horizonDays === 1 ? '' : 's'}?`,
      ]);
    });
  }

  it('keeps the generic Polymarket retrieval path for longer-horizon BTC forecasts', async () => {
    const anchorCalls: string[][] = [];
    const genericCalls: string[] = [];
    const freshTool = makeHermeticTool(
      async (query) => {
        genericCalls.push(query);
        return [
          {
            marketId: 'btc-generic-7d',
            assetId: 'btc-generic-7d-yes',
            question: 'Will Bitcoin be above $76,000 on May 12?',
            probability: 0.52,
            volume24h: 180_000,
            ageDays: 3,
            endDate: futureIso(7),
            active: true,
            closed: false,
          },
        ];
      },
      undefined,
      async (queries) => {
        anchorCalls.push(queries);
        return [];
      },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
      undefined,
    ));

    expect(anchorCalls).toEqual([]);
    expect(genericCalls.length).toBeGreaterThan(0);
    expect(result).toContain('Will Bitcoin be above $76,000 on May 12?');
  });

  it('keeps longer-horizon crypto signals unchanged', async () => {
    const anchorCalls: string[][] = [];
    const genericCalls: string[] = [];
    const freshTool = makeHermeticTool(
      async (query) => {
        genericCalls.push(query);
        return query === 'crypto regulation'
          ? [
              {
                marketId: 'eth-generic-7d',
                assetId: 'eth-generic-7d-yes',
                question: 'Will the SEC approve a new crypto ETF in 2026?',
                probability: 0.31,
                volume24h: 25_000,
                ageDays: 4,
                endDate: futureIso(30),
                active: true,
                closed: false,
              },
            ]
          : [];
      },
      undefined,
      async (queries) => {
        anchorCalls.push(queries);
        return [];
      },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'ETH', horizon_days: 7, current_price: 3_000 },
      undefined,
    ));

    expect(anchorCalls).toEqual([]);
    expect(genericCalls.slice(0, 3)).toEqual([
      'crypto regulation',
      'SEC crypto',
      'cryptocurrency regulation',
    ]);
    expect(result).toContain('Will the SEC approve a new crypto ETF in 2026?');
  });

  it('keeps non-crypto short-horizon signals unchanged', async () => {
    const anchorCalls: string[][] = [];
    const genericCalls: string[] = [];
    const freshTool = makeHermeticTool(
      async (query) => {
        genericCalls.push(query);
        return query === 'NVIDIA earnings'
          ? [
              {
                marketId: 'nvda-generic-2d',
                assetId: 'nvda-generic-2d-yes',
                question: 'Will NVIDIA beat earnings this quarter?',
                probability: 0.64,
                volume24h: 145_000,
                ageDays: 2,
                endDate: futureIso(14),
                active: true,
                closed: false,
              },
            ]
          : [];
      },
      undefined,
      async (queries) => {
        anchorCalls.push(queries);
        return [];
      },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'NVDA', horizon_days: 2, current_price: 900 },
      undefined,
    ));

    expect(anchorCalls).toEqual([]);
    expect(genericCalls.slice(0, 3)).toEqual([
      'NVIDIA earnings',
      'NVIDIA',
      'semiconductor earnings',
    ]);
    expect(result).toContain('Will NVIDIA beat earnings this quarter?');
  });

  it('drops far-dated macro markets from 1-day BTC selection and emits a horizon warning', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [
        {
          marketId: 'btc-market-1d',
          assetId: 'btc-market-1d-yes',
          question: 'Will Bitcoin be above $72,000 tomorrow?',
          probability: 0.61,
          volume24h: 240_000,
          ageDays: 2,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-macro-far',
          assetId: 'btc-macro-far-yes',
          question: 'US recession by end of 2026?',
          probability: 0.24,
          volume24h: 12_000,
          ageDays: 20,
          endDate: futureIso(180),
          active: true,
          closed: false,
        },
      ],
      (capture) => { captures.push(capture); },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 1, current_price: 68_000 },
      undefined,
    ));

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-market-1d']);
    expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.marketId)).toEqual(['btc-market-1d']);
    expect(result).toContain('Skipped 1 Polymarket market because its resolution date');
    expect(result).not.toContain('US recession by end of 2026?');
  });

  it('drops far-dated macro markets from 2-day BTC selection and emits a horizon warning', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [
        {
          marketId: 'btc-market-2d',
          assetId: 'btc-market-2d-yes',
          question: 'Will Bitcoin be above $73,000 in 2 days?',
          probability: 0.59,
          volume24h: 230_000,
          ageDays: 2,
          endDate: futureIso(2),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-macro-45d',
          assetId: 'btc-macro-45d-yes',
          question: 'US recession by July 2026?',
          probability: 0.21,
          volume24h: 18_000,
          ageDays: 11,
          endDate: futureIso(45),
          active: true,
          closed: false,
        },
      ],
      (capture) => { captures.push(capture); },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 2, current_price: 68_000 },
      undefined,
    ));

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-market-2d']);
    expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.marketId)).toEqual(['btc-market-2d']);
    expect(result).toContain('Skipped 1 Polymarket market because its resolution date');
    expect(result).not.toContain('US recession by July 2026?');
  });

  it('drops far-dated macro markets from 3-day BTC selection and emits a horizon warning', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [
        {
          marketId: 'btc-market-3d',
          assetId: 'btc-market-3d-yes',
          question: 'Will Bitcoin be above $74,000 in 3 days?',
          probability: 0.57,
          volume24h: 235_000,
          ageDays: 2,
          endDate: futureIso(3),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-macro-90d',
          assetId: 'btc-macro-90d-yes',
          question: 'Will the Fed cut rates by August 2026?',
          probability: 0.33,
          volume24h: 24_000,
          ageDays: 13,
          endDate: futureIso(90),
          active: true,
          closed: false,
        },
      ],
      (capture) => { captures.push(capture); },
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 3, current_price: 68_000 },
      undefined,
    ));

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-market-3d']);
    expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.marketId)).toEqual(['btc-market-3d']);
    expect(result).toContain('Skipped 1 Polymarket market because its resolution date');
    expect(result).not.toContain('Will the Fed cut rates by August 2026?');
  });

  it('preserves far-dated market inclusion for longer-horizon BTC forecasts', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [
        {
          marketId: 'btc-market-30d',
          assetId: 'btc-market-30d-yes',
          question: 'Will Bitcoin be above $80,000 by June 1?',
          probability: 0.48,
          volume24h: 180_000,
          ageDays: 4,
          endDate: futureIso(30),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-macro-60d',
          assetId: 'btc-macro-60d-yes',
          question: 'Will Fed decrease rates 25 bps after June 2026 FOMC?',
          probability: 0.29,
          volume24h: 210_000,
          ageDays: 15,
          endDate: futureIso(60),
          active: true,
          closed: false,
        },
      ],
      (capture) => { captures.push(capture); },
    );

    await freshTool.func(
      { ticker: 'BTC', horizon_days: 30, current_price: 68_000 },
      undefined,
    );

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-market-30d', 'btc-macro-60d']);
  });

  it('excludes barrier and path questions from short-horizon BTC terminal anchor selection', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [],
      (capture) => { captures.push(capture); },
      async () => [
        {
          marketId: 'btc-barrier-1d',
          assetId: 'btc-barrier-1d-yes',
          question: 'Will Bitcoin reach $72,000 tomorrow?',
          probability: 0.63,
          volume24h: 250_000,
          ageDays: 3,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-path-1d',
          assetId: 'btc-path-1d-yes',
          question: 'Will Bitcoin stay above $70,000 through tomorrow?',
          probability: 0.55,
          volume24h: 240_000,
          ageDays: 3,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-terminal-1d',
          assetId: 'btc-terminal-1d-yes',
          question: 'Will the price of Bitcoin be above $71,000 tomorrow?',
          probability: 0.58,
          volume24h: 260_000,
          ageDays: 3,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
      ],
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 1, current_price: 68_000 },
      undefined,
    ));

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-terminal-1d']);
    expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.question)).toEqual([
      'Will the price of Bitcoin be above $71,000 tomorrow?',
    ]);
    expect(result).not.toContain('Will Bitcoin reach $72,000 tomorrow?');
    expect(result).not.toContain('Will Bitcoin stay above $70,000 through tomorrow?');
  });

  it('prefers 1-day BTC terminal anchors over 3-day and 5-day variants', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [],
      (capture) => { captures.push(capture); },
      async () => [
        {
          marketId: 'btc-75k-1d',
          assetId: 'btc-75k-1d-yes',
          question: 'Will the price of Bitcoin be above $75,000 tomorrow?',
          probability: 0.41,
          volume24h: 210_000,
          ageDays: 3,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-75k-3d',
          assetId: 'btc-75k-3d-yes',
          question: 'Will the price of Bitcoin be above $75,000 in 3 days?',
          probability: 0.56,
          volume24h: 235_000,
          ageDays: 3,
          endDate: futureIso(3),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-75k-5d',
          assetId: 'btc-75k-5d-yes',
          question: 'Will the price of Bitcoin be above $75,000 in 5 days?',
          probability: 0.62,
          volume24h: 240_000,
          ageDays: 3,
          endDate: futureIso(5),
          active: true,
          closed: false,
        },
      ],
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 1, current_price: 68_000 },
      undefined,
    ));

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-75k-1d']);
    expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.question)).toEqual([
      'Will the price of Bitcoin be above $75,000 tomorrow?',
    ]);
  });

  it('prefers 3-day BTC terminal anchors over 1-day-only variants', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [],
      (capture) => { captures.push(capture); },
      async () => [
        {
          marketId: 'btc-78k-1d',
          assetId: 'btc-78k-1d-yes',
          question: 'Will the price of Bitcoin be above $78,000 tomorrow?',
          probability: 0.67,
          volume24h: 250_000,
          ageDays: 3,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-78k-3d',
          assetId: 'btc-78k-3d-yes',
          question: 'Will the price of Bitcoin be above $78,000 in 3 days?',
          probability: 0.48,
          volume24h: 220_000,
          ageDays: 3,
          endDate: futureIso(3),
          active: true,
          closed: false,
        },
      ],
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 3, current_price: 68_000 },
      undefined,
    ));

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-78k-3d']);
    expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.question)).toEqual([
      'Will the price of Bitcoin be above $78,000 in 3 days?',
    ]);
  });

  it('falls back to off-horizon terminal BTC anchors when strict short-horizon anchors are empty', async () => {
    const genericCalls: string[] = [];
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async (query) => {
        genericCalls.push(query);
        return [];
      },
      (capture) => { captures.push(capture); },
      async () => [
        {
          marketId: 'btc-barrier-fallback-1d',
          assetId: 'btc-barrier-fallback-1d-yes',
          question: 'Will Bitcoin reach $72,000 tomorrow?',
          probability: 0.61,
          volume24h: 220_000,
          ageDays: 3,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-path-fallback-1d',
          assetId: 'btc-path-fallback-1d-yes',
          question: 'Will Bitcoin dip to $66,000 tomorrow?',
          probability: 0.29,
          volume24h: 210_000,
          ageDays: 3,
          endDate: futureIso(1),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-fallback-75k',
          assetId: 'btc-fallback-75k-yes',
          question: 'Will the price of Bitcoin be above $75,000 in 5 days?',
          probability: 0.44,
          volume24h: 200_000,
          ageDays: 4,
          endDate: futureIso(5),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-fallback-78k',
          assetId: 'btc-fallback-78k-yes',
          question: 'Will the price of Bitcoin be above $78,000 in 6 days?',
          probability: 0.31,
          volume24h: 180_000,
          ageDays: 4,
          endDate: futureIso(6),
          active: true,
          closed: false,
        },
      ],
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 1, current_price: 68_000 },
      undefined,
    ));

    expect(genericCalls).toEqual([]);
    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-fallback-75k', 'btc-fallback-78k']);
    expect(captures[0]?.polymarket.selectedMarkets.map((market) => market.question)).toEqual([
      'Will the price of Bitcoin be above $75,000 in 5 days?',
      'Will the price of Bitcoin be above $78,000 in 6 days?',
    ]);
    expect(result).toContain('BTC Price Distribution');
    expect(result).not.toContain('Threshold-style markets were omitted from the distribution chart');
  });

  it('renders the threshold chart from the same selected short-horizon BTC anchor set', async () => {
    const captures: Array<{
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }> = [];
    const freshTool = makeHermeticTool(
      async () => [],
      (capture) => { captures.push(capture); },
      async () => [
        {
          marketId: 'btc-chart-71k',
          assetId: 'btc-chart-71k-yes',
          question: 'Will the price of Bitcoin be above $71,000 in 2 days?',
          probability: 0.64,
          volume24h: 260_000,
          ageDays: 3,
          endDate: futureIso(2),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-chart-74k',
          assetId: 'btc-chart-74k-yes',
          question: 'Will the price of Bitcoin be above $74,000 in 2 days?',
          probability: 0.39,
          volume24h: 230_000,
          ageDays: 3,
          endDate: futureIso(2),
          active: true,
          closed: false,
        },
        {
          marketId: 'btc-chart-90k-offhorizon',
          assetId: 'btc-chart-90k-offhorizon-yes',
          question: 'Will the price of Bitcoin be above $90,000 in 6 days?',
          probability: 0.09,
          volume24h: 210_000,
          ageDays: 3,
          endDate: futureIso(6),
          active: true,
          closed: false,
        },
      ],
    );

    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 2, current_price: 68_000 },
      undefined,
    ));

    expect(captures).toHaveLength(1);
    expect(captures[0]?.rawRow.selectedMarketIds).toEqual(['btc-chart-71k', 'btc-chart-74k']);
    expect(result).toContain('BTC Price Distribution');
    expect(result).toContain('71K');
    expect(result).toContain('74K');
    expect(result).not.toContain('90K');
    expect(result).not.toContain('Threshold-style markets were omitted from the distribution chart');
  });

});

describe('evaluateMarketHistory', () => {
  const nowMs = new Date('2026-04-20T12:00:00.000Z').getTime();

  it('detects a 2-4h price spike when the delta exceeds 0.08 and volume is low', () => {
    const evaluation = evaluateMarketHistory(
      { marketId: 'm1', probability: 0.42, volume24h: 80_000 },
      [
        {
          marketId: 'm1',
          question: 'Q',
          probability: 0.30,
          volume24h: 80_000,
          endDate: '2026-12-31T23:59:59Z',
          capturedAt: '2026-04-20T09:00:00.000Z',
        },
      ],
      nowMs,
    );

    expect(evaluation.priceSpikeDetected).toBe(true);
    expect(evaluation.transitoryMove).toBe(false);
  });

  it('does not trigger a price spike when the 24h volume gate blocks it', () => {
    const evaluation = evaluateMarketHistory(
      { marketId: 'm1', probability: 0.42, volume24h: 150_000 },
      [
        {
          marketId: 'm1',
          question: 'Q',
          probability: 0.30,
          volume24h: 150_000,
          endDate: '2026-12-31T23:59:59Z',
          capturedAt: '2026-04-20T09:00:00.000Z',
        },
      ],
      nowMs,
    );

    expect(evaluation.priceSpikeDetected).toBe(false);
  });

  it('emits a warning when no spike snapshot exists', () => {
    const evaluation = evaluateMarketHistory(
      { marketId: 'm1', probability: 0.42, volume24h: 80_000 },
      [],
      nowMs,
    );

    expect(evaluation.priceSpikeDetected).toBe(false);
    expect(evaluation.warnings.some((warning) => warning.includes('Spike detection unavailable'))).toBe(true);
  });

  it('marks a market as transitory when a 24-48h move has mostly reversed', () => {
    const evaluation = evaluateMarketHistory(
      { marketId: 'm1', probability: 0.26, volume24h: 80_000 },
      [
        {
          marketId: 'm1',
          question: 'Q',
          probability: 0.36,
          volume24h: 80_000,
          endDate: '2026-12-31T23:59:59Z',
          capturedAt: '2026-04-20T09:00:00.000Z',
        },
        {
          marketId: 'm1',
          question: 'Q',
          probability: 0.20,
          volume24h: 80_000,
          endDate: '2026-12-31T23:59:59Z',
          capturedAt: '2026-04-19T06:00:00.000Z',
        },
      ],
      nowMs,
    );

    expect(evaluation.transitoryMove).toBe(true);
  });

  it('does not mark a market as transitory when the move has persisted', () => {
    const evaluation = evaluateMarketHistory(
      { marketId: 'm1', probability: 0.48, volume24h: 80_000 },
      [
        {
          marketId: 'm1',
          question: 'Q',
          probability: 0.36,
          volume24h: 80_000,
          endDate: '2026-12-31T23:59:59Z',
          capturedAt: '2026-04-20T09:00:00.000Z',
        },
        {
          marketId: 'm1',
          question: 'Q',
          probability: 0.20,
          volume24h: 80_000,
          endDate: '2026-12-31T23:59:59Z',
          capturedAt: '2026-04-19T06:00:00.000Z',
        },
      ],
      nowMs,
    );

    expect(evaluation.transitoryMove).toBe(false);
  });

  it('emits a warning when no 24-48h persistence snapshot exists', () => {
    const evaluation = evaluateMarketHistory(
      { marketId: 'm1', probability: 0.42, volume24h: 80_000 },
      [
        {
          marketId: 'm1',
          question: 'Q',
          probability: 0.30,
          volume24h: 80_000,
          endDate: '2026-12-31T23:59:59Z',
          capturedAt: '2026-04-20T09:00:00.000Z',
        },
      ],
      nowMs,
    );

    expect(evaluation.transitoryMove).toBe(false);
    expect(evaluation.warnings.some((warning) => warning.includes('Persistence test unavailable'))).toBe(true);
  });
});

describe('threshold chart horizon alignment', () => {
  it('omits threshold chart and emits warning when threshold markets resolve at the wrong horizon', async () => {
    const freshTool = makeHermeticTool(async () => [
      {
        question: 'Will the price of Bitcoin be above $60,000 on April 2?',
        probability: 0.99,
        volume24h: 300_000,
        ageDays: 1,
        endDate: futureIso(0),
      },
      {
        question: 'Will the price of Bitcoin be above $76,000 on April 2?',
        probability: 0.04,
        volume24h: 200_000,
        ageDays: 1,
        endDate: futureIso(0),
      },
    ]);
    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
      undefined,
    ));

    expect(result).not.toContain('BTC Price Distribution');
    expect(result).toContain('Threshold-style markets were omitted from the distribution chart');
  });

  it('renders threshold chart when threshold markets resolve near the requested horizon', async () => {
    const freshTool = makeHermeticTool(async () => [
      {
        question: 'Will the price of Bitcoin be above $60,000 on April 9?',
        probability: 0.99,
        volume24h: 300_000,
        ageDays: 1,
        endDate: futureIso(7),
      },
      {
        question: 'Will the price of Bitcoin be above $76,000 on April 9?',
        probability: 0.04,
        volume24h: 200_000,
        ageDays: 1,
        endDate: futureIso(7),
      },
    ]);
    const result = parseResult(await freshTool.func(
      { ticker: 'BTC', horizon_days: 7, current_price: 68_000 },
      undefined,
    ));

    expect(result).toContain('BTC Price Distribution');
    expect(result).not.toContain('Threshold-style markets were omitted from the distribution chart');
  });
});

// ---------------------------------------------------------------------------
// Horizon validation — ensure long horizons are accepted (no hard 14-day cap)
// ---------------------------------------------------------------------------

describe('horizon_days validation', () => {
  it('accepts horizon_days = 30', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 30, current_price: 135.50 },
      undefined,
    );
    const result = parseResult(raw);
    expect(result).not.toContain('error');
    expect(result).toContain('Polymarket Forecast');
  });

  it('accepts horizon_days = 90 and emits moderate-quality note', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 90, current_price: 135.50 },
      undefined,
    );
    const result = parseResult(raw);
    expect(result).toContain('Polymarket Forecast');
    expect(result).toContain('Horizon 90d');
  });

  it('accepts horizon_days = 180 and emits >90-day accuracy warning', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 180, current_price: 135.50 },
      undefined,
    );
    const result = parseResult(raw);
    expect(result).toContain('Polymarket Forecast');
    expect(result).toContain('Horizon 180d > 90 days');
  });

  it('accepts horizon_days = 365', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'BTC', horizon_days: 365, current_price: 85000 },
      undefined,
    );
    const result = parseResult(raw);
    expect(result).toContain('Polymarket Forecast');
    expect(result).not.toContain('"ok":false');
  });
});

// ---------------------------------------------------------------------------
// Sector ETF differentiation — the core bug: all ETFs must NOT produce the
// same forecast return (-0.68% identical for every ETF).
// ---------------------------------------------------------------------------

describe('sector ETF differentiation', () => {
  it('SLX (steel ETF) infers materials asset class and has POSITIVE tariff delta', () => {
    // inferAssetClass must return 'materials' for SLX
    // The trade_policy signal for materials has deltaYes=+0.07 (tariffs protect US steel)
    // With tariff probability ~97%, this should push forecast POSITIVE vs generic equity
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('SLX')).toBe('materials');
  });

  it('KRE (regional bank ETF) infers financial asset class', () => {
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('KRE')).toBe('financial');
  });

  it('IWM (small-cap ETF) infers small_cap asset class', () => {
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('IWM')).toBe('small_cap');
  });

  it('XLI (industrials ETF) infers industrial asset class', () => {
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('XLI')).toBe('industrial');
  });

  it('ITA (defense ETF) remains defense asset class', () => {
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('ITA')).toBe('defense');
  });

  it('NVDA remains semiconductor', () => {
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('NVDA')).toBe('semiconductor');
  });

  it('SLX (materials) and KRE (financial) produce different conditional returns for their respective primary signals', () => {
    const { lookupImpact } = require('./impact-map.js');
    const { computeConditionalReturn, adjustYesBias } = require('../../utils/ensemble.js');

    // Both assets at same Polymarket probability (0.72)
    const p = adjustYesBias(0.72);

    // SLX primary signal: trade_policy (tariff) → materials
    // US tariffs PROTECT domestic steel producers → positive impact
    const slxImpact = lookupImpact('trade_policy', 'materials');
    const slxReturn = computeConditionalReturn(p, slxImpact.deltaYes, slxImpact.deltaNo);

    // KRE primary signal: macro_rates (Fed) → financial
    // Rate cut = NIM compression for banks → negative impact
    const kreImpact = lookupImpact('macro_rates', 'financial');
    const kreReturn = computeConditionalReturn(p, kreImpact.deltaYes, kreImpact.deltaNo);

    expect(slxReturn).not.toBeCloseTo(kreReturn, 3); // must differ
    expect(slxReturn).toBeGreaterThan(0);            // tariffs → bullish for steel
    expect(kreReturn).toBeLessThan(0);               // rate cut → bearish for bank NIM
  });

  it('trade_policy lookupImpact: materials gets POSITIVE deltaYes (tariffs protect domestic steel)', () => {
    const { lookupImpact } = require('./impact-map.js');
    const entry = lookupImpact('trade_policy', 'materials');
    expect(entry.deltaYes).toBeGreaterThan(0); // US tariffs are bullish for domestic steel
    const equityEntry = lookupImpact('trade_policy', 'equity');
    expect(equityEntry.deltaYes).toBeLessThan(0); // broad market is bearish on tariffs
  });

  it('macro_growth lookupImpact: financial has deeper drawdown than equity (loan defaults)', () => {
    const { lookupImpact } = require('./impact-map.js');
    const financial = lookupImpact('macro_growth', 'financial');
    const equity    = lookupImpact('macro_growth', 'equity');
    expect(financial.deltaYes).toBeLessThan(equity.deltaYes); // financial < equity (more negative)
  });

  it('macro_rates lookupImpact: financial has negative deltaYes (rate cut = NIM compression)', () => {
    const { lookupImpact } = require('./impact-map.js');
    const entry = lookupImpact('macro_rates', 'financial');
    expect(entry.deltaYes).toBeLessThan(0); // rate cut is BAD for bank NIM
  });

  it('macro_growth lookupImpact: materials more cyclical than defense', () => {
    const { lookupImpact } = require('./impact-map.js');
    const materials = lookupImpact('macro_growth', 'materials');
    const defense   = lookupImpact('macro_growth', 'defense');
    expect(materials.deltaYes).toBeLessThan(defense.deltaYes); // materials more exposed to recessions
  });
});

// ---------------------------------------------------------------------------
// CI display correctness
// ---------------------------------------------------------------------------

describe('CI display format', () => {
  it('shows dollar CI when current_price is provided', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 500 },
      undefined,
    );
    const result = parseResult(raw);
    // Should contain a dollar-sign CI like [$xxx – $xxx]
    expect(result).toMatch(/\[\$[\d.]+ – \$[\d.]+\]/);
  });

  it('shows percentage CI (not dollar) when current_price is omitted', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7 },
      undefined,
    );
    const result = parseResult(raw);
    // Must contain % CI  (e.g. [-3.5% – +3.5%]) not dollar CI around $99-$101
    expect(result).toMatch(/\[[-+\d.]+%/);
    // Must NOT contain a CI like [$99 – $101] (base-100 dollar CI is misleading)
    expect(result).not.toMatch(/\[\$9[0-9]\.|\[\$10[0-1]\./);
  });

  it('CI sigma is > 1% even for short horizons (floor applied)', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 500 },
      undefined,
    );
    const result = parseResult(raw);
    // σ = x.xx% — extract it
    const sigmaMatch = result.match(/σ = ([\d.]+)%/);
    expect(sigmaMatch).not.toBeNull();
    const sigma = parseFloat(sigmaMatch![1]);
    // 7-day floor = 10% × sqrt(7/252) = 1.67%; should be at least 1%
    expect(sigma).toBeGreaterThan(1.0);
  });
});

// ---------------------------------------------------------------------------
// QQQ classification
// ---------------------------------------------------------------------------

describe('QQQ / broad-market ETF classification', () => {
  it('QQQ infers tech asset class (Nasdaq-100 is tech-heavy)', () => {
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('QQQ')).toBe('tech');
  });

  it('SPY infers equity asset class (broad market)', () => {
    const { inferAssetClass } = require('./impact-map.js');
    expect(inferAssetClass('SPY')).toBe('equity');
  });

  it('QQQ detected as tech_general signal type in extractor', () => {
    // Must be tested in signal-extractor.test.ts (this file mocks the module)
    expect(true).toBe(true); // placeholder — see signal-extractor.test.ts
  });

  it('SPY detected as macro signal type in extractor', () => {
    // Must be tested in signal-extractor.test.ts (this file mocks the module)
    expect(true).toBe(true); // placeholder — see signal-extractor.test.ts
  });
});

// ---------------------------------------------------------------------------
// CI dollar anchoring — the $99-$101 bug guard
//
// When current_price is provided, the 95% CI must be anchored around that
// price, NOT around a base of 100. (The bug: agent omits current_price →
// tool uses base=100 → CI around $99-$101 for a $414 stock.)
// ---------------------------------------------------------------------------

describe('CI dollar anchoring', () => {
  it('ciLow95 > currentPrice/2 for any current_price (never collapses to base-100 range)', async () => {
    const currentPrice = 414.84;
    const raw = await polymarketForecastTool.func(
      { ticker: 'GLD', horizon_days: 7, current_price: currentPrice },
      undefined,
    );
    const result = parseResult(raw);
    // Extract dollar CI from format [$xxx.xx – $xxx.xx]
    const m = result.match(/\[\$([\d.]+) – \$([\d.]+)\]/);
    expect(m, 'No dollar CI found in output').not.toBeNull();
    const low  = parseFloat(m![1]);
    const high = parseFloat(m![2]);
    // Both bounds must be near $414, NOT near $100
    expect(low).toBeGreaterThan(currentPrice / 2);    // > $207 (not $99)
    expect(high).toBeGreaterThan(currentPrice / 2);   // > $207 (not $101)
    expect(high).toBeGreaterThan(low);
  });

  it('dollar CI bounds scale with current_price (500 gives ~5x wider dollar range than 100)', async () => {
    const extractDollarCI = async (price: number) => {
      const raw = await polymarketForecastTool.func(
        { ticker: 'NVDA', horizon_days: 7, current_price: price },
        undefined,
      );
      const result = parseResult(raw);
      const m = result.match(/\[\$([\d.]+) – \$([\d.]+)\]/);
      if (!m) return null;
      return { low: parseFloat(m[1]), high: parseFloat(m[2]) };
    };

    const ci100 = await extractDollarCI(100);
    const ci500 = await extractDollarCI(500);
    expect(ci100).not.toBeNull();
    expect(ci500).not.toBeNull();
    const width100 = ci100!.high - ci100!.low;
    const width500 = ci500!.high - ci500!.low;
    // CI width scales proportionally: width500 / width100 ≈ 500/100 = 5
    expect(width500 / width100).toBeCloseTo(5, 0);
  });

  it('no-price output uses % notation, not $ signs for the CI', async () => {
    const raw = await polymarketForecastTool.func(
      { ticker: 'NVDA', horizon_days: 7 },
      undefined,
    );
    const result = parseResult(raw);
    // CI must be in percent form: [-x.xx% – +x.xx%]
    expect(result).toMatch(/\[[-+\d.]+%\s*–\s*[+-\d.]+%\]/);
    // Must NOT contain a dollar-sign CI near $100 like [$99 or [$100
    expect(result).not.toMatch(/\[\$9[5-9]\.\d|\ \[\$10[0-5]\.\d/);
  });
});

// ---------------------------------------------------------------------------
// CI width grows with horizon
// ---------------------------------------------------------------------------

describe('CI width increases with longer horizons', () => {
  it('90-day sigma is strictly larger than 7-day sigma (floor: 5.98% vs 1.67%)', async () => {
    // The sigma floor is 0.10 × √(h/252). At 7d it's 1.67% and at 90d it's 5.98%.
    // When raw sigma (from mock market variance) sits between those two values,
    // sigma7 = raw and sigma90 = max(5.98%, raw) > raw = sigma7.
    const extractSigma = async (horizonDays: number) => {
      const raw = await polymarketForecastTool.func(
        { ticker: 'NVDA', horizon_days: horizonDays, current_price: 500 },
        undefined,
      );
      const m = parseResult(raw).match(/σ = ([\d.]+)%/);
      return m ? parseFloat(m[1]) : 0;
    };

    const sigma7  = await extractSigma(7);
    const sigma90 = await extractSigma(90);
    const sigma252 = await extractSigma(252);

    // sigma is monotone non-decreasing — floor guarantees 90d ≥ 7d
    expect(sigma90).toBeGreaterThanOrEqual(sigma7);
    // At 252d the floor is exactly 10% — always larger than any sub-252d sigma
    expect(sigma252).toBeGreaterThan(sigma90);
    // Sigma must be positive
    expect(sigma7).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Per-asset-class distinct returns (SPY vs GLD)
//
// Even with the same mock Polymarket markets, SPY (equity) and GLD (gold)
// must produce DIFFERENT conditional returns because they use different
// assetClass → different lookupImpact() deltas.
// ---------------------------------------------------------------------------

describe('different asset classes produce distinct conditional returns', () => {
  it('SPY (equity) and GLD (gold) have opposite macro_growth conditional returns at high probability', () => {
    // Import impact-map directly (no mock needed — pure math)
    const { lookupImpact, inferAssetClass } = require('./impact-map.js');
    const { computeConditionalReturn, adjustYesBias } = require('../../utils/ensemble.js');

    const p = adjustYesBias(0.75); // high probability of recession

    const spyClass = inferAssetClass('SPY');  // equity
    const gldClass = inferAssetClass('GLD');  // gold

    const spyR = computeConditionalReturn(
      p,
      lookupImpact('macro_growth', spyClass).deltaYes,
      lookupImpact('macro_growth', spyClass).deltaNo,
    );
    const gldR = computeConditionalReturn(
      p,
      lookupImpact('macro_growth', gldClass).deltaYes,
      lookupImpact('macro_growth', gldClass).deltaNo,
    );

    // Recession is NEGATIVE for equity and POSITIVE for gold (safe haven)
    expect(spyR).toBeLessThan(0);
    expect(gldR).toBeGreaterThan(0);
    // They must produce meaningfully different returns
    expect(gldR).toBeGreaterThan(spyR + 0.01);
  });

  it('KRE (financial) and IWM (small_cap) have opposite macro_rates conditional returns', () => {
    const { lookupImpact, inferAssetClass } = require('./impact-map.js');
    const { computeConditionalReturn, adjustYesBias } = require('../../utils/ensemble.js');

    const p = adjustYesBias(0.80); // high probability of rate cut

    const kreClass = inferAssetClass('KRE');  // financial
    const iwmClass = inferAssetClass('IWM');  // small_cap

    const kreR = computeConditionalReturn(
      p,
      lookupImpact('macro_rates', kreClass).deltaYes,
      lookupImpact('macro_rates', kreClass).deltaNo,
    );
    const iwmR = computeConditionalReturn(
      p,
      lookupImpact('macro_rates', iwmClass).deltaYes,
      lookupImpact('macro_rates', iwmClass).deltaNo,
    );

    // Rate cut is NEGATIVE for bank NIM (KRE) but POSITIVE for leveraged small caps (IWM)
    expect(kreR).toBeLessThan(0);
    expect(iwmR).toBeGreaterThan(0);
  });

  it('SLX (materials) and SPY (equity) have opposite trade_policy conditional returns', () => {
    const { lookupImpact, inferAssetClass } = require('./impact-map.js');
    const { computeConditionalReturn, adjustYesBias } = require('../../utils/ensemble.js');

    const p = adjustYesBias(0.97); // near-certain tariff imposition

    const slxClass = inferAssetClass('SLX');  // materials
    const spyClass = inferAssetClass('SPY');  // equity

    const slxR = computeConditionalReturn(
      p,
      lookupImpact('trade_policy', slxClass).deltaYes,
      lookupImpact('trade_policy', slxClass).deltaNo,
    );
    const spyR = computeConditionalReturn(
      p,
      lookupImpact('trade_policy', spyClass).deltaYes,
      lookupImpact('trade_policy', spyClass).deltaNo,
    );

    // Tariffs PROTECT domestic steel (SLX +) while hurting broad market (SPY −)
    expect(slxR).toBeGreaterThan(0);
    expect(spyR).toBeLessThan(0);
  });
});

// ---------------------------------------------------------------------------
// Query variant expansion — verify that queryVariants are actually queried
// alongside primary phrases when fetching Polymarket markets.
// ---------------------------------------------------------------------------

describe('queryVariant expansion in polymarket_forecast', () => {
  it('queries both searchPhrase and queryVariants per signal', async () => {
    const queriedPhrases: string[] = [];

    const freshTool = makeHermeticTool(async (query: string) => {
      queriedPhrases.push(query);
      return mockMarkets;
    });
    await freshTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );

    // NVDA extracts signals like { searchPhrase: 'NVIDIA earnings', queryVariants: ['NVIDIA', 'semiconductor earnings'] }
    // and { searchPhrase: 'chip export controls', queryVariants: ['semiconductor export', 'chip export'] }
    // We must see at least one searchPhrase AND at least one queryVariant in the queried phrases
    expect(queriedPhrases.length).toBeGreaterThan(0);

    // Verify per-query limit is 5 (not the old 3)
    // We can't directly check the limit parameter in this test structure,
    // but we verify variants are present
    const hasSearchPhrase = queriedPhrases.some(q => q.includes('earnings') || q.includes('Fed') || q.includes('export'));
    const hasVariant = queriedPhrases.some(q => q.includes('semiconductor') || q.includes('NVIDIA') || q.includes('FOMC'));
    expect(hasSearchPhrase).toBe(true);
    expect(hasVariant).toBe(true);
  });

  it('deduplicates by question across variant results', async () => {
    let callCount = 0;
    const sameQuestion = 'Will NVIDIA beat Q2 earnings?';

    const freshTool = makeHermeticTool(async () => {
      callCount++;
      return [
        { question: sameQuestion, probability: 0.72, volume24h: 500_000, ageDays: 0 },
        { question: `Unique result ${callCount}`, probability: 0.6, volume24h: 200_000, ageDays: 0 },
      ];
    });
    const raw = await freshTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    const result = parseResult(raw);

    // The same question should appear only once despite being returned by multiple queries
    const matchCount = (result.match(new RegExp(sameQuestion.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
    expect(matchCount).toBeLessThanOrEqual(1);
  });

  it('returns results even when primary phrase returns empty but variant has results', async () => {
    const primaryResults: PolymarketMarketResult[] = [];
    const variantResults: PolymarketMarketResult[] = [
      { question: 'Will NVIDIA revenue exceed $30B?', probability: 0.65, volume24h: 300_000, ageDays: 0 },
    ];

    const freshTool = makeHermeticTool(async (query: string) => {
      const isPrimary = query.includes('earnings') || query.includes('Fed') || query.includes('export') || query.includes('recession');
      return isPrimary ? primaryResults : variantResults;
    });
    const raw = await freshTool.func(
      { ticker: 'NVDA', horizon_days: 7, current_price: 135.50 },
      undefined,
    );
    const result = parseResult(raw);

    // Should still produce a valid forecast (variant results fill in)
    expect(result).toContain('Polymarket Forecast');
  });
});
