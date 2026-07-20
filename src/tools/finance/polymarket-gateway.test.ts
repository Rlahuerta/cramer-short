import { FIXED_TEST_DATE } from '@/utils/test-determinism.js';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, setSystemTime } from 'bun:test';
import { createHttpReadDriver, normalizeGammaMarket, RETRY_DELAYS, searchGammaEventsPaginated, setRetryDelays } from './polymarket-gamma-client.js';
import { createPolymarketGateway } from './polymarket-gateway.js';
import { createSdkReadDriver } from './polymarket-sdk-driver.js';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

let originalRetryDelays: number[];
beforeAll(() => {
  originalRetryDelays = [...RETRY_DELAYS];
  setRetryDelays([0, 0, 0]);
});

afterAll(() => {
  setRetryDelays(originalRetryDelays);
});

describe('normalizeGammaMarket', () => {
  it('normalizes yes/no outcomes and preserves the primary yes token id', () => {
    const raw = {
      id: 'market-1',
      conditionId: 'condition-1',
      question: 'Will Bitcoin be above $120,000 on Dec 31?',
      outcomes: '["Yes","No"]',
      outcomePrices: '["0.42","0.58"]',
      clobTokenIds: '["yes-token","no-token"]',
      endDateIso: '2026-12-31',
      volume24hr: 125_000,
      liquidityNum: 40_000,
      active: true,
      closed: false,
      createdAt: '2026-07-10T00:00:00.000Z',
    };

    expect(normalizeGammaMarket(raw)).toMatchObject({
      marketId: 'condition-1',
      question: 'Will Bitcoin be above $120,000 on Dec 31?',
      primaryYesTokenId: 'yes-token',
      outcomes: {
        yes: {
          label: 'Yes',
          probability: 0.42,
          tokenId: 'yes-token',
        },
        no: {
          label: 'No',
          probability: 0.58,
          tokenId: 'no-token',
        },
      },
      volume24h: 125_000,
      liquidity: 40_000,
      endDate: '2026-12-31',
      active: true,
      closed: false,
    });
  });

  describe('searchGammaEventsPaginated', () => {
    it('keeps paginating until it finds enough filtered matches', async () => {
      const pageOne = Array.from({ length: 50 }, (_, index) => ({
        id: `e-${index}`,
        title: 'Sports markets',
        markets: [
          {
            id: `sports-${index}`,
            question: `Will Team ${index} win the championship?`,
            outcomes: '["Yes","No"]',
            outcomePrices: '["0.40","0.60"]',
            active: true,
            closed: false,
          },
        ],
      }));

      const pageTwo = [
        {
          id: 'e-2',
          title: 'Bitcoin markets',
          markets: [
            {
              id: 'btc-1',
              conditionId: 'condition-btc-1',
              question: 'Will Bitcoin be above $120,000 on Dec 31?',
              outcomes: '["Yes","No"]',
              outcomePrices: '["0.42","0.58"]',
              clobTokenIds: '["yes-token","no-token"]',
              volume24hr: 125_000,
              liquidityNum: 40_000,
              active: true,
              closed: false,
            },
          ],
        },
      ];

      const calls: string[] = [];
      const fetchFn = async (url: string | URL) => {
        const urlString = String(url);
        calls.push(urlString);
        const body = urlString.includes('offset=50') ? pageTwo : pageOne;
        return {
          ok: true,
          status: 200,
          json: async () => body,
        } as Response;
      };

      const result = await searchGammaEventsPaginated(
        {
          query: 'bitcoin above',
          limit: 1,
          tagSlugs: ['bitcoin'],
          pageSize: 50,
        },
        fetchFn as typeof fetch,
      );

      expect(result.markets).toHaveLength(1);
      expect(result.markets[0]?.question).toContain('Bitcoin');
      expect(result.provenance.pagesRead).toBe(2);
      expect(calls.some((call) => call.includes('offset=50'))).toBe(true);
    });

    it('retries rate-limited Gamma pages before failing the search', async () => {
      let calls = 0;
      const fetchFn = async (_url: string | URL) => {
        calls++;
        if (calls < 3) {
          return {
            ok: false,
            status: 429,
            json: async () => ([]),
          } as Response;
        }
        return {
          ok: true,
          status: 200,
          json: async () => ([
            {
              id: 'e-btc',
              title: 'Bitcoin markets',
              markets: [
                {
                  id: 'btc-1',
                  conditionId: 'condition-btc-1',
                  question: 'Will Bitcoin be above $120,000 on Dec 31?',
                  outcomes: '["Yes","No"]',
                  outcomePrices: '["0.42","0.58"]',
                  clobTokenIds: '["yes-token","no-token"]',
                  volume24hr: 125_000,
                  liquidityNum: 40_000,
                  active: true,
                  closed: false,
                },
              ],
            },
          ]),
        } as Response;
      };

      const result = await searchGammaEventsPaginated(
        {
          query: 'bitcoin above',
          limit: 1,
          tagSlugs: ['bitcoin'],
        },
        fetchFn as typeof fetch,
      );

      expect(result.markets).toHaveLength(1);
      expect(calls).toBe(3);
    });
  });

  describe('createPolymarketGateway', () => {
    it('accepts the http read driver and preserves the search contract', async () => {
      const pageOne = Array.from({ length: 50 }, (_, index) => ({
        id: `e-${index}`,
        title: 'Sports markets',
        markets: [
          {
            id: `sports-${index}`,
            question: `Will Team ${index} win the championship?`,
            outcomes: '["Yes","No"]',
            outcomePrices: '["0.40","0.60"]',
            active: true,
            closed: false,
          },
        ],
      }));
      const pageTwo = [
        {
          id: 'e-btc',
          title: 'Bitcoin markets',
          markets: [
            {
              id: 'btc-1',
              conditionId: 'condition-btc-1',
              question: 'Will Bitcoin be above $120,000 on Dec 31?',
              outcomes: '["Yes","No"]',
              outcomePrices: '["0.42","0.58"]',
              clobTokenIds: '["yes-token","no-token"]',
              volume24hr: 125_000,
              liquidityNum: 40_000,
              active: true,
              closed: false,
            },
          ],
        },
      ];

      const fetchFn = async (url: string | URL) => ({
        ok: true,
        status: 200,
        json: async () => String(url).includes('offset=50') ? pageTwo : pageOne,
      }) as Response;

      const gateway = createPolymarketGateway({
        readDriver: createHttpReadDriver(fetchFn as typeof fetch),
      });

      const result = await gateway.searchMarkets({
        query: 'bitcoin above',
        limit: 1,
        tagSlugs: ['bitcoin'],
        pageSize: 50,
      });

      expect(result.markets).toHaveLength(1);
      expect(result.provenance.source).toBe('gamma-http');
    });

    it('accepts an sdk-style read driver without changing the search contract', async () => {
      const fakeClient = {
        async *listMarkets() {
          yield {
            items: [
              {
                conditionId: 'condition-btc-market',
                question: 'Will Bitcoin be above $120,000 on Dec 31?',
                endDateIso: '2026-12-31',
                volume24hr: 125_000,
                liquidityNum: 40_000,
                active: true,
                closed: false,
                outcomes: {
                  yes: { price: 0.42, tokenId: 'btc-yes' },
                  no: { price: 0.58, tokenId: 'btc-no' },
                },
              },
            ],
          };
        },
      };

      const gateway = createPolymarketGateway({
        readDriver: createSdkReadDriver(fakeClient),
      });

      const result = await gateway.searchMarkets({ query: 'bitcoin above', limit: 1 });

      expect(result.markets).toHaveLength(1);
      expect(result.markets[0]?.primaryYesTokenId).toBe('btc-yes');
      expect(result.provenance.source).toBe('sdk');
    });
  });
});
