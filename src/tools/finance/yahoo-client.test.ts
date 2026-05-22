/**
 * Unit tests for src/tools/finance/yahoo-client.ts
 *
 * All network calls are mocked — verifies the crumb acquisition flow,
 * retry-on-401 logic, response parsing, and error handling.
 */
import { describe, it, expect, beforeEach, afterAll, mock } from 'bun:test';
import { quoteSummary, _clearCrumbCache } from './yahoo-client.js';

// ---------------------------------------------------------------------------
// Fetch mock helpers
// ---------------------------------------------------------------------------

const originalFetch = global.fetch;
let fetchImpl: (url: string | URL | Request, init?: RequestInit) => Promise<Response>;

global.fetch = mock(async (url: string | URL | Request, init?: RequestInit) => {
  return fetchImpl(url, init);
}) as unknown as typeof fetch;

afterAll(() => { global.fetch = originalFetch; });

function makeJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status, headers: { 'Content-Type': 'application/json' },
  });
}
function makeTextResponse(body: string, status = 200, setCookie = ''): Response {
  const headers: Record<string, string> = { 'Content-Type': 'text/plain' };
  if (setCookie) headers['set-cookie'] = setCookie;
  return new Response(body, { status, headers });
}

const VALID_FINANCIAL_DATA = {
  targetHighPrice: 250,
  targetLowPrice: 150,
  targetMeanPrice: 200,
  targetMedianPrice: 205,
  recommendationMean: 2.1,
  recommendationKey: 'buy',
  numberOfAnalystOpinions: 30,
};

// Standard happy-path sequence: consent → crumb → data
function setupHappyPath(dataBody: unknown) {
  let callCount = 0;
  fetchImpl = (url) => {
    const u = url.toString();
    callCount++;
    if (u.includes('fc.yahoo.com')) return Promise.resolve(makeTextResponse('ok', 200, 'A1=sess'));
    if (u.includes('getcrumb')) return Promise.resolve(makeTextResponse('crumb-abc'));
    return Promise.resolve(makeJsonResponse(dataBody));
  };
  return () => callCount;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('yahoo-client — quoteSummary', () => {
  beforeEach(() => {
    _clearCrumbCache();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global.fetch as unknown as { mockClear?: () => void }).mockClear?.();
  });

  it('makes 3 fetch calls: consent + crumb + data on cold cache', async () => {
    let calls: string[] = [];
    fetchImpl = (url) => {
      const u = url.toString();
      calls.push(u);
      if (u.includes('fc.yahoo.com')) return Promise.resolve(makeTextResponse('ok', 200, 'A1=s'));
      if (u.includes('getcrumb')) return Promise.resolve(makeTextResponse('crumb1'));
      return Promise.resolve(makeJsonResponse({
        quoteSummary: { result: [{ financialData: VALID_FINANCIAL_DATA }], error: null },
      }));
    };

    const result = await quoteSummary('AAPL', { modules: ['financialData'] });
    expect(result.financialData?.targetMeanPrice).toBe(200);
    expect(calls).toHaveLength(3);
    expect(calls[0]).toContain('fc.yahoo.com');
    expect(calls[1]).toContain('getcrumb');
    expect(calls[2]).toContain('quoteSummary/AAPL');
  });

  it('uses cached crumb on second call (only 1 fetch instead of 3)', async () => {
    let calls = 0;
    fetchImpl = (url) => {
      calls++;
      const u = url.toString();
      if (u.includes('fc.yahoo.com')) return Promise.resolve(makeTextResponse('ok', 200, 'A1=s'));
      if (u.includes('getcrumb')) return Promise.resolve(makeTextResponse('crumb1'));
      return Promise.resolve(makeJsonResponse({
        quoteSummary: { result: [{ financialData: VALID_FINANCIAL_DATA }], error: null },
      }));
    };

    await quoteSummary('AAPL', { modules: ['financialData'] }); // warms cache
    const before = calls;
    await quoteSummary('TSLA', { modules: ['financialData'] }); // uses cache
    expect(calls - before).toBe(1); // only 1 fetch for the data call
  });

  it('retries once with fresh crumb on 401', async () => {
    let dataCallCount = 0;
    fetchImpl = (url) => {
      const u = url.toString();
      if (u.includes('fc.yahoo.com')) return Promise.resolve(makeTextResponse('ok', 200, 'A1=s'));
      if (u.includes('getcrumb')) return Promise.resolve(makeTextResponse('crumb-new'));
      dataCallCount++;
      if (dataCallCount === 1) return Promise.resolve(new Response('Unauthorized', { status: 401 }));
      return Promise.resolve(makeJsonResponse({
        quoteSummary: {
          result: [{
            recommendationTrend: {
              trend: [{ period: '0m', strongBuy: 2, buy: 8, hold: 4, sell: 1, strongSell: 0 }],
            },
          }],
          error: null,
        },
      }));
    };

    const result = await quoteSummary('TSLA', { modules: ['recommendationTrend'] });
    expect(result.recommendationTrend?.trend?.[0]?.buy).toBe(8);
    expect(dataCallCount).toBe(2); // first failed, second succeeded
  });

  it('throws when Yahoo returns 500 even after retry', async () => {
    fetchImpl = (url) => {
      const u = url.toString();
      if (u.includes('fc.yahoo.com')) return Promise.resolve(makeTextResponse('ok', 200, 'A1=s'));
      if (u.includes('getcrumb')) return Promise.resolve(makeTextResponse('c'));
      return Promise.resolve(new Response('Server Error', { status: 500 }));
    };

    await expect(quoteSummary('FAIL', { modules: ['financialData'] })).rejects.toThrow('500');
  });

  it('throws when Yahoo returns an error object in JSON', async () => {
    setupHappyPath({ quoteSummary: { result: null, error: { message: 'No fundamentals' } } });

    await expect(quoteSummary('NOPE', { modules: ['financialData'] })).rejects.toThrow('No fundamentals');
  });

  it('throws when result array is empty', async () => {
    setupHappyPath({ quoteSummary: { result: [], error: null } });

    await expect(quoteSummary('EMPTY', { modules: ['financialData'] })).rejects.toThrow('No data returned');
  });

  it('throws when consumed module fields have invalid shapes', async () => {
    setupHappyPath({
      quoteSummary: {
        result: [{ recommendationTrend: { trend: 'buy' } }],
        error: null,
      },
    });

    await expect(quoteSummary('BADSHAPE', { modules: ['recommendationTrend'] })).rejects.toThrow();
  });

  it('throws when requested financialData fields are missing from a nonempty result', async () => {
    setupHappyPath({
      quoteSummary: { result: [{}], error: null },
    });

    await expect(quoteSummary('BADFD', { modules: ['financialData'] })).rejects.toThrow();
  });

  it('throws when financialData consumed fields have invalid scalar types', async () => {
    setupHappyPath({
      quoteSummary: {
        result: [{ financialData: { targetMeanPrice: '200' } }],
        error: null,
      },
    });

    await expect(quoteSummary('BADFD', { modules: ['financialData'] })).rejects.toThrow();
  });

  it('throws when requested recommendationTrend is missing from a nonempty result', async () => {
    setupHappyPath({
      quoteSummary: { result: [{}], error: null },
    });

    await expect(quoteSummary('BADTREND', { modules: ['recommendationTrend'] })).rejects.toThrow();
  });

  it('throws when incomeStatementHistory consumed fields have invalid scalar types', async () => {
    setupHappyPath({
      quoteSummary: {
        result: [{
          incomeStatementHistory: {
            incomeStatementHistory: [{ endDate: '2024-12-31', totalRevenue: '17295000000' }],
          },
        }],
        error: null,
      },
    });

    await expect(quoteSummary('BADINCOME', { modules: ['incomeStatementHistory'] })).rejects.toThrow();
  });

  it('accepts valid incomeStatementHistory fields including nullable metrics', async () => {
    setupHappyPath({
      quoteSummary: {
        result: [{
          incomeStatementHistory: {
            incomeStatementHistory: [{
              endDate: '2024-12-31',
              totalRevenue: 17_295_000_000,
              grossProfit: null,
              operatingIncome: 800_000_000,
              netIncome: 499_000_000,
              ebit: null,
            }],
          },
        }],
        error: null,
      },
    });

    const result = await quoteSummary('GOODINCOME', { modules: ['incomeStatementHistory'] });
    expect(result.incomeStatementHistory?.incomeStatementHistory?.[0]?.totalRevenue)
      .toBe(17_295_000_000);
  });

  it('throws when crumb endpoint returns empty string', async () => {
    fetchImpl = (url) => {
      const u = url.toString();
      if (u.includes('fc.yahoo.com')) return Promise.resolve(makeTextResponse('ok', 200, 'A1=s'));
      if (u.includes('getcrumb')) return Promise.resolve(makeTextResponse('  ')); // blank
      return Promise.resolve(makeJsonResponse({}));
    };

    await expect(quoteSummary('CRUMB', { modules: ['financialData'] })).rejects.toThrow('empty crumb');
  });

  it('encodes ticker and crumb in the quoteSummary URL', async () => {
    let dataUrl = '';
    fetchImpl = (url) => {
      const u = url.toString();
      if (u.includes('fc.yahoo.com')) return Promise.resolve(makeTextResponse('ok', 200, 'A1=s'));
      if (u.includes('getcrumb')) return Promise.resolve(makeTextResponse('my crumb/special'));
      dataUrl = u;
      return Promise.resolve(makeJsonResponse({
        quoteSummary: { result: [{ financialData: VALID_FINANCIAL_DATA }], error: null },
      }));
    };

    await quoteSummary('SAP.DE', { modules: ['financialData'] });
    expect(dataUrl).toContain('SAP.DE');
    expect(dataUrl).toContain('crumb=');
    expect(dataUrl).toContain('financialData');
  });
});
