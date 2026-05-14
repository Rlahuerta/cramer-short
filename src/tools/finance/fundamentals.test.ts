import { describe, test, expect, mock, beforeEach, afterEach, spyOn } from 'bun:test';

// ---------------------------------------------------------------------------
// Mock ./fmp.js BEFORE importing fundamentals.js so the dynamic import picks
// up our stubs instead of the real FMP tools.
// ---------------------------------------------------------------------------

const mockTool = (name: string, invoke: ReturnType<typeof mock>) => ({ invoke, name });

const mockFmpIncomeInvoke = mock(async (_input: unknown): Promise<string> => '{}');
const mockFmpBalanceInvoke = mock(async (_input: unknown): Promise<string> => '{}');
const mockFmpCashFlowInvoke = mock(async (_input: unknown): Promise<string> => '{}');

mock.module('./fmp.js', () => ({
  getFmpIncomeStatements: mockTool('get_fmp_income_statements', mockFmpIncomeInvoke),
  getFmpBalanceSheets: mockTool('get_fmp_balance_sheets', mockFmpBalanceInvoke),
  getFmpCashFlowStatements: mockTool('get_fmp_cash_flow_statements', mockFmpCashFlowInvoke),
  fmpApi: { get: mock(async () => []) },
  FMP_PREMIUM_REQUIRED: 'FMP_PREMIUM_REQUIRED',
}));

// ---------------------------------------------------------------------------
// Mock ./yahoo-finance.js — provides getYahooIncomeStatements fallback
// ---------------------------------------------------------------------------

const mockYahooIncomeInvoke = mock(async (_input: unknown): Promise<string> => '{}');
const mockYahooAnalystTargetsInvoke = mock(async (_input: unknown): Promise<string> => '{}');
const mockYahooAnalystRecommendationsInvoke = mock(async (_input: unknown): Promise<string> => '{}');
const mockYahooUpgradeDowngradeHistoryInvoke = mock(async (_input: unknown): Promise<string> => '{}');

mock.module('./yahoo-finance.js', () => ({
  getYahooIncomeStatements: mockTool('get_yahoo_income_statements', mockYahooIncomeInvoke),
  getYahooAnalystTargets: mockTool('get_yahoo_analyst_targets', mockYahooAnalystTargetsInvoke),
  getYahooAnalystRecommendations: mockTool(
    'get_yahoo_analyst_recommendations',
    mockYahooAnalystRecommendationsInvoke,
  ),
  getYahooUpgradeDowngradeHistory: mockTool(
    'get_yahoo_upgrade_downgrade_history',
    mockYahooUpgradeDowngradeHistoryInvoke,
  ),
}));

// ---------------------------------------------------------------------------
// Mock ../search/tavily.js — provides web search last-resort fallback
// ---------------------------------------------------------------------------

const mockTavilyInvoke = mock(
  async (_input: unknown): Promise<string> =>
    JSON.stringify({ data: { error: 'No data' }, sourceUrls: [] }),
);

mock.module('../search/tavily.js', () => ({
  tavilySearch: mockTool('web_search', mockTavilyInvoke),
}));

// Signal that FMP is configured — fallback will only run when the key is set
process.env.FMP_API_KEY = 'test-fmp-key';

import { api } from './api.js';
const { getIncomeStatements, getBalanceSheets, getCashFlowStatements } =
  await import('./fundamentals.js');

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let apiGetSpy: ReturnType<typeof spyOn<any, any>> | undefined;

afterEach(() => {
  apiGetSpy?.mockRestore();
  apiGetSpy = undefined;
  mockFmpIncomeInvoke.mockReset();
  mockFmpBalanceInvoke.mockReset();
  mockFmpCashFlowInvoke.mockReset();
  mockYahooIncomeInvoke.mockReset();
  mockYahooAnalystTargetsInvoke.mockReset();
  mockYahooAnalystRecommendationsInvoke.mockReset();
  mockYahooUpgradeDowngradeHistoryInvoke.mockReset();
  mockTavilyInvoke.mockReset();
  delete process.env.TAVILY_API_KEY;
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseResult(raw: unknown): { data: unknown; sourceUrls?: string[] } {
  return JSON.parse(raw as string);
}

const FD_INCOME_STUB = [{ period: '2024', revenue: 400_000_000_000, netIncome: 93_000_000_000 }];
const FD_BALANCE_STUB = [{ period: '2024', totalAssets: 350_000_000_000 }];
const FD_CASHFLOW_STUB = [{ period: '2024', operatingCashFlow: 110_000_000_000 }];

const FD_ENDPOINTS = {
  income_statements: 'https://api.financialdatasets.ai/financials/income-statements/',
  balance_sheets: 'https://api.financialdatasets.ai/financials/balance-sheets/',
  cash_flow_statements: 'https://api.financialdatasets.ai/financials/cash-flow-statements/',
} as const;

type FdStatementKey = keyof typeof FD_ENDPOINTS;

function mockFdResponse(data: Record<string, unknown>, url = ''): void {
  apiGetSpy?.mockRestore();
  apiGetSpy = spyOn(api, 'get').mockResolvedValue({ data, url });
}

function mockFdStatements(key: FdStatementKey, rows: unknown[], url: string = FD_ENDPOINTS[key]): void {
  mockFdResponse({ [key]: rows }, url);
}

function mockFdEmpty(key: FdStatementKey): void {
  mockFdStatements(key, [], '');
}

function mockFdFailure(message = '[Financial Datasets API] 404'): void {
  apiGetSpy?.mockRestore();
  apiGetSpy = spyOn(api, 'get').mockRejectedValue(new Error(message));
}

function enableTavily(): void {
  process.env.TAVILY_API_KEY = 'test-tavily-key';
}

const fmpIncomeResult = (ticker: string) =>
  JSON.stringify({
    data: [{ date: '2024-12-31', symbol: ticker, revenue: 16_500_000_000, netIncome: 528_000_000 }],
    sourceUrls: [`https://financialmodelingprep.com/financial-statements/${ticker}`],
  });

const fmpBalanceResult = (ticker: string) =>
  JSON.stringify({
    data: [{ date: '2024-12-31', symbol: ticker, totalAssets: 24_000_000_000 }],
    sourceUrls: [`https://financialmodelingprep.com/financial-statements/${ticker}`],
  });

const fmpCashFlowResult = (ticker: string) =>
  JSON.stringify({
    data: [{ date: '2024-12-31', symbol: ticker, freeCashFlow: 400_000_000 }],
    sourceUrls: [`https://financialmodelingprep.com/financial-statements/${ticker}`],
  });

const fmpErrorResult = () =>
  JSON.stringify({ data: { error: 'No data found for UNKNOWN on FMP.' }, sourceUrls: [] });

// ===========================================================================
// getIncomeStatements — FMP fallback
// ===========================================================================

describe('getIncomeStatements — FMP fallback', () => {
  beforeEach(() => {
    mockFmpIncomeInvoke.mockReset();
    mockYahooIncomeInvoke.mockReset();
    // Default: Yahoo returns error so it doesn't interfere with FMP fallback tests
    mockYahooIncomeInvoke.mockResolvedValue(
      JSON.stringify({ data: { error: 'No data' }, sourceUrls: [] }),
    );
  });

  test('does NOT call FMP when financialdatasets.ai returns data (non-annual)', async () => {
    mockFdStatements('income_statements', FD_INCOME_STUB);

    // period=quarterly skips cross-validation — FMP should not be called
    await getIncomeStatements.invoke({ ticker: 'AAPL', period: 'quarterly', limit: 4 });

    expect(mockFmpIncomeInvoke).not.toHaveBeenCalled();
  });

  test('calls FMP for cross-validation when financialdatasets.ai returns annual data', async () => {
    mockFdStatements('income_statements', FD_INCOME_STUB);
    // FMP returns agreeing data — no warning appended
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpIncomeResult('AAPL'));

    const result = await getIncomeStatements.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });

    expect(mockFmpIncomeInvoke).toHaveBeenCalledTimes(1);
    expect(result).not.toContain('⚠️');
  });

  test('calls FMP when financialdatasets.ai returns empty array', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpIncomeResult('VWS.CO'));

    await getIncomeStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockFmpIncomeInvoke).toHaveBeenCalledTimes(1);
  });

  test('calls FMP with correct ticker and period', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpIncomeResult('VWS.CO'));

    await getIncomeStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    const [calledInput] = mockFmpIncomeInvoke.mock.calls[0] as [{ ticker: string; period: string }];
    expect(calledInput.ticker).toBe('VWS.CO');
    expect(calledInput.period).toBe('annual');
  });

  test('maps period=ttm to annual when calling FMP', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpIncomeResult('AAPL'));

    await getIncomeStatements.invoke({ ticker: 'AAPL', period: 'ttm', limit: 1 });

    const [calledInput] = mockFmpIncomeInvoke.mock.calls[0] as [{ period: string }];
    expect(calledInput.period).toBe('annual');
  });

  test('calls FMP when financialdatasets.ai throws', async () => {
    mockFdFailure('[Financial Datasets API] 404 Not Found');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpIncomeResult('VWS.CO'));

    await getIncomeStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockFmpIncomeInvoke).toHaveBeenCalledTimes(1);
  });

  test('returns FMP data and sourceUrls when FD.ai is empty', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpIncomeResult('VWS.CO'));

    const raw = await getIncomeStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });
    const result = parseResult(raw);

    expect((result.data as Record<string, unknown>[])[0]?.revenue).toBe(16_500_000_000);
    expect(result.sourceUrls?.some((u) => u.includes('financialmodelingprep'))).toBe(true);
  });

  test('returns error when FMP and Yahoo both return errors', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpErrorResult());
    mockYahooIncomeInvoke.mockResolvedValueOnce(
      JSON.stringify({ data: { error: 'No income data for UNKNOWN' }, sourceUrls: [] }),
    );
    // TAVILY_API_KEY not set → Tavily skipped

    const raw = await getIncomeStatements.invoke({ ticker: 'UNKNOWN', period: 'annual', limit: 4 });
    const result = parseResult(raw);

    expect((result.data as Record<string, unknown>).error).toBeDefined();
  });
});

// ===========================================================================
// getBalanceSheets — FMP fallback
// ===========================================================================

describe('getBalanceSheets — FMP fallback', () => {
  beforeEach(() => {
    mockFmpBalanceInvoke.mockReset();
  });

  test('does NOT call FMP when financialdatasets.ai returns data', async () => {
    mockFdStatements('balance_sheets', FD_BALANCE_STUB);

    await getBalanceSheets.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });

    expect(mockFmpBalanceInvoke).not.toHaveBeenCalled();
  });

  test('calls FMP when financialdatasets.ai returns empty array', async () => {
    mockFdEmpty('balance_sheets');
    mockFmpBalanceInvoke.mockResolvedValueOnce(fmpBalanceResult('VWS.CO'));

    await getBalanceSheets.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockFmpBalanceInvoke).toHaveBeenCalledTimes(1);
  });

  test('calls FMP when financialdatasets.ai throws', async () => {
    mockFdFailure();
    mockFmpBalanceInvoke.mockResolvedValueOnce(fmpBalanceResult('VWS.CO'));

    await getBalanceSheets.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockFmpBalanceInvoke).toHaveBeenCalledTimes(1);
  });

  test('returns FMP data when FD.ai is empty', async () => {
    mockFdEmpty('balance_sheets');
    mockFmpBalanceInvoke.mockResolvedValueOnce(fmpBalanceResult('VWS.CO'));

    const raw = await getBalanceSheets.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });
    const result = parseResult(raw);

    expect((result.data as Record<string, unknown>[])[0]?.totalAssets).toBe(24_000_000_000);
    expect(result.sourceUrls?.some((u) => u.includes('financialmodelingprep'))).toBe(true);
  });
});

// ===========================================================================
// getCashFlowStatements — FMP fallback
// ===========================================================================

describe('getCashFlowStatements — FMP fallback', () => {
  beforeEach(() => {
    mockFmpCashFlowInvoke.mockReset();
  });

  test('does NOT call FMP when financialdatasets.ai returns data', async () => {
    mockFdStatements('cash_flow_statements', FD_CASHFLOW_STUB);

    await getCashFlowStatements.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });

    expect(mockFmpCashFlowInvoke).not.toHaveBeenCalled();
  });

  test('calls FMP when financialdatasets.ai returns empty array', async () => {
    mockFdEmpty('cash_flow_statements');
    mockFmpCashFlowInvoke.mockResolvedValueOnce(fmpCashFlowResult('VWS.CO'));

    await getCashFlowStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockFmpCashFlowInvoke).toHaveBeenCalledTimes(1);
  });

  test('calls FMP when financialdatasets.ai throws', async () => {
    mockFdFailure();
    mockFmpCashFlowInvoke.mockResolvedValueOnce(fmpCashFlowResult('VWS.CO'));

    await getCashFlowStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockFmpCashFlowInvoke).toHaveBeenCalledTimes(1);
  });

  test('returns FMP data when FD.ai is empty', async () => {
    mockFdEmpty('cash_flow_statements');
    mockFmpCashFlowInvoke.mockResolvedValueOnce(fmpCashFlowResult('VWS.CO'));

    const raw = await getCashFlowStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });
    const result = parseResult(raw);

    expect((result.data as Record<string, unknown>[])[0]?.freeCashFlow).toBe(400_000_000);
    expect(result.sourceUrls?.some((u) => u.includes('financialmodelingprep'))).toBe(true);
  });
});

// ===========================================================================
// Helpers for premium fallback tests
// ===========================================================================

const fmpPremiumResult = () =>
  JSON.stringify({
    data: { error: 'FMP_PREMIUM_REQUIRED: This ticker requires a paid FMP subscription.' },
    sourceUrls: [],
  });

const yahooIncomeResult = (ticker: string) =>
  JSON.stringify({
    data: [{ date: '2024-12-31', totalRevenue: 17_295_000_000, netIncome: 499_000_000 }],
    sourceUrls: [`https://finance.yahoo.com/quote/${ticker}/financials`],
  });

const tavilyResult = () =>
  JSON.stringify({
    data: [{ title: 'Vestas Wind Systems Annual Report', url: 'https://example.com/vws-financials' }],
    sourceUrls: ['https://example.com/vws-financials'],
  });

// ===========================================================================
// getIncomeStatements — Yahoo Finance fallback when FMP is premium-only
// ===========================================================================

describe('getIncomeStatements — Yahoo Finance fallback on FMP premium', () => {
  beforeEach(() => {
    mockFmpIncomeInvoke.mockReset();
    mockYahooIncomeInvoke.mockReset();
    mockTavilyInvoke.mockReset();
  });

  test('calls Yahoo Finance when FMP returns premium error', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpPremiumResult());
    mockYahooIncomeInvoke.mockResolvedValueOnce(yahooIncomeResult('VWS.CO'));

    await getIncomeStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockYahooIncomeInvoke).toHaveBeenCalledTimes(1);
  });

  test('does NOT call Yahoo Finance when FMP succeeds', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpIncomeResult('AAPL'));

    await getIncomeStatements.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });

    expect(mockYahooIncomeInvoke).not.toHaveBeenCalled();
  });

  test('returns Yahoo Finance data with yahoo sourceUrls on FMP premium', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpPremiumResult());
    mockYahooIncomeInvoke.mockResolvedValueOnce(yahooIncomeResult('VWS.CO'));

    const raw = await getIncomeStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });
    const result = parseResult(raw);

    expect((result.data as Record<string, unknown>[])[0]?.totalRevenue).toBe(17_295_000_000);
    expect(result.sourceUrls?.some((u) => u.includes('yahoo.com'))).toBe(true);
  });

  test('falls through to Tavily when Yahoo also returns error', async () => {
    mockFdEmpty('income_statements');
    mockFmpIncomeInvoke.mockResolvedValueOnce(fmpPremiumResult());
    mockYahooIncomeInvoke.mockResolvedValueOnce(
      JSON.stringify({ data: { error: 'No income data for VWS.CO' }, sourceUrls: [] }),
    );
    enableTavily();
    mockTavilyInvoke.mockResolvedValueOnce(tavilyResult());

    await getIncomeStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockTavilyInvoke).toHaveBeenCalledTimes(1);
  });
});

// ===========================================================================
// getBalanceSheets — Tavily fallback when FMP is premium-only
// ===========================================================================

describe('getBalanceSheets — Tavily fallback on FMP premium', () => {
  beforeEach(() => {
    mockFmpBalanceInvoke.mockReset();
    mockTavilyInvoke.mockReset();
  });

  test('calls Tavily when FMP returns premium error and TAVILY_API_KEY is set', async () => {
    mockFdEmpty('balance_sheets');
    mockFmpBalanceInvoke.mockResolvedValueOnce(fmpPremiumResult());
    enableTavily();
    mockTavilyInvoke.mockResolvedValueOnce(tavilyResult());

    await getBalanceSheets.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockTavilyInvoke).toHaveBeenCalledTimes(1);
  });

  test('Tavily query includes ticker and balance sheet', async () => {
    mockFdEmpty('balance_sheets');
    mockFmpBalanceInvoke.mockResolvedValueOnce(fmpPremiumResult());
    enableTavily();
    mockTavilyInvoke.mockResolvedValueOnce(tavilyResult());

    await getBalanceSheets.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    const [calledInput] = mockTavilyInvoke.mock.calls[0] as [{ query: string }];
    expect(calledInput.query).toContain('VWS.CO');
    expect(calledInput.query.toLowerCase()).toContain('balance');
  });

  test('skips Tavily when TAVILY_API_KEY is not set', async () => {
    mockFdEmpty('balance_sheets');
    mockFmpBalanceInvoke.mockResolvedValueOnce(fmpPremiumResult());

    const raw = await getBalanceSheets.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });
    const result = parseResult(raw);

    expect(mockTavilyInvoke).not.toHaveBeenCalled();
    expect((result.data as Record<string, unknown>).error).toBeDefined();
  });
});

// ===========================================================================
// getCashFlowStatements — Tavily fallback when FMP is premium-only
// ===========================================================================

describe('getCashFlowStatements — Tavily fallback on FMP premium', () => {
  beforeEach(() => {
    mockFmpCashFlowInvoke.mockReset();
    mockTavilyInvoke.mockReset();
  });

  test('calls Tavily when FMP returns premium error and TAVILY_API_KEY is set', async () => {
    mockFdEmpty('cash_flow_statements');
    mockFmpCashFlowInvoke.mockResolvedValueOnce(fmpPremiumResult());
    enableTavily();
    mockTavilyInvoke.mockResolvedValueOnce(tavilyResult());

    await getCashFlowStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    expect(mockTavilyInvoke).toHaveBeenCalledTimes(1);
  });

  test('Tavily query includes ticker and cash flow', async () => {
    mockFdEmpty('cash_flow_statements');
    mockFmpCashFlowInvoke.mockResolvedValueOnce(fmpPremiumResult());
    enableTavily();
    mockTavilyInvoke.mockResolvedValueOnce(tavilyResult());

    await getCashFlowStatements.invoke({ ticker: 'VWS.CO', period: 'annual', limit: 4 });

    const [calledInput] = mockTavilyInvoke.mock.calls[0] as [{ query: string }];
    expect(calledInput.query).toContain('VWS.CO');
    expect(calledInput.query.toLowerCase()).toContain('cash flow');
  });
});
