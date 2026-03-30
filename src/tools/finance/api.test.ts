import { describe, it, expect, mock, spyOn, beforeEach, afterEach } from 'bun:test';

// Mock cache module BEFORE importing api.ts
const mockReadCache = mock((..._args: unknown[]) => null as unknown);
const mockWriteCache = mock((..._args: unknown[]) => undefined);
const mockDescribeRequest = mock((endpoint: string) => String(endpoint));

mock.module('../../utils/cache.js', () => ({
  readCache: mockReadCache,
  writeCache: mockWriteCache,
  describeRequest: mockDescribeRequest,
}));

// Use a cache-busting query param to force Bun to re-evaluate api.ts with the mocked
// cache.js above. Other finance test files statically import api.js (which loads cache.ts at
// module init time before any mock is set), leaving spyOn(api, 'get') stacks that are never
// restored. The ?t= suffix guarantees this file gets a fresh api module whose cache
// bindings point at the mocks and whose .get method has never been spied on.
const { api, stripFieldsDeep } = await import(`./api.js?t=${Date.now()}`) as typeof import('./api.js');
const { logger } = await import('../../utils/logger.js');

// ---------------------------------------------------------------------------
// stripFieldsDeep
// ---------------------------------------------------------------------------

describe('stripFieldsDeep', () => {
  it('removes specified keys from a flat object', () => {
    const result = stripFieldsDeep({ a: 1, b: 2, c: 3 }, ['b']);
    expect(result).toEqual({ a: 1, c: 3 });
  });

  it('removes keys recursively from nested objects', () => {
    const result = stripFieldsDeep({ a: 1, nested: { b: 2, keep: 'yes' } }, ['b']);
    expect(result).toEqual({ a: 1, nested: { keep: 'yes' } });
  });

  it('removes keys from arrays of objects', () => {
    const result = stripFieldsDeep([{ a: 1, strip: 'x' }, { a: 2, strip: 'y' }], ['strip']);
    expect(result).toEqual([{ a: 1 }, { a: 2 }]);
  });

  it('leaves non-matching keys intact', () => {
    const input = { x: 1, y: 2 };
    const result = stripFieldsDeep(input, ['z']);
    expect(result).toEqual({ x: 1, y: 2 });
  });

  it('returns primitives unchanged', () => {
    expect(stripFieldsDeep(42, ['a'])).toBe(42);
    expect(stripFieldsDeep('hello', ['a'])).toBe('hello');
    expect(stripFieldsDeep(null, ['a'])).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// api.get
// ---------------------------------------------------------------------------

describe('api.get', () => {
  let fetchSpy: ReturnType<typeof spyOn<typeof globalThis, 'fetch'>>;
  let originalApiKey: string | undefined;

  beforeEach(() => {
    mockReadCache.mockClear();
    mockWriteCache.mockClear();
    originalApiKey = process.env.FINANCIAL_DATASETS_API_KEY;
    process.env.FINANCIAL_DATASETS_API_KEY = 'test-key';

    fetchSpy = spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: () => Promise.resolve({ earnings: { revenue: 1000 } }),
    } as Response);
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    if (originalApiKey === undefined) {
      delete process.env.FINANCIAL_DATASETS_API_KEY;
    } else {
      process.env.FINANCIAL_DATASETS_API_KEY = originalApiKey;
    }
  });

  it('builds correct URL for a successful GET', async () => {
    const result = await api.get('/earnings', { ticker: 'AAPL' });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const calledUrl = (fetchSpy.mock.calls[0] as [string, RequestInit])[0];
    expect(calledUrl).toContain('/earnings');
    expect(calledUrl).toContain('ticker=AAPL');
    expect(result.data).toEqual({ earnings: { revenue: 1000 } });
  });

  it('appends array params multiple times', async () => {
    await api.get('/filings/', { ticker: 'AAPL', filing_type: ['10-K', '10-Q'] });
    const calledUrl = (fetchSpy.mock.calls[0] as [string, RequestInit])[0];
    expect(calledUrl).toContain('filing_type=10-K');
    expect(calledUrl).toContain('filing_type=10-Q');
  });

  it('skips undefined params', async () => {
    await api.get('/filings/', { ticker: 'AAPL', limit: undefined });
    const calledUrl = (fetchSpy.mock.calls[0] as [string, RequestInit])[0];
    expect(calledUrl).not.toContain('limit');
  });

  it('throws with message on network error', async () => {
    fetchSpy.mockRejectedValue(new Error('ECONNREFUSED'));
    await expect(api.get('/earnings', { ticker: 'AAPL' })).rejects.toThrow('ECONNREFUSED');
  });

  it('throws on non-ok response (404)', async () => {
    fetchSpy.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: () => Promise.resolve({}),
    } as Response);
    await expect(api.get('/earnings', { ticker: 'AAPL' })).rejects.toThrow('404');
  });

  it('throws on JSON parse error', async () => {
    fetchSpy.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: () => Promise.reject(new Error('invalid json')),
    } as Response);
    await expect(api.get('/earnings', { ticker: 'AAPL' })).rejects.toThrow(/invalid JSON|invalid json/i);
  });

  it('throws when API key is missing', async () => {
    delete process.env.FINANCIAL_DATASETS_API_KEY;
    await expect(api.get('/earnings', { ticker: 'AAPL' })).rejects.toThrow(
      /FINANCIAL_DATASETS_API_KEY is not set/,
    );
  });

  it('returns cached data on cache hit when cacheable=true', async () => {
    const cachedEntry = { data: { cached: true }, url: 'https://cached.example.com' };
    mockReadCache.mockReturnValueOnce(cachedEntry);

    const result = await api.get('/prices/', { ticker: 'AAPL' }, { cacheable: true });
    expect(result).toEqual(cachedEntry);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('calls writeCache after successful fetch when cacheable=true', async () => {
    mockReadCache.mockReturnValueOnce(null);
    await api.get('/prices/', { ticker: 'AAPL', start_date: '2023-01-01' }, { cacheable: true });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(mockWriteCache).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// api.post
// ---------------------------------------------------------------------------

describe('api.post', () => {
  let fetchSpy: ReturnType<typeof spyOn<typeof globalThis, 'fetch'>>;

  beforeEach(() => {
    process.env.FINANCIAL_DATASETS_API_KEY = 'test-key';
    fetchSpy = spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: () => Promise.resolve({ result: 'ok' }),
    } as Response);
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it('sends POST with correct body', async () => {
    const body = { query: 'earnings', ticker: 'AAPL' };
    const result = await api.post('/search', body);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual(body);
    expect(result.data).toEqual({ result: 'ok' });
  });

  it('throws on network error', async () => {
    fetchSpy.mockRejectedValue(new Error('Network down'));
    await expect(api.post('/search', {})).rejects.toThrow('Network down');
  });
});
