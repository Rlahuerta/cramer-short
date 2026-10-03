import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { describe, test, expect, beforeEach, afterEach, afterAll, mock, setSystemTime } from 'bun:test';
import { existsSync, mkdirSync, writeFileSync, rmSync } from 'fs';
import { join, resolve, sep } from 'path';
import { tmpdir } from 'os';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

// Unique temp dir per worker run — prevents parallel worker contamination of the real .cramer-short/cache
const TEST_CACHE_DIR = join(tmpdir(), `dexter-cache-test-${nextTestId('path')}`);

// Mock paths.js BEFORE cache.ts loads so the module-level CACHE_DIR = cramerShortPath('cache') resolves to our temp dir
mock.module('./paths.js', () => ({
  cramerShortPath: (first?: string, ...rest: string[]) =>
    first === 'cache' ? TEST_CACHE_DIR : join(TEST_CACHE_DIR, ...(first ? [first, ...rest] : [])),
  getCramerShortDir: () => TEST_CACHE_DIR,
}));

// Use a cache-busting query param to force Bun to re-evaluate cache.ts so that
// the module-level CACHE_DIR = cramerShortPath('cache') is evaluated AFTER paths.js
// is mocked above. Without the ?t= suffix, Bun returns a previously-cached
// cache.ts instance (loaded by filings.test.ts → api.ts → cache.ts) that
// already resolved CACHE_DIR against the real .cramer-short directory.
const { buildCacheKey, readCache, writeCache } = await import(`./cache.js?t=${nextTestId('module')}`) as typeof import('./cache.js');

// ---------------------------------------------------------------------------
// buildCacheKey
// ---------------------------------------------------------------------------

describe('buildCacheKey', () => {
  test('produces the same key regardless of param insertion order', () => {
    const paramsA = { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31', interval: 'day', interval_multiplier: 1 };
    const paramsB = { interval_multiplier: 1, end_date: '2024-12-31', ticker: 'AAPL', interval: 'day', start_date: '2024-01-01' };
    expect(buildCacheKey('/prices/', paramsA)).toBe(buildCacheKey('/prices/', paramsB));
  });

  test('sorts array values without mutating the original', () => {
    const items = ['Item-7', 'Item-1', 'Item-1A'];
    const original = [...items];
    buildCacheKey('/filings/items/', { ticker: 'AAPL', item: items });
    expect(items).toEqual(original); // not mutated
  });

  test('produces different keys for different params', () => {
    const keyA = buildCacheKey('/prices/', { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-06-30' });
    const keyB = buildCacheKey('/prices/', { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31' });
    expect(keyA).not.toBe(keyB);
  });

  test('includes ticker prefix for readable filenames', () => {
    const key = buildCacheKey('/prices/', { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31' });
    expect(key).toMatch(/^prices\/AAPL_/);
    expect(key).toMatch(/\.json$/);
  });

  test('omits undefined and null params', () => {
    const keyA = buildCacheKey('/prices/', { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31', limit: undefined });
    const keyB = buildCacheKey('/prices/', { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31' });
    expect(keyA).toBe(keyB);
  });
});

// ---------------------------------------------------------------------------
// buildCacheKey path-traversal hardening
// ---------------------------------------------------------------------------

describe('buildCacheKey path-traversal hardening', () => {
  test('keeps the exact historical filename for valid tickers (BTC-USD, BRK.B)', () => {
    expect(buildCacheKey('/prices/', { ticker: 'BTC-USD', interval: 'day', limit: 30 }))
      .toBe('prices/BTC-USD_1c2c58aab189.json');
    expect(buildCacheKey('/prices/', { ticker: 'BRK.B', interval: 'day', limit: 30 }))
      .toBe('prices/BRK.B_3a9bc0fb0814.json');
  });

  test('sanitizes a path-traversal ticker so the file stays under the cache root', () => {
    const key = buildCacheKey('/prices/', { ticker: '../../../../tmp/pwn' });
    const [dir, filename] = key.split('/');
    expect(dir).toBe('prices');
    expect(filename).toBeDefined();
    expect(filename!).not.toContain('/');
    expect(filename!).not.toContain('..');
    expect(filename!).toMatch(/^[A-Za-z0-9._-]{1,32}_[0-9a-f]{12}\.json$/);
    const root = resolve(TEST_CACHE_DIR);
    expect(resolve(root, key).startsWith(root + sep)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// readCache / writeCache round-trip
// ---------------------------------------------------------------------------

afterAll(() => {
  if (existsSync(TEST_CACHE_DIR)) {
    rmSync(TEST_CACHE_DIR, { recursive: true, force: true });
  }
});

describe('readCache / writeCache', () => {
  beforeEach(() => {
    if (existsSync(TEST_CACHE_DIR)) {
      rmSync(TEST_CACHE_DIR, { recursive: true });
    }
  });

  afterEach(() => {
    if (existsSync(TEST_CACHE_DIR)) {
      rmSync(TEST_CACHE_DIR, { recursive: true });
    }
  });

  test('round-trips data through write then read', () => {
    const endpoint = '/prices/';
    const params = { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31', interval: 'day', interval_multiplier: 1 };
    const data = { prices: [{ open: 100, close: 105, high: 106, low: 99 }] };
    const url = 'https://api.financialdatasets.ai/prices/?ticker=AAPL&start_date=2024-01-01&end_date=2024-12-31';

    writeCache(endpoint, params, data, url);
    const cached = readCache(endpoint, params);

    expect(cached).not.toBeNull();
    expect(cached!.data).toEqual(data);
    expect(cached!.url).toBe(url);
  });

  test('returns null on cache miss (no file)', () => {
    const cached = readCache('/prices/', { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31' });
    expect(cached).toBeNull();
  });

  test('round-trips a sanitized path-traversal ticker so lookups still cache', () => {
    const endpoint = '/prices/';
    const params = { ticker: '../../../../tmp/pwn', interval: 'day' };
    const data = { prices: [{ close: 42 }] };
    const url = 'https://api.financialdatasets.ai/prices/';

    writeCache(endpoint, params, data, url);
    const cached = readCache(endpoint, params);

    expect(cached).not.toBeNull();
    expect(cached!.data).toEqual(data);
    expect(cached!.url).toBe(url);
  });

  test('returns null and removes file when cache entry is corrupted JSON', () => {
    const endpoint = '/prices/';
    const params = { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31', interval: 'day', interval_multiplier: 1 };

    const key = buildCacheKey(endpoint, params);
    const filepath = join(TEST_CACHE_DIR, key);
    const dir = join(TEST_CACHE_DIR, key.split('/')[0]!);
    mkdirSync(dir, { recursive: true });
    writeFileSync(filepath, '{ broken json!!!');

    const cached = readCache(endpoint, params);
    expect(cached).toBeNull();
    expect(existsSync(filepath)).toBe(false);
  });

  test('returns null and removes file when cache entry has invalid structure', () => {
    const endpoint = '/prices/';
    const params = { ticker: 'AAPL', start_date: '2024-01-01', end_date: '2024-12-31', interval: 'day', interval_multiplier: 1 };

    const key = buildCacheKey(endpoint, params);
    const filepath = join(TEST_CACHE_DIR, key);
    const dir = join(TEST_CACHE_DIR, key.split('/')[0]!);
    mkdirSync(dir, { recursive: true });
    writeFileSync(filepath, JSON.stringify({ wrong: 'shape' }));

    const cached = readCache(endpoint, params);
    expect(cached).toBeNull();
    expect(existsSync(filepath)).toBe(false);
  });
});
