import { describe, expect, it } from 'bun:test';
import {
  resolveTimeoutSeconds,
  resolveCacheTtlMs,
  normalizeCacheKey,
  readCache,
  writeCache,
  withTimeout,
  readResponseText,
  DEFAULT_TIMEOUT_SECONDS,
  DEFAULT_CACHE_TTL_MINUTES,
  type CacheEntry,
} from './cache.js';

// ---------------------------------------------------------------------------
// resolveTimeoutSeconds
// ---------------------------------------------------------------------------

describe('resolveTimeoutSeconds', () => {
  it('returns the value when it is a valid positive number', () => {
    expect(resolveTimeoutSeconds(30, 10)).toBe(30);
  });

  it('uses fallback when value is NaN', () => {
    expect(resolveTimeoutSeconds(NaN, 10)).toBe(10);
  });

  it('uses fallback when value is not a number', () => {
    expect(resolveTimeoutSeconds('abc', 10)).toBe(10);
  });

  it('uses fallback when value is undefined', () => {
    expect(resolveTimeoutSeconds(undefined, 15)).toBe(15);
  });

  it('clamps minimum to 1', () => {
    expect(resolveTimeoutSeconds(0, 10)).toBe(1);
    expect(resolveTimeoutSeconds(-5, 10)).toBe(1);
  });

  it('floors decimal values', () => {
    expect(resolveTimeoutSeconds(9.9, 10)).toBe(9);
  });

  it('exposes DEFAULT_TIMEOUT_SECONDS constant', () => {
    expect(typeof DEFAULT_TIMEOUT_SECONDS).toBe('number');
    expect(DEFAULT_TIMEOUT_SECONDS).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// resolveCacheTtlMs
// ---------------------------------------------------------------------------

describe('resolveCacheTtlMs', () => {
  it('converts minutes to milliseconds', () => {
    expect(resolveCacheTtlMs(1, 15)).toBe(60_000);
    expect(resolveCacheTtlMs(15, 15)).toBe(900_000);
  });

  it('returns 0 ms for 0 minutes', () => {
    expect(resolveCacheTtlMs(0, 15)).toBe(0);
  });

  it('uses fallback when value is not a number', () => {
    expect(resolveCacheTtlMs('bad', 5)).toBe(300_000);
  });

  it('uses fallback for NaN', () => {
    expect(resolveCacheTtlMs(NaN, 10)).toBe(600_000);
  });

  it('clamps negative values to 0', () => {
    expect(resolveCacheTtlMs(-5, 10)).toBe(0);
  });

  it('exposes DEFAULT_CACHE_TTL_MINUTES constant', () => {
    expect(typeof DEFAULT_CACHE_TTL_MINUTES).toBe('number');
    expect(DEFAULT_CACHE_TTL_MINUTES).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// normalizeCacheKey
// ---------------------------------------------------------------------------

describe('normalizeCacheKey', () => {
  it('trims leading and trailing whitespace', () => {
    expect(normalizeCacheKey('  hello  ')).toBe('hello');
  });

  it('lowercases the key', () => {
    expect(normalizeCacheKey('Hello World')).toBe('hello world');
  });

  it('trims and lowercases together', () => {
    expect(normalizeCacheKey('  AAPL  ')).toBe('aapl');
  });
});

// ---------------------------------------------------------------------------
// readCache
// ---------------------------------------------------------------------------

describe('readCache', () => {
  it('returns null for a missing key', () => {
    const cache = new Map<string, CacheEntry<string>>();
    expect(readCache(cache, 'missing')).toBeNull();
  });

  it('returns the cached value for a valid (unexpired) entry', () => {
    const cache = new Map<string, CacheEntry<string>>();
    cache.set('key', { value: 'hello', expiresAt: Date.now() + 60_000, insertedAt: Date.now() });
    const result = readCache(cache, 'key');
    expect(result?.value).toBe('hello');
    expect(result?.cached).toBe(true);
  });

  it('returns null and deletes the entry for an expired key', () => {
    const cache = new Map<string, CacheEntry<string>>();
    cache.set('key', { value: 'stale', expiresAt: Date.now() - 1, insertedAt: Date.now() - 1000 });
    const result = readCache(cache, 'key');
    expect(result).toBeNull();
    expect(cache.has('key')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// writeCache
// ---------------------------------------------------------------------------

describe('writeCache', () => {
  it('writes a value to the cache', () => {
    const cache = new Map<string, CacheEntry<string>>();
    writeCache(cache, 'k', 'v', 60_000);
    expect(cache.has('k')).toBe(true);
    expect(cache.get('k')?.value).toBe('v');
  });

  it('does NOT write when ttlMs <= 0', () => {
    const cache = new Map<string, CacheEntry<string>>();
    writeCache(cache, 'k', 'v', 0);
    expect(cache.has('k')).toBe(false);
  });

  it('evicts the oldest entry when cache is full (100 entries)', () => {
    const cache = new Map<string, CacheEntry<string>>();
    for (let i = 0; i < 100; i++) {
      cache.set(`key-${i}`, { value: `v${i}`, expiresAt: Date.now() + 60_000, insertedAt: Date.now() });
    }
    const firstKey = cache.keys().next().value;
    writeCache(cache, 'new-key', 'new-value', 60_000);
    expect(cache.size).toBe(100); // still 100 after eviction + insert
    expect(cache.has(firstKey!)).toBe(false); // oldest evicted
    expect(cache.has('new-key')).toBe(true);
  });

  it('sets expiresAt approximately ttlMs from now', () => {
    const cache = new Map<string, CacheEntry<string>>();
    const before = Date.now();
    writeCache(cache, 'k', 'v', 5_000);
    const entry = cache.get('k')!;
    expect(entry.expiresAt).toBeGreaterThanOrEqual(before + 4_900);
    expect(entry.expiresAt).toBeLessThanOrEqual(before + 5_100);
  });
});

// ---------------------------------------------------------------------------
// withTimeout
// ---------------------------------------------------------------------------

describe('withTimeout', () => {
  it('returns a signal that aborts after the timeout', async () => {
    const signal = withTimeout(undefined, 50);
    expect(signal.aborted).toBe(false);
    await new Promise((r) => setTimeout(r, 80));
    expect(signal.aborted).toBe(true);
  });

  it('returns a signal immediately when timeoutMs <= 0 and no signal is given', () => {
    const signal = withTimeout(undefined, 0);
    expect(signal).toBeDefined();
  });

  it('passes through an existing signal when timeout is 0', () => {
    const ctrl = new AbortController();
    const signal = withTimeout(ctrl.signal, 0);
    expect(signal).toBe(ctrl.signal);
  });

  it('aborts when the parent signal aborts before the timeout', async () => {
    const ctrl = new AbortController();
    const signal = withTimeout(ctrl.signal, 10_000);
    expect(signal.aborted).toBe(false);
    ctrl.abort();
    // Allow microtask queue to settle
    await new Promise((r) => setTimeout(r, 10));
    expect(signal.aborted).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// readResponseText
// ---------------------------------------------------------------------------

describe('readResponseText', () => {
  it('returns the response text on success', async () => {
    const res = new Response('hello world', { status: 200 });
    expect(await readResponseText(res)).toBe('hello world');
  });

  it('returns empty string when res.text() throws', async () => {
    const bad = { text: async () => { throw new Error('fail'); } } as unknown as Response;
    expect(await readResponseText(bad)).toBe('');
  });
});
