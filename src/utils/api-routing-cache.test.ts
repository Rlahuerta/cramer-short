import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { describe, test, expect, beforeEach, afterEach, spyOn, setSystemTime } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as paths from './paths.js';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

// We test the pure logic of the routing cache by importing the module-level functions.
// The in-memory singleton is reset between tests by manipulating module state.

// Pure helpers extracted from the module for isolated unit testing
const TTL_MS = 30 * 24 * 60 * 60 * 1000;

function isStale(updatedAt: string, nowMs = FIXED_TEST_NOW_MS): boolean {
  return nowMs - new Date(updatedAt).getTime() > TTL_MS;
}

function normalizeKey(ticker: string): string {
  return ticker.toUpperCase().trim();
}

describe('api-routing-cache helpers', () => {
  describe('normalizeKey', () => {
    test('uppercases ticker', () => {
      expect(normalizeKey('aapl')).toBe('AAPL');
      expect(normalizeKey('NVDA')).toBe('NVDA');
    });

    test('trims whitespace', () => {
      expect(normalizeKey('  TSLA  ')).toBe('TSLA');
    });

    test('handles dot-notation tickers', () => {
      expect(normalizeKey('vws.co')).toBe('VWS.CO');
      expect(normalizeKey('san.mc')).toBe('SAN.MC');
    });
  });

  describe('isStale', () => {
    test('returns false for entry updated 1 day ago', () => {
      const updatedAt = new Date(FIXED_TEST_NOW_MS - 1 * 24 * 60 * 60 * 1000).toISOString();
      expect(isStale(updatedAt)).toBe(false);
    });

    test('returns false for entry updated 29 days ago', () => {
      const updatedAt = new Date(FIXED_TEST_NOW_MS - 29 * 24 * 60 * 60 * 1000).toISOString();
      expect(isStale(updatedAt)).toBe(false);
    });

    test('returns true for entry updated 31 days ago', () => {
      const updatedAt = new Date(FIXED_TEST_NOW_MS - 31 * 24 * 60 * 60 * 1000).toISOString();
      expect(isStale(updatedAt)).toBe(true);
    });

    test('returns true for a very old timestamp', () => {
      const updatedAt = new Date(0).toISOString(); // epoch
      expect(isStale(updatedAt)).toBe(true);
    });

    test('returns false for entry updated right now', () => {
      const updatedAt = new Date(FIXED_TEST_NOW_MS).toISOString();
      expect(isStale(updatedAt)).toBe(false);
    });
  });
});

// The real module keeps a singleton cache and resolves its file path at load time, so each test
// re-imports it (cache-busting query) with getCramerShortDir pointed at a temp dir.
describe('api-routing-cache (real module)', () => {
  let tmpDir: string;
  let dirSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'api-routing-test-'));
    mkdirSync(tmpDir, { recursive: true });
    dirSpy = spyOn(paths, 'getCramerShortDir').mockReturnValue(tmpDir);
  });

  afterEach(() => {
    dirSpy.mockRestore();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  function writeRoutes(routes: Record<string, { preferred: string; updatedAt: string }>): string {
    const file = join(tmpDir, 'api-routing.json');
    writeFileSync(file, JSON.stringify({ version: 1, routes }));
    return file;
  }

  async function loadModule() {
    return (await import(`./api-routing-cache.js?t=${nextTestId('module')}`)) as typeof import('./api-routing-cache.js');
  }

  test('an entry whose updatedAt cannot be parsed is stale, like an expired one', async () => {
    writeRoutes({
      AAPL: { preferred: 'fmp', updatedAt: 'not-a-date' },
      MSFT: { preferred: 'yahoo', updatedAt: '2020-01-01T00:00:00.000Z' },
      NVDA: { preferred: 'yahoo', updatedAt: new Date(FIXED_TEST_NOW_MS).toISOString() },
    });
    const { getPreferredApi } = await loadModule();
    expect(await getPreferredApi('AAPL')).toBeNull();
    expect(await getPreferredApi('MSFT')).toBeNull();
    expect(await getPreferredApi('NVDA')).toBe('yahoo');
  });

  test('re-confirming the same backend refreshes a stale entry', async () => {
    const file = writeRoutes({ AAPL: { preferred: 'fmp', updatedAt: 'not-a-date' } });
    const { getPreferredApi, setPreferredApi } = await loadModule();
    expect(await getPreferredApi('AAPL')).toBeNull();

    await setPreferredApi('AAPL', 'fmp');
    expect(await getPreferredApi('AAPL')).toBe('fmp');

    // Let the fire-and-forget persist finish before the temp dir is removed.
    for (let i = 0; i < 100 && readFileSync(file, 'utf-8').includes('not-a-date'); i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    expect(readFileSync(file, 'utf-8')).not.toContain('not-a-date');
  });
});
