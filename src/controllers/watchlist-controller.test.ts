/**
 * WatchlistController tests — written BEFORE the implementation (TDD).
 *
 * All tests use a temporary directory so they never touch the real
 * .cramer-short/watchlist.json.
 */
import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { afterEach, beforeEach, describe, expect, it, setSystemTime } from 'bun:test';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { WatchlistController } from './watchlist-controller.js';

let tmpDir: string;
let ctrl: WatchlistController;

beforeEach(() => {
  tmpDir = join(tmpdir(), `dexter-wl-test-${nextTestId('path')}`);
  mkdirSync(tmpDir, { recursive: true });
  ctrl = new WatchlistController(tmpDir);
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------
describe('load()', () => {
  it('returns empty entries when file does not exist', () => {
    const data = ctrl.load();
    expect(data.entries).toEqual([]);
    expect(data.version).toBe(1);
  });

  it('returns persisted entries after a save', () => {
    ctrl.add('NVDA', 400, 100);
    const data = new WatchlistController(tmpDir).load();
    expect(data.entries).toHaveLength(1);
    expect(data.entries[0]!.ticker).toBe('NVDA');
  });
});

// ---------------------------------------------------------------------------
// load() malformed-file safety
// ---------------------------------------------------------------------------
describe('load() malformed-file safety', () => {
  const writeRaw = (raw: string): void => {
    const dir = join(tmpDir, '.cramer-short');
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'watchlist.json'), raw, 'utf-8');
  };

  it('returns empty entries (no throw) when the file parses to null', () => {
    writeRaw('null');
    expect(() => ctrl.load().entries).not.toThrow();
    expect(ctrl.load().entries).toEqual([]);
  });

  it('returns empty entries when the file parses to an object without entries', () => {
    writeRaw('{}');
    expect(ctrl.load().entries).toEqual([]);
  });

  it('returns empty entries when entries is not an array', () => {
    writeRaw('{"version":1,"entries":"nope"}');
    expect(ctrl.load().entries).toEqual([]);
  });

  it('warns on a malformed shape', () => {
    writeRaw('{}');
    const original = console.warn;
    let warned = 0;
    console.warn = () => { warned++; };
    try {
      ctrl.load();
    } finally {
      console.warn = original;
    }
    expect(warned).toBeGreaterThan(0);
  });

  it('list() and add() do not throw on a malformed file', () => {
    writeRaw('null');
    expect(() => ctrl.list()).not.toThrow();
    expect(() => ctrl.add('AAPL')).not.toThrow();
    expect(ctrl.list().some((e) => e.ticker === 'AAPL')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// add()
// ---------------------------------------------------------------------------
describe('add()', () => {
  it('creates watchlist.json if it does not exist', () => {
    const file = join(tmpDir, '.cramer-short', 'watchlist.json');
    expect(existsSync(file)).toBe(false);
    ctrl.add('AAPL');
    expect(existsSync(file)).toBe(true);
  });

  it('persists ticker with costBasis and shares', () => {
    ctrl.add('MSFT', 380, 50);
    const { entries } = ctrl.load();
    expect(entries[0]!.ticker).toBe('MSFT');
    expect(entries[0]!.costBasis).toBe(380);
    expect(entries[0]!.shares).toBe(50);
  });

  it('persists ticker without optional fields', () => {
    ctrl.add('TSLA');
    const { entries } = ctrl.load();
    expect(entries[0]!.costBasis).toBeUndefined();
    expect(entries[0]!.shares).toBeUndefined();
  });

  it('stores addedAt as ISO date string YYYY-MM-DD', () => {
    ctrl.add('GOOG');
    const { entries } = ctrl.load();
    expect(entries[0]!.addedAt).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it('deduplicates: re-adding same ticker updates fields, does not duplicate', () => {
    ctrl.add('NVDA', 400, 100);
    ctrl.add('NVDA', 450, 150);
    const { entries } = ctrl.load();
    expect(entries).toHaveLength(1);
    expect(entries[0]!.costBasis).toBe(450);
    expect(entries[0]!.shares).toBe(150);
  });

  it('ticker is stored in uppercase', () => {
    ctrl.add('nvda');
    const { entries } = ctrl.load();
    expect(entries[0]!.ticker).toBe('NVDA');
  });
});

// ---------------------------------------------------------------------------
// remove()
// ---------------------------------------------------------------------------
describe('remove()', () => {
  it('removes an existing ticker', () => {
    ctrl.add('AAPL');
    ctrl.add('MSFT');
    ctrl.remove('AAPL');
    const tickers = ctrl.list().map((e) => e.ticker);
    expect(tickers).not.toContain('AAPL');
    expect(tickers).toContain('MSFT');
  });

  it('is a no-op and does not throw when ticker is not in the list', () => {
    ctrl.add('AAPL');
    expect(() => ctrl.remove('TSLA')).not.toThrow();
    expect(ctrl.list()).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// list()
// ---------------------------------------------------------------------------
describe('list()', () => {
  it('returns entries sorted alphabetically by ticker', () => {
    ctrl.add('TSLA');
    ctrl.add('AAPL');
    ctrl.add('MSFT');
    const tickers = ctrl.list().map((e) => e.ticker);
    expect(tickers).toEqual(['AAPL', 'MSFT', 'TSLA']);
  });

  it('returns empty array when watchlist is empty', () => {
    expect(ctrl.list()).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// isEmpty()
// ---------------------------------------------------------------------------
describe('isEmpty()', () => {
  it('returns true when no entries', () => {
    expect(ctrl.isEmpty()).toBe(true);
  });

  it('returns false when at least one entry exists', () => {
    ctrl.add('NVDA');
    expect(ctrl.isEmpty()).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// round-trip
// ---------------------------------------------------------------------------
describe('round-trip persistence', () => {
  it('add on one instance, new instance returns same list', () => {
    ctrl.add('NVDA', 400, 100);
    ctrl.add('AAPL', 180, 50);
    ctrl.add('MSFT');

    const ctrl2 = new WatchlistController(tmpDir);
    const tickers = ctrl2.list().map((e) => e.ticker);
    expect(tickers).toContain('NVDA');
    expect(tickers).toContain('AAPL');
    expect(tickers).toContain('MSFT');
  });
});

// ---------------------------------------------------------------------------
// schema
// ---------------------------------------------------------------------------
describe('JSON schema', () => {
  it('saved file has version:1 and entries array', () => {
    ctrl.add('GOOG');
    const raw = JSON.parse(
      readFileSync(join(tmpDir, '.cramer-short', 'watchlist.json'), 'utf-8'),
    );
    expect(raw.version).toBe(1);
    expect(Array.isArray(raw.entries)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// parseWatchlistSubcommand (pure function — tested independently of CLI)
// ---------------------------------------------------------------------------
import { parseWatchlistSubcommand } from './watchlist-controller.js';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

describe('parseWatchlistSubcommand()', () => {
  it('returns { cmd: "briefing" } for bare /watchlist', () => {
    expect(parseWatchlistSubcommand('')).toEqual({ cmd: 'briefing' });
  });

  it('parses "add NVDA"', () => {
    expect(parseWatchlistSubcommand('add NVDA')).toEqual({
      cmd: 'add', ticker: 'NVDA', costBasis: undefined, shares: undefined,
    });
  });

  it('parses "add NVDA 400 100"', () => {
    expect(parseWatchlistSubcommand('add NVDA 400 100')).toEqual({
      cmd: 'add', ticker: 'NVDA', costBasis: 400, shares: 100,
    });
  });

  it('parses "remove AAPL"', () => {
    expect(parseWatchlistSubcommand('remove AAPL')).toEqual({
      cmd: 'remove', ticker: 'AAPL',
    });
  });

  it('parses "list"', () => {
    expect(parseWatchlistSubcommand('list')).toEqual({ cmd: 'list' });
  });

  it('is case-insensitive for subcommand', () => {
    expect(parseWatchlistSubcommand('ADD nvda 400')).toEqual({
      cmd: 'add', ticker: 'NVDA', costBasis: 400, shares: undefined,
    });
  });

  it('parses "show AAPL"', () => {
    expect(parseWatchlistSubcommand('show AAPL')).toEqual({ cmd: 'show', ticker: 'AAPL' });
  });

  it('parses "show" with lowercase ticker', () => {
    expect(parseWatchlistSubcommand('show nvda')).toEqual({ cmd: 'show', ticker: 'NVDA' });
  });

  it('returns briefing when "show" has no ticker', () => {
    expect(parseWatchlistSubcommand('show')).toEqual({ cmd: 'briefing' });
  });

  it('parses "snapshot"', () => {
    expect(parseWatchlistSubcommand('snapshot')).toEqual({ cmd: 'snapshot' });
  });

  it('parses "SNAPSHOT" case-insensitively', () => {
    expect(parseWatchlistSubcommand('SNAPSHOT')).toEqual({ cmd: 'snapshot' });
  });
});
