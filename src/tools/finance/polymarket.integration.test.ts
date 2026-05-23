/**
 * Live integration tests for the Polymarket tool.
 * These hit the real Gamma API (no key required).
 *
 * Run with:  bun test --filter polymarket.integration
 * Skipped in normal `bun test` runs because they make real network calls.
 */
import { describe, expect } from 'bun:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { polymarketTool, fetchPolymarketMarkets } from './polymarket.js';
import { readSnapshotRecords } from './polymarket-snapshots.js';
import { integrationIt as maybeIt } from '@/utils/test-guards.js';

describe('Polymarket integration (live API)', () => {
  maybeIt('returns real prediction markets for a finance query', async () => {
    const result = await polymarketTool.invoke({ query: 'Federal Reserve interest rates', limit: 5 });
    const text = typeof result === 'string' ? result : (result as { data: { result: string } }).data.result;

    // Should contain either market data or a clear empty/error message.
    // Polymarket markets use varying outcome labels so we check for % signs
    // (present in every probability line) rather than a specific label.
    const hasMarket = text.includes('%') || text.includes('No active Polymarket') || text.includes('search failed');
    expect(hasMarket).toBe(true);

    // Source footer should be present if we got results
    if (text.includes('%')) {
      expect(text).toContain('polymarket.com');
    }
  }, 15_000);

  maybeIt('returns results for a geopolitical query', async () => {
    const result = await polymarketTool.invoke({ query: 'US recession 2026', limit: 3 });
    const text = typeof result === 'string' ? result : (result as { data: { result: string } }).data.result;
    expect(text).toContain('Polymarket');
    expect(typeof text).toBe('string');
    expect(text.length).toBeGreaterThan(20);
  }, 15_000);

  maybeIt('respects the limit parameter', async () => {
    const result = await polymarketTool.invoke({ query: 'election', limit: 2 });
    const text = typeof result === 'string' ? result : (result as { data: { result: string } }).data.result;
    // Count "▸" bullets — should be at most 2
    const bulletCount = (text.match(/▸/g) ?? []).length;
    expect(bulletCount).toBeLessThanOrEqual(2);
  }, 15_000);

  maybeIt('gracefully handles an obscure query with no matching markets', async () => {
    const result = await polymarketTool.invoke({ query: 'zxqwerty12345nomarket', limit: 3 });
    const text = typeof result === 'string' ? result : (result as { data: { result: string } }).data.result;
    // Either no results message or valid market data — never throws
    expect(typeof text).toBe('string');
  }, 15_000);

  maybeIt('writes one snapshot record per returned market during fetch', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'polymarket-live-snapshots-'));
    const snapshotFilePath = join(tmpDir, 'polymarket-snapshots.jsonl');

    try {
      const capturedAt = '2026-04-20T14:30:00.000Z';
      const markets = await fetchPolymarketMarkets('Federal Reserve interest rates', 3, {
        snapshotFilePath,
        capturedAt,
      });

      const records = readSnapshotRecords(snapshotFilePath);
      expect(records.length).toBe(markets.length);

      for (let index = 0; index < markets.length; index++) {
        const market = markets[index]!;
        const record = records[index]!;
        if (market.marketId === undefined) {
          throw new Error('Expected fetched market to include a marketId');
        }
        expect(record.marketId).toBe(market.marketId);
        expect(record.probability).toBeCloseTo(market.probability, 6);
        expect(record.capturedAt).toBe(capturedAt);
      }
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  }, 15_000);
});
