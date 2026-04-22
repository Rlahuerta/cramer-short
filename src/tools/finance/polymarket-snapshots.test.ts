import { describe, expect, it } from 'bun:test';
import {
  findSnapshotInWindow,
  parseSnapshotLine,
  type PolymarketSnapshotRecord,
} from './polymarket-snapshots.js';

const NOW = new Date('2026-04-20T12:00:00.000Z').getTime();

function snapshot(hoursAgo: number, overrides: Partial<PolymarketSnapshotRecord> = {}): PolymarketSnapshotRecord {
  return {
    marketId: overrides.marketId ?? 'market-1',
    question: overrides.question ?? 'Will BTC reach $150k by end of 2026?',
    probability: overrides.probability ?? 0.42,
    volume24h: overrides.volume24h ?? 55_000,
    endDate: overrides.endDate ?? '2026-12-31T23:59:59Z',
    capturedAt: overrides.capturedAt ?? new Date(NOW - hoursAgo * 3_600_000).toISOString(),
  };
}

describe('parseSnapshotLine', () => {
  it('returns a snapshot record for valid JSONL input', () => {
    const expected = snapshot(3);
    expect(parseSnapshotLine(JSON.stringify(expected))).toEqual(expected);
  });

  it('returns null for malformed JSON input', () => {
    expect(parseSnapshotLine('{ not valid json')).toBeNull();
  });

  it('returns null when required fields are missing or invalid', () => {
    expect(parseSnapshotLine(JSON.stringify({ marketId: 'm1', probability: 0.4 }))).toBeNull();
    expect(parseSnapshotLine(JSON.stringify({
      marketId: 'm1',
      question: 'Q',
      probability: 1.2,
      volume24h: 10,
      endDate: '2026-12-31T23:59:59Z',
      capturedAt: '2026-04-20T12:00:00.000Z',
    }))).toBeNull();
  });

  it('accepts empty string endDate (market with no end date)', () => {
    const record = snapshot(3, { endDate: '' });
    expect(parseSnapshotLine(JSON.stringify(record))).toEqual(record);
  });

  it('rejects non-parseable endDate string', () => {
    const record = { ...snapshot(3), endDate: 'not-a-date' };
    expect(parseSnapshotLine(JSON.stringify(record))).toBeNull();
  });

  it('accepts valid ISO endDate string', () => {
    const record = snapshot(3, { endDate: '2026-12-31T23:59:59Z' });
    expect(parseSnapshotLine(JSON.stringify(record))).toEqual(record);
  });
});

describe('findSnapshotInWindow', () => {
  const records: PolymarketSnapshotRecord[] = [
    snapshot(49, { marketId: 'market-1', probability: 0.10 }),
    snapshot(25, { marketId: 'market-1', probability: 0.20 }),
    snapshot(3.5, { marketId: 'market-1', probability: 0.30 }),
    snapshot(3, { marketId: 'market-1', probability: 0.35 }),
    snapshot(1, { marketId: 'market-1', probability: 0.40 }),
    snapshot(3, { marketId: 'market-2', probability: 0.55 }),
  ];

  it('returns the most recent matching snapshot inside the 2-4h window', () => {
    const found = findSnapshotInWindow(records, 'market-1', NOW - 4 * 3_600_000, NOW - 2 * 3_600_000);
    expect(found?.probability).toBe(0.35);
  });

  it('matches by marketId rather than question text', () => {
    const found = findSnapshotInWindow(records, 'market-2', NOW - 4 * 3_600_000, NOW - 2 * 3_600_000);
    expect(found?.marketId).toBe('market-2');
    expect(found?.probability).toBe(0.55);
  });

  it('returns the 24-48h snapshot for persistence queries', () => {
    const found = findSnapshotInWindow(records, 'market-1', NOW - 48 * 3_600_000, NOW - 24 * 3_600_000);
    expect(found?.probability).toBe(0.20);
  });

  it('returns undefined when no matching snapshot exists in the window', () => {
    const found = findSnapshotInWindow(records, 'market-1', NOW - 12 * 3_600_000, NOW - 8 * 3_600_000);
    expect(found).toBeUndefined();
  });
});
