/**
 * P2c — Snapshot file pruning + bid/ask + map index.
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §11.2
 *
 * Pure helpers under test:
 *   - pruneSnapshots(records, n=3)        keep top-N most recent per marketId
 *   - buildSnapshotIndex(records)         Map<marketId, sorted records>
 *   - parseSnapshotLine                   accepts optional bid/ask fields
 *   - createSnapshotRecord                accepts optional bid/ask
 */

import { describe, expect, it } from 'bun:test';
import {
  buildSnapshotIndex,
  createSnapshotRecord,
  parseSnapshotLine,
  pruneSnapshots,
  type PolymarketSnapshotRecord,
} from './polymarket-snapshots.js';

function makeRecord(
  marketId: string,
  capturedAtIsoOrOffsetSec: string | number,
  extras: Partial<PolymarketSnapshotRecord> = {},
): PolymarketSnapshotRecord {
  const capturedAt =
    typeof capturedAtIsoOrOffsetSec === 'number'
      ? new Date(Date.UTC(2026, 0, 1, 0, 0, capturedAtIsoOrOffsetSec)).toISOString()
      : capturedAtIsoOrOffsetSec;
  return {
    marketId,
    question: 'q',
    probability: 0.5,
    volume24h: 100,
    endDate: '',
    capturedAt,
    ...extras,
  };
}

describe('pruneSnapshots', () => {
  it('keeps top-N most recent per marketId (default N=3)', () => {
    const records: PolymarketSnapshotRecord[] = [
      makeRecord('A', 1),
      makeRecord('A', 2),
      makeRecord('A', 3),
      makeRecord('A', 4),
      makeRecord('A', 5),
      makeRecord('B', 1),
      makeRecord('B', 2),
    ];
    const pruned = pruneSnapshots(records);
    const aRecords = pruned.filter((r) => r.marketId === 'A');
    expect(aRecords).toHaveLength(3);
    // Should be the 3 most-recent (offset 3, 4, 5)
    const offsets = aRecords.map((r) => new Date(r.capturedAt).getUTCSeconds()).sort();
    expect(offsets).toEqual([3, 4, 5]);

    const bRecords = pruned.filter((r) => r.marketId === 'B');
    expect(bRecords).toHaveLength(2);
  });

  it('respects custom N', () => {
    const records = [
      makeRecord('A', 1),
      makeRecord('A', 2),
      makeRecord('A', 3),
      makeRecord('A', 4),
    ];
    const pruned = pruneSnapshots(records, 2);
    expect(pruned).toHaveLength(2);
    const offsets = pruned.map((r) => new Date(r.capturedAt).getUTCSeconds()).sort();
    expect(offsets).toEqual([3, 4]);
  });

  it('handles empty input', () => {
    expect(pruneSnapshots([])).toEqual([]);
  });

  it('drops records with unparseable capturedAt', () => {
    const records = [
      makeRecord('A', 1),
      { ...makeRecord('A', 2), capturedAt: 'not-a-date' },
      makeRecord('A', 3),
    ];
    const pruned = pruneSnapshots(records);
    expect(pruned).toHaveLength(2);
  });
});

describe('buildSnapshotIndex', () => {
  it('groups records by marketId, sorted oldest→newest', () => {
    const records = [
      makeRecord('A', 3),
      makeRecord('B', 1),
      makeRecord('A', 1),
      makeRecord('A', 2),
      makeRecord('B', 5),
    ];
    const idx = buildSnapshotIndex(records);
    expect(idx.size).toBe(2);
    const aRecs = idx.get('A')!;
    expect(aRecs).toHaveLength(3);
    const aOffsets = aRecs.map((r) => new Date(r.capturedAt).getUTCSeconds());
    expect(aOffsets).toEqual([1, 2, 3]);
  });

  it('returns empty Map for empty input', () => {
    const idx = buildSnapshotIndex([]);
    expect(idx.size).toBe(0);
  });

  it('skips records with bad timestamps', () => {
    const records = [
      makeRecord('A', 1),
      { ...makeRecord('A', 2), capturedAt: 'bad' },
    ];
    const idx = buildSnapshotIndex(records);
    expect(idx.get('A')).toHaveLength(1);
  });
});

describe('parseSnapshotLine — bid/ask fields', () => {
  it('parses optional bid and ask when present', () => {
    const line = JSON.stringify({
      marketId: 'm1',
      question: 'q?',
      probability: 0.5,
      volume24h: 1000,
      endDate: '2026-12-31T00:00:00Z',
      capturedAt: '2026-01-01T00:00:00Z',
      bid: 0.48,
      ask: 0.52,
    });
    const rec = parseSnapshotLine(line);
    expect(rec?.bid).toBeCloseTo(0.48, 6);
    expect(rec?.ask).toBeCloseTo(0.52, 6);
  });

  it('omits bid/ask when absent (backward compatible)', () => {
    const line = JSON.stringify({
      marketId: 'm1',
      question: 'q?',
      probability: 0.5,
      volume24h: 1000,
      endDate: '2026-12-31T00:00:00Z',
      capturedAt: '2026-01-01T00:00:00Z',
    });
    const rec = parseSnapshotLine(line);
    expect(rec).not.toBeNull();
    expect(rec?.bid).toBeUndefined();
    expect(rec?.ask).toBeUndefined();
  });

  it('rejects out-of-range bid/ask', () => {
    const line = JSON.stringify({
      marketId: 'm1',
      question: 'q?',
      probability: 0.5,
      volume24h: 1000,
      endDate: '',
      capturedAt: '2026-01-01T00:00:00Z',
      bid: 1.5,
    });
    const rec = parseSnapshotLine(line);
    // Invalid bid silently dropped (record still valid), or rejected outright.
    // Spec choice: reject the bid but keep the record.
    expect(rec).not.toBeNull();
    expect(rec?.bid).toBeUndefined();
  });
});

describe('createSnapshotRecord — bid/ask passthrough', () => {
  it('forwards bid/ask when supplied', () => {
    const rec = createSnapshotRecord(
      {
        marketId: 'm1',
        question: 'q',
        probability: 0.5,
        volume24h: 100,
        endDate: '',
        bid: 0.48,
        ask: 0.52,
      },
      '2026-01-01T00:00:00Z',
    );
    expect(rec?.bid).toBeCloseTo(0.48, 6);
    expect(rec?.ask).toBeCloseTo(0.52, 6);
  });

  it('works without bid/ask', () => {
    const rec = createSnapshotRecord(
      { marketId: 'm1', question: 'q', probability: 0.5, volume24h: 100, endDate: '' },
      '2026-01-01T00:00:00Z',
    );
    expect(rec?.bid).toBeUndefined();
    expect(rec?.ask).toBeUndefined();
  });
});
