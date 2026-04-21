import { describe, expect } from 'bun:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { integrationIt as maybeIt } from '@/utils/test-guards.js';
import {
  appendSnapshotRecord,
  readSnapshotRecords,
  type PolymarketSnapshotRecord,
} from './polymarket-snapshots.js';

function record(id: string, capturedAt: string): PolymarketSnapshotRecord {
  return {
    marketId: id,
    question: `Question for ${id}`,
    probability: 0.42,
    volume24h: 55_000,
    endDate: '2026-12-31T23:59:59Z',
    capturedAt,
  };
}

describe('polymarket snapshot store integration', () => {
  maybeIt('writes records to JSONL and reads them back', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'polymarket-snapshots-'));
    const filePath = join(tmpDir, 'polymarket-snapshots.jsonl');

    try {
      appendSnapshotRecord(filePath, record('m1', '2026-04-20T09:00:00.000Z'));
      appendSnapshotRecord(filePath, record('m2', '2026-04-20T10:00:00.000Z'));
      appendSnapshotRecord(filePath, record('m3', '2026-04-20T11:00:00.000Z'));

      const records = readSnapshotRecords(filePath);
      expect(records).toHaveLength(3);
      expect(records.map((entry) => entry.marketId)).toEqual(['m1', 'm2', 'm3']);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  maybeIt('appends subsequent records instead of overwriting the file', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'polymarket-snapshots-'));
    const filePath = join(tmpDir, 'polymarket-snapshots.jsonl');

    try {
      appendSnapshotRecord(filePath, record('m1', '2026-04-20T09:00:00.000Z'));
      appendSnapshotRecord(filePath, record('m2', '2026-04-20T10:00:00.000Z'));
      appendSnapshotRecord(filePath, record('m3', '2026-04-20T11:00:00.000Z'));
      appendSnapshotRecord(filePath, record('m4', '2026-04-20T12:00:00.000Z'));

      const records = readSnapshotRecords(filePath);
      expect(records).toHaveLength(4);
      expect(records[3]?.marketId).toBe('m4');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  maybeIt('skips malformed lines when reading', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'polymarket-snapshots-'));
    const filePath = join(tmpDir, 'polymarket-snapshots.jsonl');

    try {
      appendSnapshotRecord(filePath, record('m1', '2026-04-20T09:00:00.000Z'));
      await writeFile(filePath, `${await readFile(filePath, 'utf-8')}not-json\n`, 'utf-8');
      appendSnapshotRecord(filePath, record('m2', '2026-04-20T10:00:00.000Z'));

      const records = readSnapshotRecords(filePath);
      expect(records).toHaveLength(2);
      expect(records.map((entry) => entry.marketId)).toEqual(['m1', 'm2']);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  maybeIt('treats a missing file as an empty store and creates it on first write', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'polymarket-snapshots-'));
    const filePath = join(tmpDir, 'polymarket-snapshots.jsonl');

    try {
      expect(readSnapshotRecords(filePath)).toEqual([]);
      appendSnapshotRecord(filePath, record('m1', '2026-04-20T09:00:00.000Z'));
      expect(readSnapshotRecords(filePath)).toHaveLength(1);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});
