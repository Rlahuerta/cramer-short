import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { cramerShortPath } from '../../utils/paths.js';

export interface PolymarketSnapshotRecord {
  marketId: string;
  question: string;
  probability: number;
  volume24h: number;
  endDate: string;
  capturedAt: string;
}

export const DEFAULT_POLYMARKET_SNAPSHOTS_PATH = cramerShortPath('polymarket-snapshots.jsonl');

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function normalizeTimestamp(value: number | string | Date): number {
  if (typeof value === 'number') return value;
  if (value instanceof Date) return value.getTime();
  return Date.parse(value);
}

export function parseSnapshotLine(rawLine: string): PolymarketSnapshotRecord | null {
  const trimmed = rawLine.trim();
  if (!trimmed) return null;

  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    const marketId = parsed['marketId'];
    const question = parsed['question'];
    const probability = parsed['probability'];
    const volume24h = parsed['volume24h'];
    const endDate = parsed['endDate'];
    const capturedAt = parsed['capturedAt'];

    if (typeof marketId !== 'string' || marketId.trim().length === 0) return null;
    if (typeof question !== 'string') return null;
    if (!isFiniteNumber(probability) || probability < 0 || probability > 1) return null;
    if (!isFiniteNumber(volume24h) || volume24h < 0) return null;
    if (typeof endDate !== 'string') return null;
    if (endDate !== '' && !Number.isFinite(Date.parse(endDate))) return null;
    if (typeof capturedAt !== 'string' || !Number.isFinite(Date.parse(capturedAt))) return null;

    return {
      marketId,
      question,
      probability,
      volume24h,
      endDate,
      capturedAt,
    };
  } catch {
    return null;
  }
}

export function findSnapshotInWindow(
  records: PolymarketSnapshotRecord[],
  marketId: string,
  windowStart: number | string | Date,
  windowEnd: number | string | Date,
): PolymarketSnapshotRecord | undefined {
  const startMs = normalizeTimestamp(windowStart);
  const endMs = normalizeTimestamp(windowEnd);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || marketId.trim().length === 0) {
    return undefined;
  }

  let latest: PolymarketSnapshotRecord | undefined;
  let latestMs = Number.NEGATIVE_INFINITY;

  for (const record of records) {
    if (record.marketId !== marketId) continue;
    const capturedMs = Date.parse(record.capturedAt);
    if (!Number.isFinite(capturedMs)) continue;
    if (capturedMs < startMs || capturedMs > endMs) continue;
    if (capturedMs > latestMs) {
      latest = record;
      latestMs = capturedMs;
    }
  }

  return latest;
}

export function readSnapshotRecords(
  filePath: string = DEFAULT_POLYMARKET_SNAPSHOTS_PATH,
  marketId?: string,
): PolymarketSnapshotRecord[] {
  if (!existsSync(filePath)) return [];

  const content = readFileSync(filePath, 'utf-8');
  if (!content.trim()) return [];

  const records: PolymarketSnapshotRecord[] = [];
  const lines = content.split(/\r?\n/);

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (!line || !line.trim()) continue;

    const parsed = parseSnapshotLine(line);
    if (!parsed) {
      console.warn(
        `[polymarket-snapshots] Skipping malformed snapshot line ${index + 1} in ${filePath}`,
      );
      continue;
    }

    if (marketId && parsed.marketId !== marketId) continue;
    records.push(parsed);
  }

  return records;
}

export function appendSnapshotRecord(
  filePath: string = DEFAULT_POLYMARKET_SNAPSHOTS_PATH,
  record: PolymarketSnapshotRecord,
): void {
  mkdirSync(dirname(filePath), { recursive: true });
  appendFileSync(filePath, `${JSON.stringify(record)}\n`, 'utf-8');
}

export function appendSnapshotRecords(
  filePath: string = DEFAULT_POLYMARKET_SNAPSHOTS_PATH,
  records: PolymarketSnapshotRecord[],
): void {
  if (records.length === 0) return;
  mkdirSync(dirname(filePath), { recursive: true });
  const payload = `${records.map((record) => JSON.stringify(record)).join('\n')}\n`;
  appendFileSync(filePath, payload, 'utf-8');
}

export function createSnapshotRecord(
  market: {
    marketId?: string;
    question: string;
    probability: number;
    volume24h: number;
    endDate?: string | null;
  },
  capturedAt: string = new Date().toISOString(),
): PolymarketSnapshotRecord | null {
  if (!market.marketId) return null;

  return {
    marketId: market.marketId,
    question: market.question,
    probability: market.probability,
    volume24h: market.volume24h,
    endDate: market.endDate ?? '',
    capturedAt,
  };
}
