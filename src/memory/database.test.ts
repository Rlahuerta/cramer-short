import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { MemoryDatabase } from './database.js';
import type { MemoryChunk } from './types.js';

const TEST_DIR = join(process.cwd(), '.cramer-short-test', 'memory-database');
const DB_PATH = join(TEST_DIR, 'index.sqlite');

let db: MemoryDatabase;

function chunk(overrides: Partial<MemoryChunk> = {}): MemoryChunk {
  return {
    filePath: overrides.filePath ?? 'notes/aapl.md',
    startLine: overrides.startLine ?? 1,
    endLine: overrides.endLine ?? 3,
    content: overrides.content ?? 'AAPL operating leverage improved as services margin expanded.',
    contentHash: overrides.contentHash ?? 'chunk-aapl-operating-leverage',
    source: overrides.source ?? 'memory',
  };
}

beforeEach(async () => {
  await rm(TEST_DIR, { recursive: true, force: true });
  await mkdir(TEST_DIR, { recursive: true });
  db = await MemoryDatabase.create(DB_PATH);
});

afterEach(async () => {
  db.close();
  await rm(TEST_DIR, { recursive: true, force: true });
});

describe('MemoryDatabase chunk indexing', () => {
  it('round-trips chunks through keyword search, ordered id loading, and metadata helpers', () => {
    const first = db.upsertChunk({
      chunk: chunk(),
      embedding: null,
      provider: 'local',
      model: 'none',
    });
    const second = db.upsertChunk({
      chunk: chunk({
        filePath: 'notes/nvda.md',
        content: 'NVDA datacenter demand remained strong.',
        contentHash: 'chunk-nvda-demand',
        source: 'sessions',
      }),
      embedding: null,
    });

    const keyword = db.searchKeyword('operating leverage', 5);
    expect(keyword.map((candidate) => candidate.chunkId)).toContain(first.id);

    const loaded = db.loadResultsByIds([second.id, first.id]);
    expect(loaded.map((result) => result.path)).toEqual(['notes/nvda.md', 'notes/aapl.md']);
    expect(loaded[0]?.contentSource).toBe('sessions');
    expect(loaded[1]?.tickers).toContain('AAPL');

    expect(db.listIndexedFiles().sort()).toEqual(['notes/aapl.md', 'notes/nvda.md']);
    expect(db.getChunkTickers([first.id, second.id]).get(second.id)).toContain('NVDA');
  });

  it('uses deterministic local vector ranking whether sqlite-vec is loaded or JS fallback is used', () => {
    db.upsertChunk({
      chunk: chunk({ content: 'AAPL earnings quality', contentHash: 'vec-aapl' }),
      embedding: [1, 0, 0],
    });
    db.upsertChunk({
      chunk: chunk({ content: 'TSLA deliveries', contentHash: 'vec-tsla' }),
      embedding: [0, 1, 0],
    });
    db.upsertChunk({
      chunk: chunk({ content: 'NVDA chips', contentHash: 'vec-nvda' }),
      embedding: [0, 0, 1],
    });

    const results = db.searchVector([0.95, 0.05, 0], 2);
    expect(results).toHaveLength(2);
    expect(results[0]!.score).toBeGreaterThanOrEqual(results[1]!.score);

    const [top] = db.loadResultsByIds([results[0]!.chunkId]);
    expect(top?.snippet).toContain('AAPL');
    expect(typeof db.isVecEnabled).toBe('boolean');
  });

  it('stores embedding cache and provider fingerprint without external services', () => {
    expect(db.getCachedEmbedding('hash-one')).toBeNull();

    db.setCachedEmbedding({
      contentHash: 'hash-one',
      embedding: [0.25, 0.5, 0.75],
      provider: 'local-test',
      model: 'unit-vector',
    });
    expect(db.getCachedEmbedding('hash-one')).toEqual([0.25, 0.5, 0.75]);

    expect(db.getProviderFingerprint()).toBeNull();
    db.setProviderFingerprint('local-test:unit-vector');
    expect(db.getProviderFingerprint()).toBe('local-test:unit-vector');
  });
});

describe('MemoryDatabase vector table lifecycle', () => {
  it('clearEmbeddings removes stale vec_chunks rows (same-dim provider switch)', () => {
    db.upsertChunk({
      chunk: chunk({ content: 'old-model AAPL vector', contentHash: 'stale-a' }),
      embedding: [1, 0, 0, 0],
      provider: 'old',
      model: 'old',
    });
    db.upsertChunk({
      chunk: chunk({ content: 'old-model TSLA vector', contentHash: 'stale-t' }),
      embedding: [0, 1, 0, 0],
      provider: 'old',
      model: 'old',
    });
    // Sanity: the old-model vectors are searchable before the switch.
    expect(db.searchVector([1, 0, 0, 0], 5).length).toBeGreaterThan(0);

    db.clearEmbeddings();

    // A provider switch must not resurrect old-model vectors from the vec table.
    expect(db.searchVector([1, 0, 0, 0], 5)).toEqual([]);
  });

  it('recreates vec_chunks when the embedding dimension changes', () => {
    db.upsertChunk({
      chunk: chunk({ content: 'dim4 old vector', contentHash: 'dim4' }),
      embedding: [1, 0, 0, 0],
    });
    db.clearEmbeddings();

    const eight = [1, 0, 0, 0, 0, 0, 0, 0];
    const inserted = db.upsertChunk({
      chunk: chunk({ content: 'dim8 new vector', contentHash: 'dim8' }),
      embedding: eight,
    });

    const raw = (db as unknown as { db: { query: <T>(sql: string) => { get(): T | null } } }).db;
    const tableSql =
      raw.query<{ sql: string }>("SELECT sql FROM sqlite_master WHERE name = 'vec_chunks'").get()?.sql ?? '';
    expect(tableSql).toContain('float[8]');

    const results = db.searchVector(eight, 5);
    expect(results.map((result) => result.chunkId)).toEqual([inserted.id]);
  });
});

describe('MemoryDatabase.loadResultsByIds id alignment', () => {
  it('returns entries carrying their own id when some ids are missing', () => {
    const first = db.upsertChunk({
      chunk: chunk({ content: 'AAPL note', contentHash: 'keep-first' }),
      embedding: null,
    });
    const second = db.upsertChunk({
      chunk: chunk({ filePath: 'notes/nvda.md', content: 'NVDA demand', contentHash: 'keep-second' }),
      embedding: null,
    });

    const results = db.loadResultsByIds([first.id, 999_999, second.id]);

    expect(results.map((entry) => entry.id)).toEqual([first.id, second.id]);
    expect(results[1]?.snippet).toContain('NVDA');
  });
});

describe('MemoryDatabase.searchInsightsFts rank order', () => {
  it('keeps FTS rank order in the returned rows', () => {
    const weak = db.upsertInsight({
      ticker: 'AAPL',
      tags: '[]',
      content: 'alpha mention once',
      contentHash: 'rank-weak',
    });
    const strong = db.upsertInsight({
      ticker: 'MSFT',
      tags: '[]',
      content: 'alpha alpha alpha alpha alpha',
      contentHash: 'rank-strong',
    });

    const ranked = db.searchInsightsFts('alpha', 2);

    expect(ranked.map((row) => row.id)).toEqual([strong, weak]);
  });
});

describe('MemoryDatabase financial insights', () => {
  it('upserts insights, refreshes FTS rows, and searches tickers case-insensitively', () => {
    const id = db.upsertInsight({
      ticker: 'AAPL',
      tags: '["margin"]',
      content: 'oldphrase iPhone margin note',
      contentHash: 'insight-aapl-margin',
      namespace: 'test',
    });

    const updatedId = db.upsertInsight({
      ticker: 'aapl',
      exchange: 'NASDAQ',
      sector: 'Technology',
      tags: '["services"]',
      content: 'Services margin expansion supports cash generation.',
      contentHash: 'insight-aapl-margin',
      routing: 'unit',
      source: 'test-suite',
      namespace: 'updated',
    });

    expect(updatedId).toBe(id);
    expect(db.searchInsightsFts('oldphrase', 5)).toEqual([]);

    const [ftsMatch] = db.searchInsightsFts('services margin', 5);
    expect(ftsMatch?.id).toBe(id);
    expect(ftsMatch?.namespace).toBe('updated');

    const [tickerMatch] = db.searchInsightsByTicker('AAPL');
    expect(tickerMatch?.ticker).toBe('aapl');
    expect(tickerMatch?.exchange).toBe('NASDAQ');
  });

  it('stores knowledge graph edges between insight records', () => {
    const sourceId = db.upsertInsight({
      ticker: 'AAPL',
      tags: '[]',
      content: 'AAPL source insight',
      contentHash: 'edge-source',
    });
    const targetId = db.upsertInsight({
      ticker: 'MSFT',
      tags: '[]',
      content: 'MSFT target insight',
      contentHash: 'edge-target',
    });

    db.addEdge(sourceId, targetId, 'peer-readthrough', 0.8);
    const edges = db.getEdgesForInsight(sourceId);

    expect(edges).toHaveLength(1);
    expect(edges[0]).toMatchObject({
      source_id: sourceId,
      target_id: targetId,
      relation: 'peer-readthrough',
      confidence: 0.8,
    });
  });
});
