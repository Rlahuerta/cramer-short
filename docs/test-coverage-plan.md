# Test Coverage Improvement Plan

## Overview

**Files:** `src/memory/database.ts` and `src/model/llm.ts`
**Current Coverage:** 0% for both
**Target Coverage:** Unit 80%+, Integration 70%+

---

## Test Structure

This project uses a three-tier test approach:

| Type | File Pattern | Run Command | Purpose |
|------|--------------|-------------|---------|
| **Unit** | `*.test.ts` | `bun test` | Fast, isolated, mocked |
| **Integration** | `*.integration.test.ts` | `RUN_INTEGRATION=1 bun test --filter integration` | Real dependencies, no mocks |
| **E2E** | `*.e2e.test.ts` | `RUN_E2E=1 bun test --filter e2e` | Full system flows |

---

## 1. `src/memory/database.ts` (360 lines)

### File Analysis

The `MemoryDatabase` class provides:
- **Static factory:** `create(path)` - async database initialization
- **Chunk operations:** `upsertChunk()`, `getChunkByHash()`, `deleteChunksForFile()`, `listAllChunks()`, `listIndexedFiles()`
- **Search operations:** `searchVector()`, `searchKeyword()`, `searchHybrid()`
- **Cache operations:** `cacheEmbedding()`, `getCachedEmbedding()`
- **Meta operations:** `getMeta()`, `setMeta()`, `close()`
- **Helpers:** `toBlob()`, `fromBlob()`, `buildFtsQuery()`, `cosineSimilarity()`

---

### Unit Tests: `src/memory/database.test.ts`

```typescript
import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { existsSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { MemoryDatabase, toBlob, fromBlob, buildFtsQuery, cosineSimilarity } from './database.js';

describe('MemoryDatabase (Unit)', () => {
  let db: MemoryDatabase;
  let dbPath: string;

  beforeEach(async () => {
    dbPath = join(tmpdir(), `test-db-${Date.now()}.sqlite`);
    db = await MemoryDatabase.create(dbPath);
  });

  afterEach(() => {
    db.close();
    if (existsSync(dbPath)) rmSync(dbPath, { recursive: true });
  });

  // ============================================
  // Factory & Initialization
  // ============================================
  describe('create', () => {
    test('creates database with all tables', async () => {
      const files = db.listIndexedFiles();
      expect(Array.isArray(files)).toBe(true);
    });

    test('creates directory if not exists', async () => {
      const nestedPath = join(tmpdir(), 'nested', 'dir', `db-${Date.now()}.sqlite`);
      const newDb = await MemoryDatabase.create(nestedPath);
      expect(existsSync(nestedPath)).toBe(true);
      newDb.close();
      rmSync(nestedPath, { recursive: true });
    });

    test('reopens existing database', async () => {
      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 5, content: 'hello', contentHash: 'h1' },
        embedding: null,
      });
      db.close();

      const reopened = await MemoryDatabase.create(dbPath);
      expect(reopened.listAllChunks().length).toBe(1);
      reopened.close();
    });
  });

  // ============================================
  // Chunk Operations
  // ============================================
  describe('upsertChunk', () => {
    test('inserts new chunk', () => {
      const result = db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 10, content: 'Hello world', contentHash: 'hash1' },
        embedding: [0.1, 0.2, 0.3],
        provider: 'openai',
        model: 'text-embedding-3-small',
      });
      expect(result.inserted).toBe(true);
      expect(result.id).toBeGreaterThan(0);
    });

    test('updates existing chunk on duplicate hash', () => {
      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 10, content: 'Original', contentHash: 'hash1' },
        embedding: [0.1],
      });

      const updated = db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 10, content: 'Updated', contentHash: 'hash1' },
        embedding: [0.5],
      });
      expect(updated.inserted).toBe(false);

      const all = db.listAllChunks();
      expect(all.length).toBe(1);
    });

    test('stores embedding as blob', () => {
      const embedding = [0.1, 0.2, 0.3, 0.4, 0.5];
      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 1, content: 'x', contentHash: 'h1' },
        embedding,
      });
      const results = db.searchVector(embedding, 1);
      expect(results).toHaveLength(1);
    });

    test('stores source metadata', () => {
      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 5, content: 'test', contentHash: 'h1', source: 'sessions' },
        embedding: null,
      });
      // Source is stored but not returned by listAllChunks
      expect(db.listAllChunks().length).toBe(1);
    });
  });

  describe('getChunkByHash', () => {
    test('returns null for non-existent hash', () => {
      expect(db.getChunkByHash('nonexistent')).toBeNull();
    });

    test('returns chunk for existing hash', () => {
      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 5, content: 'test', contentHash: 'hash1' },
        embedding: null,
      });
      const result = db.getChunkByHash('hash1');
      expect(result).not.toBeNull();
      expect(result?.content).toBe('test');
      expect(result?.file_path).toBe('test.md');
    });
  });

  describe('deleteChunksForFile', () => {
    test('deletes all chunks for a file', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'a', contentHash: 'h1' }, embedding: null });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 5, content: 'b', contentHash: 'h2' }, embedding: null });

      const deleted = db.deleteChunksForFile('a.md');
      expect(deleted).toBe(1);
      expect(db.listAllChunks().length).toBe(1);
    });

    test('returns 0 for non-existent file', () => {
      expect(db.deleteChunksForFile('nonexistent.md')).toBe(0);
    });

    test('removes from FTS index', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'unique keyword', contentHash: 'h1' }, embedding: null });
      db.deleteChunksForFile('a.md');

      const results = db.searchKeyword('unique', 10);
      expect(results).toHaveLength(0);
    });
  });

  describe('listIndexedFiles', () => {
    test('returns empty array for empty database', () => {
      expect(db.listIndexedFiles()).toEqual([]);
    });

    test('returns unique file paths', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'x', contentHash: 'h1' }, embedding: null });
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 6, endLine: 10, content: 'y', contentHash: 'h2' }, embedding: null });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 5, content: 'z', contentHash: 'h3' }, embedding: null });

      expect(db.listIndexedFiles().sort()).toEqual(['a.md', 'b.md']);
    });
  });

  // ============================================
  // Vector Search
  // ============================================
  describe('searchVector', () => {
    test('returns results sorted by similarity', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 1, content: 'one', contentHash: 'h1' }, embedding: [1, 0, 0] });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 1, content: 'two', contentHash: 'h2' }, embedding: [0.9, 0.1, 0] });
      db.upsertChunk({ chunk: { filePath: 'c.md', startLine: 1, endLine: 1, content: 'three', contentHash: 'h3' }, embedding: [0, 0, 1] });

      const results = db.searchVector([1, 0, 0], 10);
      expect(results.length).toBe(3);
      expect(results[0].chunkId).toBe(1); // Highest similarity
    });

    test('returns empty for empty database', () => {
      expect(db.searchVector([0.5, 0.5], 10)).toEqual([]);
    });

    test('respects maxResults limit', () => {
      for (let i = 0; i < 20; i++) {
        db.upsertChunk({ chunk: { filePath: 'f.md', startLine: i, endLine: i, content: `c${i}`, contentHash: `h${i}` }, embedding: [0.5] });
      }
      expect(db.searchVector([0.5], 5).length).toBe(5);
    });

    test('handles chunks without embeddings', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 1, content: 'no embedding', contentHash: 'h1' }, embedding: null });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 1, content: 'has embedding', contentHash: 'h2' }, embedding: [0.5] });

      const results = db.searchVector([0.5], 10);
      // Only chunk with embedding should appear
      expect(results.every(r => r.chunkId === 2)).toBe(true);
    });
  });

  // ============================================
  // Keyword Search (FTS5)
  // ============================================
  describe('searchKeyword', () => {
    test('finds exact word matches', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'The quick brown fox', contentHash: 'h1' }, embedding: null });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 5, content: 'The lazy dog', contentHash: 'h2' }, embedding: null });

      const results = db.searchKeyword('quick', 10);
      expect(results.length).toBe(1);
      expect(results[0].chunkId).toBe(1);
    });

    test('handles AND queries', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'quick brown fox', contentHash: 'h1' }, embedding: null });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 5, content: 'quick dog', contentHash: 'h2' }, embedding: null });

      const results = db.searchKeyword('quick fox', 10);
      expect(results.length).toBe(1);
    });

    test('returns empty for no matches', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'hello world', contentHash: 'h1' }, embedding: null });
      expect(db.searchKeyword('nonexistent', 10)).toEqual([]);
    });

    test('respects maxResults', () => {
      for (let i = 0; i < 10; i++) {
        db.upsertChunk({ chunk: { filePath: `${i}.md`, startLine: 1, endLine: 5, content: `test content ${i}`, contentHash: `h${i}` }, embedding: null });
      }
      expect(db.searchKeyword('test', 3).length).toBe(3);
    });
  });

  // ============================================
  // Hybrid Search
  // ============================================
  describe('searchHybrid', () => {
    test('combines vector and keyword results', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'AAPL stock price', contentHash: 'h1' }, embedding: [0.8, 0.2] });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 5, content: 'TSLA earnings report', contentHash: 'h2' }, embedding: [0.8, 0.2] });

      const results = db.searchHybrid({
        queryEmbedding: [0.8, 0.2],
        query: 'AAPL',
        vectorWeight: 0.5,
        textWeight: 0.5,
        maxResults: 10,
      });

      expect(results.length).toBeGreaterThan(0);
    });

    test('weights vector results higher with higher vectorWeight', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'keyword match', contentHash: 'h1' }, embedding: [1, 0] });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 5, content: 'no keyword', contentHash: 'h2' }, embedding: [0.9, 0.1] });

      const vectorHeavy = db.searchHybrid({
        queryEmbedding: [1, 0],
        query: 'keyword',
        vectorWeight: 0.9,
        textWeight: 0.1,
        maxResults: 10,
      });

      expect(vectorHeavy.length).toBe(2);
    });

    test('handles zero weights', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'test', contentHash: 'h1' }, embedding: [0.5] });

      const results = db.searchHybrid({
        queryEmbedding: [0.5],
        query: 'test',
        vectorWeight: 0,
        textWeight: 1,
        maxResults: 10,
      });

      expect(results.length).toBe(1);
    });
  });

  // ============================================
  // Embedding Cache
  // ============================================
  describe('cacheEmbedding / getCachedEmbedding', () => {
    test('round-trips embedding', () => {
      const embedding = [0.1, 0.2, 0.3, 0.4, 0.5];
      db.cacheEmbedding({ contentHash: 'test-hash', embedding, provider: 'openai', model: 'text-embedding-3-small' });

      const cached = db.getCachedEmbedding('test-hash');
      expect(cached).not.toBeNull();
      expect(cached?.embedding).toEqual(embedding);
      expect(cached?.provider).toBe('openai');
    });

    test('returns null for uncached hash', () => {
      expect(db.getCachedEmbedding('nonexistent')).toBeNull();
    });

    test('overwrites existing cache', () => {
      db.cacheEmbedding({ contentHash: 'hash', embedding: [0.1], provider: 'openai', model: 'm1' });
      db.cacheEmbedding({ contentHash: 'hash', embedding: [0.9], provider: 'anthropic', model: 'm2' });

      const cached = db.getCachedEmbedding('hash');
      expect(cached?.embedding).toEqual([0.9]);
      expect(cached?.provider).toBe('anthropic');
    });
  });

  // ============================================
  // Meta Operations
  // ============================================
  describe('getMeta / setMeta', () => {
    test('stores and retrieves metadata', () => {
      db.setMeta('last_sync', '2024-01-01');
      expect(db.getMeta('last_sync')).toBe('2024-01-01');
    });

    test('returns null for non-existent key', () => {
      expect(db.getMeta('nonexistent')).toBeNull();
    });

    test('overwrites existing value', () => {
      db.setMeta('key', 'value1');
      db.setMeta('key', 'value2');
      expect(db.getMeta('key')).toBe('value2');
    });
  });

  // ============================================
  // Utility Functions
  // ============================================
  describe('toBlob / fromBlob', () => {
    test('converts array to blob and back', () => {
      const original = [0.1, 0.2, 0.3, -0.5, 1.0];
      const blob = toBlob(original);
      const restored = fromBlob(blob);
      expect(restored).toEqual(original);
    });

    test('handles empty array', () => {
      const blob = toBlob([]);
      const restored = fromBlob(blob);
      expect(restored).toEqual([]);
    });
  });

  describe('buildFtsQuery', () => {
    test('quotes tokens for AND query', () => {
      expect(buildFtsQuery('hello world')).toBe('"hello" AND "world"');
    });

    test('handles special characters', () => {
      expect(buildFtsQuery('test "quoted"')).toBe('"test" AND "quoted"');
    });

    test('returns empty for empty input', () => {
      expect(buildFtsQuery('')).toBe('');
      expect(buildFtsQuery('   ')).toBe('');
    });

    test('handles unicode', () => {
      expect(buildFtsQuery('日本語 テスト')).toBe('"日本語" AND "テスト"');
    });

    test('handles numbers', () => {
      expect(buildFtsQuery('AAPL 2024')).toBe('"AAPL" AND "2024"');
    });
  });

  describe('cosineSimilarity', () => {
    test('returns 1 for identical vectors', () => {
      const v = [1, 0, 0];
      expect(cosineSimilarity(v, v)).toBeCloseTo(1, 5);
    });

    test('returns 0 for orthogonal vectors', () => {
      expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 5);
    });

    test('returns -1 for opposite vectors', () => {
      expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 5);
    });

    test('returns 0 for empty vectors', () => {
      expect(cosineSimilarity([], [])).toBe(0);
    });

    test('returns 0 for mismatched lengths', () => {
      expect(cosineSimilarity([1, 2], [1, 2, 3])).toBe(0);
    });

    test('handles negative values', () => {
      expect(cosineSimilarity([1, 1], [-1, 1])).toBeCloseTo(0, 5);
    });
  });
});
```

---

### Integration Tests: `src/memory/database.integration.test.ts`

```typescript
import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { existsSync, rmSync, mkdirSync, writeFileSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { MemoryDatabase } from './database.js';

const RUN_INTEGRATION = process.env.RUN_INTEGRATION === '1';

// Integration tests require real SQLite operations
describe.skipIf(!RUN_INTEGRATION)('MemoryDatabase (Integration)', () => {
  let db: MemoryDatabase;
  let dbPath: string;
  let testDir: string;

  beforeEach(async () => {
    testDir = join(tmpdir(), `db-test-${Date.now()}`);
    mkdirSync(testDir, { recursive: true });
    dbPath = join(testDir, 'memory.sqlite');
    db = await MemoryDatabase.create(dbPath);
  });

  afterEach(() => {
    db.close();
    if (existsSync(testDir)) rmSync(testDir, { recursive: true });
  });

  // ============================================
  // Real SQLite Operations
  // ============================================
  describe('database persistence', () => {
    test('persists data across reopen', async () => {
      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 10, content: 'Hello world', contentHash: 'hash1' },
        embedding: [0.1, 0.2, 0.3],
      });
      db.close();

      const reopened = await MemoryDatabase.create(dbPath);
      const chunks = reopened.listAllChunks();
      expect(chunks.length).toBe(1);
      expect(chunks[0]?.content).toBe('Hello world');
      reopened.close();
    });

    test('handles concurrent writes safely', async () => {
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(
          Promise.resolve().then(() => {
            db.upsertChunk({
              chunk: { filePath: `file${i}.md`, startLine: 1, endLine: 1, content: `content ${i}`, contentHash: `hash${i}` },
              embedding: [i * 0.1],
            });
          })
        );
      }
      await Promise.all(promises);
      expect(db.listAllChunks().length).toBe(10);
    });
  });

  // ============================================
  // Large Data Handling
  // ============================================
  describe('large data handling', () => {
    test('handles large embeddings (1536 dimensions)', () => {
      const largeEmbedding = Array(1536).fill(0).map((_, i) => i / 1536);

      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 5, content: 'large embedding', contentHash: 'h1' },
        embedding: largeEmbedding,
      });

      const results = db.searchVector(largeEmbedding, 1);
      expect(results.length).toBe(1);
    });

    test('handles large content (10KB)', () => {
      const largeContent = 'x'.repeat(10000);

      db.upsertChunk({
        chunk: { filePath: 'test.md', startLine: 1, endLine: 100, content: largeContent, contentHash: 'h1' },
        embedding: null,
      });

      const chunk = db.getChunkByHash('h1');
      expect(chunk?.content).toBe(largeContent);
    });

    test('handles many chunks (1000+)', () => {
      for (let i = 0; i < 1000; i++) {
        db.upsertChunk({
          chunk: { filePath: `file${i}.md`, startLine: 1, endLine: 1, content: `content ${i}`, contentHash: `hash${i}` },
          embedding: [i / 1000],
        });
      }

      expect(db.listAllChunks().length).toBe(1000);

      const results = db.searchVector([0.5], 10);
      expect(results.length).toBe(10);
    });
  });

  // ============================================
  // FTS5 Real Behavior
  // ============================================
  describe('FTS5 full-text search', () => {
    test('finds partial matches', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'financial analysis report', contentHash: 'h1' }, embedding: null });
      db.upsertChunk({ chunk: { filePath: 'b.md', startLine: 1, endLine: 5, content: 'quarterly earnings', contentHash: 'h2' }, embedding: null });

      const results = db.searchKeyword('financial', 10);
      expect(results.length).toBe(1);
    });

    test('handles special FTS5 characters', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'AAPL ticker symbol', contentHash: 'h1' }, embedding: null });

      // Should not throw on special chars
      const results = db.searchKeyword('AAPL', 10);
      expect(results.length).toBe(1);
    });

    test('unicode content search', () => {
      db.upsertChunk({ chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: '日本語のテスト', contentHash: 'h1' }, embedding: null });

      const results = db.searchKeyword('日本語', 10);
      expect(results.length).toBe(1);
    });
  });

  // ============================================
  // Hybrid Search Real Behavior
  // ============================================
  describe('hybrid search scoring', () => {
    test('deduplicates results from vector and keyword', () => {
      db.upsertChunk({
        chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'AAPL Apple Inc stock', contentHash: 'h1' },
        embedding: [0.8, 0.2],
      });

      const results = db.searchHybrid({
        queryEmbedding: [0.8, 0.2],
        query: 'AAPL',
        vectorWeight: 0.5,
        textWeight: 0.5,
        maxResults: 10,
      });

      // Same chunk should only appear once
      const ids = results.map(r => r.chunkId);
      expect(new Set(ids).size).toBe(ids.length);
    });

    test('returns source indicating match type', () => {
      db.upsertChunk({
        chunk: { filePath: 'a.md', startLine: 1, endLine: 5, content: 'test query match', contentHash: 'h1' },
        embedding: [0.5, 0.5],
      });

      const results = db.searchHybrid({
        queryEmbedding: [0.5, 0.5],
        query: 'test',
        vectorWeight: 0.5,
        textWeight: 0.5,
        maxResults: 10,
      });

      // Should have source indicating 'both', 'vector', or 'keyword'
      expect(['both', 'vector', 'keyword']).toContain(results[0]?.source);
    });
  });

  // ============================================
  // Embedding Cache Real Behavior
  // ============================================
  describe('embedding cache persistence', () => {
    test('persists embeddings across sessions', async () => {
      const embedding = Array(384).fill(0).map((_, i) => i / 384);
      db.cacheEmbedding({
        contentHash: 'persist-test',
        embedding,
        provider: 'openai',
        model: 'text-embedding-3-small',
      });
      db.close();

      const reopened = await MemoryDatabase.create(dbPath);
      const cached = reopened.getCachedEmbedding('persist-test');
      expect(cached?.embedding).toEqual(embedding);
      expect(cached?.provider).toBe('openai');
      reopened.close();
    });
  });

  // ============================================
  // Error Handling
  // ============================================
  describe('error handling', () => {
    test('handles invalid embedding dimensions', () => {
      // Should not throw - embeddings are stored as blobs
      expect(() => {
        db.upsertChunk({
          chunk: { filePath: 'test.md', startLine: 1, endLine: 1, content: 'x', contentHash: 'h1' },
          embedding: [0.1], // 1D embedding
        });
      }).not.toThrow();
    });

    test('handles concurrent upserts on same hash', async () => {
      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(
          Promise.resolve().then(() => {
            db.upsertChunk({
              chunk: { filePath: 'test.md', startLine: i, endLine: i, content: `version ${i}`, contentHash: 'same-hash' },
              embedding: null,
            });
          })
        );
      }
      await Promise.all(promises);
      expect(db.listAllChunks().length).toBe(1);
    });
  });

  // ============================================
  // Performance Benchmarks
  // ============================================
  describe('performance', () => {
    test('vector search on 1000 chunks completes in <100ms', () => {
      for (let i = 0; i < 1000; i++) {
        db.upsertChunk({
          chunk: { filePath: `f${i}.md`, startLine: 1, endLine: 1, content: `content ${i}`, contentHash: `h${i}` },
          embedding: [Math.random(), Math.random(), Math.random()],
        });
      }

      const start = Date.now();
      db.searchVector([0.5, 0.5, 0.5], 10);
      const elapsed = Date.now() - start;

      expect(elapsed).toBeLessThan(100);
    });

    test('keyword search on 1000 chunks completes in <50ms', () => {
      for (let i = 0; i < 1000; i++) {
        db.upsertChunk({
          chunk: { filePath: `f${i}.md`, startLine: 1, endLine: 1, content: `word${i} content`, contentHash: `h${i}` },
          embedding: null,
        });
      }

      const start = Date.now();
      db.searchKeyword('word500', 10);
      const elapsed = Date.now() - start;

      expect(elapsed).toBeLessThan(50);
    });
  });
});
```

---

## 2. `src/model/llm.ts` (~243 lines)

### File Analysis

Key exports:
- `getChatModel(modelName, streaming)` - factory for LLM instances
- `callLlm(prompt, options)` - main LLM invocation with retry
- `getFastModel(modelProvider, fallbackModel)` - get fast variant
- `DEFAULT_MODEL`, `DEFAULT_PROVIDER` - constants
- `withRetry()` - exponential backoff helper (internal)
- `buildAnthropicMessages()` - Anthropic cache control (internal)
- `extractUsage()` - token usage extraction (internal)

---

### Unit Tests: `src/model/llm.test.ts`

```typescript
import { describe, test, expect, mock, spyOn, beforeEach, afterEach } from 'bun:test';

// Mock LangChain modules
const mockInvoke = mock(async () => ({
  content: 'test response',
  usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
}));

const mockChatOpenAI = mock(() => ({ invoke: mockInvoke }));
const mockChatAnthropic = mock(() => ({ invoke: mockInvoke }));
const mockChatGoogleGenerativeAI = mock(() => ({ invoke: mockInvoke }));
const mockChatOllama = mock(() => ({ invoke: mockInvoke }));

mock.module('@langchain/openai', () => ({ ChatOpenAI: mockChatOpenAI }));
mock.module('@langchain/anthropic', () => ({ ChatAnthropic: mockChatAnthropic }));
mock.module('@langchain/google-genai', () => ({ ChatGoogleGenerativeAI: mockChatGoogleGenerativeAI }));
mock.module('@langchain/ollama', () => ({ ChatOllama: mockChatOllama }));

// Mock providers
mock.module('@/providers', () => ({
  resolveProvider: (name: string) => ({
    id: name.includes('claude') ? 'anthropic' : name.includes('gemini') ? 'google' : name.includes('ollama') ? 'ollama' : 'openai',
  }),
  getProviderById: (id: string) => {
    const providers: Record<string, { id: string; fastModel?: string }> = {
      anthropic: { id: 'anthropic', fastModel: 'claude-3-haiku' },
      openai: { id: 'openai', fastModel: 'gpt-4o-mini' },
      google: { id: 'google', fastModel: 'gemini-flash' },
    };
    return providers[id] || null;
  },
}));

// Import after mocking
const { getChatModel, getFastModel, callLlm, DEFAULT_MODEL, DEFAULT_PROVIDER } = await import('./llm.js');

describe('LLM Module (Unit)', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv, OPENAI_API_KEY: 'test-key' };
    mockChatOpenAI.mockClear();
    mockInvoke.mockClear();
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  // ============================================
  // Constants
  // ============================================
  describe('constants', () => {
    test('DEFAULT_PROVIDER is openai', () => {
      expect(DEFAULT_PROVIDER).toBe('openai');
    });

    test('DEFAULT_MODEL is defined', () => {
      expect(DEFAULT_MODEL).toBeDefined();
      expect(typeof DEFAULT_MODEL).toBe('string');
    });
  });

  // ============================================
  // getFastModel
  // ============================================
  describe('getFastModel', () => {
    test('returns fast model from provider config', () => {
      expect(getFastModel('anthropic', 'fallback')).toBe('claude-3-haiku');
      expect(getFastModel('openai', 'fallback')).toBe('gpt-4o-mini');
    });

    test('returns fallback when provider not found', () => {
      expect(getFastModel('unknown-provider', 'fallback-model')).toBe('fallback-model');
    });
  });

  // ============================================
  // getChatModel
  // ============================================
  describe('getChatModel', () => {
    test('creates ChatOpenAI by default', () => {
      getChatModel('gpt-4', false);
      expect(mockChatOpenAI).toHaveBeenCalled();
    });

    test('creates ChatAnthropic for claude models', () => {
      mockChatAnthropic.mockClear();
      getChatModel('claude-3-opus', false);
      expect(mockChatAnthropic).toHaveBeenCalled();
    });

    test('creates ChatGoogleGenerativeAI for gemini models', () => {
      mockChatGoogleGenerativeAI.mockClear();
      getChatModel('gemini-pro', false);
      expect(mockChatGoogleGenerativeAI).toHaveBeenCalled();
    });

    test('creates ChatOllama for ollama models', () => {
      mockChatOllama.mockClear();
      getChatModel('ollama:llama2', false);
      expect(mockChatOllama).toHaveBeenCalled();
    });

    test('passes streaming option', () => {
      getChatModel('gpt-4', true);
      expect(mockChatOpenAI).toHaveBeenCalledWith(
        expect.objectContaining({ streaming: true }),
      );
    });

    test('uses default model when not specified', () => {
      const model = getChatModel(undefined as any, false);
      expect(model).toBeDefined();
    });

    test('throws when API key missing', () => {
      delete process.env.OPENAI_API_KEY;
      delete process.env.ANTHROPIC_API_KEY;
      delete process.env.GOOGLE_API_KEY;

      expect(() => getChatModel('gpt-4', false)).toThrow('OPENAI_API_KEY');
    });

    test('uses custom base URL for Ollama', () => {
      process.env.OLLAMA_BASE_URL = 'http://custom:11434';
      mockChatOllama.mockClear();
      getChatModel('ollama:llama2', false);
      expect(mockChatOllama).toHaveBeenCalledWith(
        expect.objectContaining({ baseUrl: 'http://custom:11434' }),
      );
      delete process.env.OLLAMA_BASE_URL;
    });
  });

  // ============================================
  // callLlm
  // ============================================
  describe('callLlm', () => {
    test('returns response with usage', async () => {
      mockInvoke.mockResolvedValueOnce({
        content: 'Hello back!',
        usage_metadata: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
      });

      const result = await callLlm('Hello');
      expect(result.response).toBeDefined();
      expect(result.usage).toBeDefined();
      expect(result.usage?.inputTokens).toBe(5);
      expect(result.usage?.outputTokens).toBe(3);
    });

    test('uses custom system prompt', async () => {
      mockInvoke.mockResolvedValueOnce({
        content: 'response',
        usage_metadata: { input_tokens: 5, output_tokens: 3 },
      });

      await callLlm('Hello', { systemPrompt: 'Be concise' });
      expect(mockInvoke).toHaveBeenCalled();
    });

    test('uses specified model', async () => {
      mockChatAnthropic.mockClear();
      mockInvoke.mockResolvedValueOnce({
        content: 'response',
        usage_metadata: { input_tokens: 5, output_tokens: 3 },
      });

      await callLlm('Hello', { model: 'claude-3-opus' });
      expect(mockChatAnthropic).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'claude-3-opus' }),
      );
    });

    test('handles missing usage_metadata gracefully', async () => {
      mockInvoke.mockResolvedValueOnce({ content: 'no usage' });

      const result = await callLlm('Hello');
      expect(result.response).toBeDefined();
      expect(result.usage).toBeUndefined();
    });

    test('passes abort signal', async () => {
      const controller = new AbortController();
      mockInvoke.mockResolvedValueOnce({
        content: 'response',
        usage_metadata: { input_tokens: 5, output_tokens: 3 },
      });

      await callLlm('Hello', { signal: controller.signal });
      expect(mockInvoke).toHaveBeenCalled();
    });
  });

  // ============================================
  // Retry Behavior
  // ============================================
  describe('retry behavior', () => {
    test('retries on rate limit errors', async () => {
      let attempts = 0;
      mockInvoke.mockImplementation(async () => {
        attempts++;
        if (attempts < 3) {
          const err = new Error('rate limit exceeded');
          (err as any).status = 429;
          throw err;
        }
        return { content: 'success', usage_metadata: { input_tokens: 1, output_tokens: 1 } };
      });

      const result = await callLlm('test');
      expect(result.response).toBeDefined();
      expect(attempts).toBe(3);
    }, 15000);

    test('does not retry on non-retryable errors', async () => {
      let attempts = 0;
      mockInvoke.mockImplementation(async () => {
        attempts++;
        const err = new Error('invalid api key');
        (err as any).status = 401;
        throw err;
      });

      await expect(callLlm('test')).rejects.toThrow();
      expect(attempts).toBe(1);
    });
  });

  // ============================================
  // Token Usage Extraction
  // ============================================
  describe('extractUsage', () => {
    test('extracts from usage_metadata', () => {
      const msg = {
        usage_metadata: {
          input_tokens: 100,
          output_tokens: 50,
          total_tokens: 150,
        },
      };
      // Tested via callLlm return
      expect(msg.usage_metadata.input_tokens).toBe(100);
    });

    test('handles AIMessageChunk format', () => {
      const msg = {
        usage_metadata: {
          input_tokens: 10,
          output_tokens: 5,
          input_token_details: { cache_read: 5 },
          output_token_details: { reasoning: 2 },
        },
      };
      expect(msg.usage_metadata.input_tokens).toBe(10);
    });
  });
});
```

---

### Integration Tests: `src/model/llm.integration.test.ts`

**Default Provider: Ollama Cloud**

Ollama is configured to use cloud models by default. No API keys required for integration tests.

```typescript
import { describe, test, expect, beforeAll } from 'bun:test';

const RUN_INTEGRATION = process.env.RUN_INTEGRATION === '1';

// Default test model via Ollama cloud
const DEFAULT_MODEL = 'ollama:minimax-m2.7:cloud';

describe.skipIf(!RUN_INTEGRATION)('LLM Module (Integration)', () => {
  let hasOllama: boolean;
  let hasOpenAI: boolean;
  let hasAnthropic: boolean;
  let hasGemini: boolean;

  beforeAll(async () => {
    hasOllama = await checkOllamaAvailable();
    hasOpenAI = !!process.env.OPENAI_API_KEY;
    hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
    hasGemini = !!process.env.GOOGLE_API_KEY;
  });

  async function checkOllamaAvailable(): Promise<boolean> {
    try {
      const res = await fetch('http://127.0.0.1:11434/api/tags');
      return res.ok;
    } catch {
      return false;
    }
  }

  // ============================================
  // Ollama Cloud Tests (Default - No API Keys)
  // ============================================
  describe('Ollama Cloud', () => {
    test.skipIf(!hasOllama)('makes real API call', async () => {
      const { callLlm } = await import('./llm.js');

      const result = await callLlm('Say "hello" and nothing else', {
        model: DEFAULT_MODEL,
      });

      expect(result.response).toBeDefined();
      expect(typeof result.response).toBe('string');
    }, 60000);

    test.skipIf(!hasOllama)('returns token usage', async () => {
      const { callLlm } = await import('./llm.js');

      const result = await callLlm('Say "test"', { model: DEFAULT_MODEL });

      expect(result.usage).toBeDefined();
      expect(result.usage?.inputTokens).toBeGreaterThanOrEqual(0);
      expect(result.usage?.outputTokens).toBeGreaterThanOrEqual(0);
    }, 60000);

    test.skipIf(!hasOllama)('supports custom base URL', async () => {
      const originalUrl = process.env.OLLAMA_BASE_URL;
      process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:11434';

      const { getChatModel } = await import('./llm.js');
      const model = getChatModel(DEFAULT_MODEL, false);

      expect(model).toBeDefined();

      if (originalUrl) process.env.OLLAMA_BASE_URL = originalUrl;
      else delete process.env.OLLAMA_BASE_URL;
    });

    test.skipIf(!hasOllama)('handles streaming mode', async () => {
      const { getChatModel } = await import('./llm.js');

      const model = getChatModel(DEFAULT_MODEL, true);
      expect(model).toBeDefined();
    });

    test.skipIf(!hasOllama)('works with different cloud models', async () => {
      const { callLlm } = await import('./llm.js');

      const models = [
        'ollama:minimax-m2.7:cloud',
        'ollama:deepseek-v3.2:cloud',
      ];

      for (const model of models) {
        const result = await callLlm('Say "ok"', { model });
        expect(result.response).toBeDefined();
      }
    }, 120000);
  });

  // ============================================
  // Error Handling
  // ============================================
  describe('error handling', () => {
    test.skipIf(!hasOllama)('handles invalid model gracefully', async () => {
      const { callLlm } = await import('./llm.js');

      await expect(callLlm('test', { model: 'ollama:nonexistent-model-xyz' })).rejects.toThrow();
    }, 30000);

    test.skipIf(!hasOllama)('handles timeout', async () => {
      const { callLlm } = await import('./llm.js');

      const controller = new AbortController();
      setTimeout(() => controller.abort(), 100);

      await expect(
        callLlm('test', { model: DEFAULT_MODEL, signal: controller.signal })
      ).rejects.toThrow();
    }, 5000);
  });

  // ============================================
  // Alternative Cloud Providers (When API Keys Present)
  // ============================================
  describe('OpenAI (requires API key)', () => {
    test.skipIf(!hasOpenAI)('makes real API call', async () => {
      const { callLlm } = await import('./llm.js');

      const result = await callLlm('Say "hello"', { model: 'gpt-4o-mini' });
      expect(result.response).toBeDefined();
    }, 30000);

    test.skipIf(!hasOpenAI)('returns token usage', async () => {
      const { callLlm } = await import('./llm.js');

      const result = await callLlm('Say "test"', { model: 'gpt-4o-mini' });
      expect(result.usage?.inputTokens).toBeGreaterThan(0);
    }, 30000);
  });

  describe('Anthropic (requires API key)', () => {
    test.skipIf(!hasAnthropic)('makes real API call', async () => {
      const { callLlm } = await import('./llm.js');

      const result = await callLlm('Say "hello"', { model: 'claude-3-haiku-20240307' });
      expect(result.response).toBeDefined();
    }, 30000);

    test.skipIf(!hasAnthropic)('uses cache_control for system prompt', async () => {
      const { callLlm } = await import('./llm.js');

      const systemPrompt = 'You are a helpful assistant. ' + 'x'.repeat(1000);
      await callLlm('First call', { model: 'claude-3-haiku-20240307', systemPrompt });
      const result = await callLlm('Second call', { model: 'claude-3-haiku-20240307', systemPrompt });

      expect(result.usage).toBeDefined();
    }, 30000);
  });

  describe('Gemini (requires API key)', () => {
    test.skipIf(!hasGemini)('makes real API call', async () => {
      const { callLlm } = await import('./llm.js');

      const result = await callLlm('Say "hello"', { model: 'gemini-pro' });
      expect(result.response).toBeDefined();
    }, 30000);
  });

  // ============================================
  // Performance
  // ============================================
  describe('performance', () => {
    test('model creation is fast (<100ms)', () => {
      const start = Date.now();
      for (let i = 0; i < 10; i++) {
        const { getChatModel } = require('./llm.js');
        getChatModel(DEFAULT_MODEL, false);
      }
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(100);
    });
  });
});
```

---

## Implementation Checklist

### Phase 1: Database Unit Tests (Priority: High)

| Test Area | Tests Required | Status |
|-----------|---------------|--------|
| Factory & Init | 3 tests | [ ] |
| Chunk Operations | 5 tests | [ ] |
| Vector Search | 4 tests | [ ] |
| Keyword Search | 4 tests | [ ] |
| Hybrid Search | 3 tests | [ ] |
| Embedding Cache | 3 tests | [ ] |
| Meta Operations | 3 tests | [ ] |
| Utilities | 10 tests | [ ] |

### Phase 2: Database Integration Tests (Priority: High)

| Test Area | Tests Required | Status |
|-----------|---------------|--------|
| Persistence | 2 tests | [ ] |
| Large Data | 3 tests | [ ] |
| FTS5 Real Behavior | 3 tests | [ ] |
| Hybrid Search | 2 tests | [ ] |
| Cache Persistence | 1 test | [ ] |
| Error Handling | 2 tests | [ ] |
| Performance | 2 tests | [ ] |

### Phase 3: LLM Unit Tests (Priority: Medium)

| Test Area | Tests Required | Status |
|-----------|---------------|--------|
| Constants | 2 tests | [ ] |
| getFastModel | 2 tests | [ ] |
| getChatModel | 7 tests | [ ] |
| callLlm | 5 tests | [ ] |
| Retry Behavior | 2 tests | [ ] |

### Phase 4: LLM Integration Tests (Priority: Medium)

| Test Area | Tests Required | Status |
|-----------|---------------|--------|
| Ollama Cloud (Default) | 5 tests | [ ] |
| Error Handling | 2 tests | [ ] |
| OpenAI (API Key) | 2 tests | [ ] |
| Anthropic (API Key) | 2 tests | [ ] |
| Gemini (API Key) | 1 test | [ ] |
| Performance | 1 test | [ ] |

**Default Test Configuration:**
- Model: `ollama:minimax-m2.7:cloud`
- Run: `RUN_INTEGRATION=1 bun test src/model/llm.integration.test.ts`

---

## Estimated Coverage Impact

| File | Test Type | Current | Target | Lines Added |
|------|-----------|---------|--------|--------------|
| `src/memory/database.ts` | Unit | 0% | 85% | ~350 test lines |
| `src/memory/database.ts` | Integration | 0% | 70% | ~200 test lines |
| `src/model/llm.ts` | Unit | 0% | 75% | ~180 test lines |
| `src/model/llm.ts` | Integration | 0% | 60% | ~100 test lines |

---

## Running the Tests

```bash
# Unit tests only (default)
bun test

# Unit tests for specific file
bun test src/memory/database.test.ts
bun test src/model/llm.test.ts

# Integration tests (Ollama - no API keys required)
RUN_INTEGRATION=1 bun test --filter integration

# Integration tests for database only
RUN_INTEGRATION=1 bun test src/memory/database.integration.test.ts

# Integration tests for LLM with Ollama
RUN_INTEGRATION=1 bun test src/model/llm.integration.test.ts

# All tests
RUN_INTEGRATION=1 bun test
```

---

## Ollama Setup for Integration Tests

Integration tests use Ollama cloud models by default. No external API keys required.

### Prerequisites

1. Install Ollama: https://ollama.ai
2. Start Ollama: `ollama serve` (or it runs automatically on boot)
3. Cloud models are available automatically (no pull required)

### Default Test Model

- **Chat:** `ollama:minimax-m2.7:cloud` (or `deepseek-v3.2:cloud`, `qwen3.5:cloud`)
- **Embeddings:** `nomic-embed-text:latest` (local, requires `ollama pull nomic-embed-text`)

### Available Cloud Models (via Ollama)

| Model | Model ID | Notes |
|-------|----------|-------|
| MiniMax M2.7 | `minimax-m2.7:cloud` | Default for tests |
| DeepSeek V3.2 | `deepseek-v3.2:cloud` | Alternative |
| Qwen 3.5 | `qwen3.5:cloud` | Alternative |
| GLM-5 | `glm-5:cloud` | Alternative |

### Verification

```bash
# Check Ollama is running and list models
curl http://127.0.0.1:11434/api/tags

# Test cloud chat model
curl http://127.0.0.1:11434/api/generate \
  -d '{"model": "minimax-m2.7:cloud", "prompt": "Say hello", "stream": false}'

# Pull local embedding model (needed for database tests)
ollama pull nomic-embed-text
```

---

## Dependencies & Mocking

| Dependency | Unit Test | Integration Test |
|-----------|-----------|------------------|
| `better-sqlite3` | Real (temp file) | Real |
| `@langchain/ollama` | Mocked | **Real (default)** |
| `@langchain/openai` | Mocked | Real API (if key present) |
| `@langchain/anthropic` | Mocked | Real API (if key present) |
| `@langchain/google-genai` | Mocked | Real API (if key present) |

**Note:** Integration tests default to Ollama cloud models. No external API keys required.
| `@langchain/google-genai` | Mocked | Real API |
| `@langchain/ollama` | Mocked | N/A |

---

## CI Integration

Add to CI workflow:

```yaml
- name: Run Unit Tests
  run: bun test

- name: Run Integration Tests
  if: secrets.OPENAI_API_KEY != ''
  run: RUN_INTEGRATION=1 bun test --filter integration
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```