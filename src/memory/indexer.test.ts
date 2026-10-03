import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { MemoryDatabase } from './database.js';
import { MemoryIndexer } from './indexer.js';
import { MemoryStore } from './store.js';

let baseDir: string;
let memDir: string;
let db: MemoryDatabase;

beforeEach(async () => {
  baseDir = await mkdtemp(join(tmpdir(), 'dexter-indexer-'));
  memDir = join(baseDir, 'memory');
  await mkdir(memDir, { recursive: true });
  db = await MemoryDatabase.create(join(baseDir, 'index.sqlite'));
});

afterEach(async () => {
  db.close();
  await rm(baseDir, { recursive: true, force: true });
});

function createIndexer(store: MemoryStore): MemoryIndexer {
  return new MemoryIndexer(store, db, {
    chunkTokens: 400,
    overlapTokens: 80,
    watchDebounceMs: 5,
    embeddingClient: null,
    indexSessions: false,
  });
}

describe('MemoryIndexer.sync — changes during an in-flight sync', () => {
  it('indexes a change marked dirty mid-sync via the bounded second pass', async () => {
    await writeFile(join(memDir, 'MEMORY.md'), 'AAPL baseline note');

    let indexer!: MemoryIndexer;
    class MidSyncStore extends MemoryStore {
      private listCalls = 0;
      override async listMemoryFiles(): Promise<string[]> {
        this.listCalls += 1;
        const files = await super.listMemoryFiles();
        if (this.listCalls === 1) {
          await writeFile(join(memDir, '2026-01-01.md'), 'NVDA mid-sync discovery');
          indexer.markDirty();
        }
        return files;
      }
    }

    const store = new MidSyncStore(baseDir);
    indexer = createIndexer(store);

    await indexer.sync();

    expect(db.listIndexedFiles()).toContain('2026-01-01.md');
    expect(indexer.isDirty()).toBe(false);
  });
});
