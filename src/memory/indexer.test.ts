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

describe('MemoryIndexer.sync — edited files', () => {
  it('drops chunks whose content was edited out of a file that still exists', async () => {
    const store = new MemoryStore(baseDir);
    const indexer = new MemoryIndexer(store, db, {
      chunkTokens: 8,
      overlapTokens: 0,
      watchDebounceMs: 1000,
      embeddingClient: null,
      indexSessions: false,
    });

    await store.writeMemoryFile('MEMORY.md', 'Alpha paragraph about AAPL.\n\nBeta paragraph about MSFT.\n');
    await indexer.sync();
    expect(db.listAllChunks().some((c) => c.content.includes('MSFT'))).toBe(true);

    await store.writeMemoryFile('MEMORY.md', 'Alpha paragraph about AAPL.\n\nGamma paragraph about NVDA.\n');
    indexer.markDirty();
    await indexer.sync();

    const contents = db.listAllChunks().map((c) => c.content);
    expect(contents.some((c) => c.includes('MSFT'))).toBe(false);
    expect(contents.some((c) => c.includes('NVDA'))).toBe(true);
    expect(contents.some((c) => c.includes('AAPL'))).toBe(true);
  });
});
