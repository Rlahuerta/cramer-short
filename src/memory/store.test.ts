import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { MemoryStore } from './store.js';

let baseDir: string;
let store: MemoryStore;

beforeEach(async () => {
  baseDir = mkdtempSync(join(tmpdir(), 'cramer-store-'));
  store = new MemoryStore(baseDir);
  await store.ensureDirectoryExists();
});

afterEach(() => rmSync(baseDir, { recursive: true, force: true }));

describe('MemoryStore writes', () => {
  it('parallel appends to the same file keep every write', async () => {
    const lines = Array.from({ length: 25 }, (_, i) => `- note ${i}\n`);
    await Promise.all(lines.map((line) => store.appendMemoryFile('MEMORY.md', line)));
    const content = await store.readMemoryFile('MEMORY.md');
    for (const line of lines) expect(content).toContain(line);
  });

  it('parallel append + edit on the same file keep both changes', async () => {
    await store.writeMemoryFile('MEMORY.md', '- old\n');
    await Promise.all([
      store.editInMemoryFile('MEMORY.md', '- old', '- new'),
      store.appendMemoryFile('MEMORY.md', '- appended\n'),
    ]);
    expect(await store.readMemoryFile('MEMORY.md')).toBe('- new\n- appended\n');
  });

  it('edit treats $ patterns in the new text literally', async () => {
    await store.writeMemoryFile('MEMORY.md', 'price: X\n');
    await store.editInMemoryFile('MEMORY.md', 'X', '$$100 $& $1');
    expect(await store.readMemoryFile('MEMORY.md')).toBe('price: $$100 $& $1\n');
  });
});

describe('MemoryStore.resolveMemoryPath', () => {
  it('accepts MEMORY.md, FINANCE.md and daily files', () => {
    for (const name of ['MEMORY.md', 'FINANCE.md', '2026-10-23.md']) {
      expect(store.resolveMemoryPath(name)).toBe(join(store.getMemoryDir(), name));
    }
    expect(store.resolveMemoryPath(join(store.getMemoryDir(), 'MEMORY.md'))).toBe(
      join(store.getMemoryDir(), 'MEMORY.md'),
    );
  });

  it('rejects anything else inside the memory dir', () => {
    for (const name of ['.dream-meta.json', 'index.sqlite', 'notes.txt', 'archive/2026-01-01.md', 'MEMORY.md.bak']) {
      expect(() => store.resolveMemoryPath(name)).toThrow();
    }
  });

  it('rejects paths outside the memory dir', () => {
    expect(() => store.resolveMemoryPath('../settings.json')).toThrow();
  });
});

describe('MemoryStore.listMemoryFiles', () => {
  it('includes FINANCE.md', async () => {
    for (const name of ['MEMORY.md', 'FINANCE.md', '2026-10-23.md', 'notes.txt']) {
      writeFileSync(join(store.getMemoryDir(), name), 'x');
    }
    expect(await store.listMemoryFiles()).toEqual(['2026-10-23.md', 'FINANCE.md', 'MEMORY.md']);
  });
});
