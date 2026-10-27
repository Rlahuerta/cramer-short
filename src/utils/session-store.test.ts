/**
 * TDD tests for session-store.ts — session persistence file I/O.
 * All tests use an isolated tmp directory; no side effects on .cramer-short/.
 */

import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { describe, it, expect, beforeEach, afterEach, setSystemTime } from 'bun:test';
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { HistoryItem } from '../shared/history-types.js';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});
import {
  generateSessionName,
  createSession,
  saveSession,
  listSessions,
  loadSession,
  type SessionFile,
} from './session-store.js';

// ─── helpers ────────────────────────────────────────────────────────────────

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await mkdtemp(join(tmpdir(), 'dexter-sessions-test-'));
});

afterEach(async () => {
  await rm(tmpDir, { recursive: true, force: true });
});

function makeItem(query: string, answer: string): HistoryItem {
  return { id: String(FIXED_TEST_NOW_MS), query, events: [], answer, status: 'complete' };
}

// Fixed UTC timestamps for deterministic name assertions.
const TS_2026_03_26 = new Date('2026-03-26T12:00:00.000Z').getTime();

// ─── generateSessionName ────────────────────────────────────────────────────

describe('generateSessionName', () => {
  it('prefixes with UTC date derived from timestamp', () => {
    const name = generateSessionName('hello world', TS_2026_03_26);
    expect(name.startsWith('2026-03-26 ')).toBe(true);
  });

  it('appends first 6 words of the query', () => {
    const name = generateSessionName('perform a deep analysis of Chevron stock today', TS_2026_03_26);
    // "perform a deep analysis of Chevron" — 6 words, "stock today" omitted
    expect(name).toBe('2026-03-26 perform a deep analysis of Chevron');
  });

  it('works fine with fewer than 6 words', () => {
    const name = generateSessionName('hello', TS_2026_03_26);
    expect(name).toBe('2026-03-26 hello');
  });

  it('truncates names longer than 60 chars with ellipsis', () => {
    // Use long individual words so the first 6 exceed 60 chars when combined with the date.
    const long = Array.from({ length: 6 }, () => 'averylongtoken').join(' ');
    const name = generateSessionName(long, TS_2026_03_26);
    expect(name.length).toBeLessThanOrEqual(60);
    expect(name.endsWith('…')).toBe(true);
  });

  it('does not add ellipsis when name fits within 60 chars', () => {
    const name = generateSessionName('short query', TS_2026_03_26);
    expect(name.endsWith('…')).toBe(false);
    expect(name.length).toBeLessThanOrEqual(60);
  });
});

// ─── createSession ───────────────────────────────────────────────────────────

describe('createSession', () => {
  it('returns a SessionFile with correct initial shape', async () => {
    const session = await createSession('analyze Chevron stock', tmpDir);
    expect(session.version).toBe(1);
    expect(typeof session.id).toBe('string');
    expect(session.id.length).toBeGreaterThan(0);
    expect(session.name).toContain('analyze Chevron');
    expect(session.queryCount).toBe(0);
    expect(session.llmMessages).toEqual([]);
    expect(session.history).toEqual([]);
    expect(typeof session.created).toBe('number');
    expect(typeof session.lastModified).toBe('number');
  });

  it('persists the session file to disk', async () => {
    const session = await createSession('my first query', tmpDir);
    const loaded = await loadSession(session.id, tmpDir);
    expect(loaded).not.toBeNull();
    expect(loaded!.id).toBe(session.id);
  });

  it('adds the new session to the index', async () => {
    await createSession('first query', tmpDir);
    const sessions = await listSessions(tmpDir);
    expect(sessions).toHaveLength(1);
    expect(sessions[0].name).toContain('first query');
  });

  it('stores firstQuery verbatim in the index entry', async () => {
    await createSession('What is the current P/E ratio of Apple?', tmpDir);
    const sessions = await listSessions(tmpDir);
    expect(sessions[0].firstQuery).toBe('What is the current P/E ratio of Apple?');
  });

  it('preserves firstQuery after subsequent saveSession calls', async () => {
    const session = await createSession('initial question about Tesla valuation', tmpDir);
    const updated: SessionFile = {
      ...session,
      queryCount: 3,
      lastModified: FIXED_TEST_NOW_MS,
      llmMessages: [],
      history: [],
    };
    await saveSession(updated, tmpDir);
    const sessions = await listSessions(tmpDir);
    expect(sessions[0].firstQuery).toBe('initial question about Tesla valuation');
    expect(sessions[0].queryCount).toBe(3);
  });

  it('generates unique IDs for concurrent calls', async () => {
    // Even if called in very quick succession the IDs must differ.
    const [a, b] = await Promise.all([
      createSession('query A', tmpDir),
      createSession('query B', tmpDir),
    ]);
    expect(a.id).not.toBe(b.id);
  });

  it('keeps every index entry when sessions are created concurrently', async () => {
    const created = await Promise.all(
      Array.from({ length: 5 }, (_, i) => createSession(`query ${i}`, tmpDir)),
    );
    const listed = await listSessions(tmpDir);
    expect(listed.map((s) => s.id).sort()).toEqual(created.map((s) => s.id).sort());
  });

  it('keeps an index entry created concurrently with a saveSession of another session', async () => {
    const first = await createSession('first', tmpDir);
    const [second] = await Promise.all([
      createSession('second', tmpDir),
      saveSession({ ...first, queryCount: 2 }, tmpDir),
    ]);
    const listed = await listSessions(tmpDir);
    expect(listed.map((s) => s.id).sort()).toEqual([first.id, second.id].sort());
    expect(listed.find((s) => s.id === first.id)?.queryCount).toBe(2);
  });

  describe('with an unparseable index', () => {
    async function writeCorruptIndex(): Promise<string> {
      const dir = join(tmpDir, '.cramer-short', 'sessions');
      await mkdir(dir, { recursive: true });
      await writeFile(join(dir, '_index.json'), '{"sessions":[{"id":"old-1","name":"OLD-SESSION-MARKER"}]} BROKEN');
      return dir;
    }

    it('moves it to a .corrupt-* backup instead of overwriting it', async () => {
      const dir = await writeCorruptIndex();
      await createSession('new query', tmpDir);

      const backups = (await readdir(dir)).filter((f) => f.startsWith('_index.json.corrupt-'));
      expect(backups).toHaveLength(1);
      expect(await readFile(join(dir, backups[0]), 'utf-8')).toContain('OLD-SESSION-MARKER');
      expect(await listSessions(tmpDir)).toHaveLength(1);
    });

    it('listSessions returns an empty list and still keeps the old content', async () => {
      const dir = await writeCorruptIndex();
      expect(await listSessions(tmpDir)).toEqual([]);

      const backups = (await readdir(dir)).filter((f) => f.startsWith('_index.json.corrupt-'));
      expect(await readFile(join(dir, backups[0]), 'utf-8')).toContain('OLD-SESSION-MARKER');
    });
  });
});

// ─── saveSession ─────────────────────────────────────────────────────────────

describe('saveSession', () => {
  it('persists llmMessages and history to disk', async () => {
    const session = await createSession('analyze Tesla', tmpDir);
    const item = makeItem('analyze Tesla', 'Tesla is an EV manufacturer');
    const updated: SessionFile = {
      ...session,
      queryCount: 1,
      lastModified: FIXED_TEST_NOW_MS,
      llmMessages: [{ query: 'analyze Tesla', answer: 'Tesla is an EV manufacturer', summary: 'EV analysis' }],
      history: [item],
    };
    await saveSession(updated, tmpDir);

    const loaded = await loadSession(session.id, tmpDir);
    expect(loaded).not.toBeNull();
    expect(loaded!.llmMessages).toHaveLength(1);
    expect(loaded!.llmMessages[0].query).toBe('analyze Tesla');
    expect(loaded!.history).toHaveLength(1);
    expect(loaded!.queryCount).toBe(1);
  });

  it('updates the index queryCount and lastModified', async () => {
    const session = await createSession('analyze Tesla', tmpDir);
    const updated: SessionFile = {
      ...session,
      queryCount: 7,
      lastModified: session.created + 60_000,
      llmMessages: [],
      history: [],
    };
    await saveSession(updated, tmpDir);

    const sessions = await listSessions(tmpDir);
    expect(sessions[0].queryCount).toBe(7);
    expect(sessions[0].lastModified).toBe(session.created + 60_000);
  });

  it('persists priorSummary when provided', async () => {
    const session = await createSession('long session', tmpDir);
    const updated: SessionFile = {
      ...session,
      queryCount: 15,
      lastModified: FIXED_TEST_NOW_MS,
      priorSummary: 'Key insight: Chevron is undervalued.',
      llmMessages: [],
      history: [],
    };
    await saveSession(updated, tmpDir);

    const loaded = await loadSession(session.id, tmpDir);
    expect(loaded!.priorSummary).toBe('Key insight: Chevron is undervalued.');
  });

  it('is atomic — full file is readable after save', async () => {
    const session = await createSession('resilience test', tmpDir);
    // Simulate multiple rapid saves
    const saves = Array.from({ length: 5 }, (_, i) =>
      saveSession({ ...session, queryCount: i + 1, lastModified: FIXED_TEST_NOW_MS, llmMessages: [], history: [] }, tmpDir),
    );
    await Promise.all(saves);
    const loaded = await loadSession(session.id, tmpDir);
    expect(loaded).not.toBeNull();
    // queryCount should be one of 1-5 — file is not corrupted
    expect(loaded!.queryCount).toBeGreaterThanOrEqual(1);
    expect(loaded!.queryCount).toBeLessThanOrEqual(5);
  });
});

// ─── listSessions ────────────────────────────────────────────────────────────

describe('listSessions', () => {
  it('returns empty array when no sessions exist', async () => {
    expect(await listSessions(tmpDir)).toEqual([]);
  });

  it('returns sessions newest-first', async () => {
    await createSession('first', tmpDir);
    setSystemTime(new Date(FIXED_TEST_NOW_MS + 5));
    await createSession('second', tmpDir);

    const sessions = await listSessions(tmpDir);
    expect(sessions).toHaveLength(2);
    expect(sessions[0].name).toContain('second');
    expect(sessions[1].name).toContain('first');
  });

  it('returns only index metadata — no llmMessages or history', async () => {
    await createSession('metadata only', tmpDir);
    const sessions = await listSessions(tmpDir);
    expect((sessions[0] as any).llmMessages).toBeUndefined();
    expect((sessions[0] as any).history).toBeUndefined();
  });

  it('includes id, name, created, lastModified, queryCount, file', async () => {
    await createSession('structured entry', tmpDir);
    const [entry] = await listSessions(tmpDir);
    expect(entry.id).toBeTruthy();
    expect(entry.name).toBeTruthy();
    expect(entry.created).toBeGreaterThan(0);
    expect(entry.lastModified).toBeGreaterThan(0);
    expect(typeof entry.queryCount).toBe('number');
    expect(entry.file).toMatch(/\.json$/);
  });
});

// ─── loadSession ─────────────────────────────────────────────────────────────

describe('loadSession', () => {
  it('returns null for a non-existent ID', async () => {
    expect(await loadSession('ghost-id-999', tmpDir)).toBeNull();
  });

  it('returns the full SessionFile by ID', async () => {
    const session = await createSession('load me please', tmpDir);
    const loaded = await loadSession(session.id, tmpDir);
    expect(loaded).not.toBeNull();
    expect(loaded!.id).toBe(session.id);
    expect(loaded!.name).toContain('load me');
  });
});
