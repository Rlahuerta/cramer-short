/**
 * PI-27: Keyword-only search drops the weakest BM25 match
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `searchKeyword` min-max normalises BM25 ranks to [0, 1], so the weakest row in
 * every batch is assigned exactly 0. `hybridSearch` then filters with the
 * default `minScore` of 0.1 and discards it. With no embedding client
 * (keyword-only mode) a query with two distinct matches returns one result.
 */

import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));

class HarnessFailure extends Error {}

const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;
function normalize(text: string, root: string): string {
  return text
    .replace(ANSI, '')
    .split(root)
    .join('<TMP>')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

const EXTRA_SCRUB = [
  'OLLAMA_BASE_URL', 'DEXTER_GATEWAY_CONFIG', 'DEXTER_PAIRING_PATH',
  'CRAMER_SHORT_SESSIONS_DIR', 'KALSHI_API_KEY', 'OPENROUTER_API_KEY',
  'E2E_MODEL', 'FORECAST_LAB_OUTPUT_METRICS', 'COVERAGE_LINES_MIN', 'TEST_API_KEY',
];

async function prepare(): Promise<{ root: string; restore: () => void }> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) throw new HarnessFailure('CS_REPRO_ROOT is not set');
  const originalCwd = process.cwd();
  const saved = new Map<string, string | undefined>();
  const remember = (key: string) => { if (!saved.has(key)) saved.set(key, process.env[key]); };

  process.chdir(root);
  if (homedir() !== root) throw new HarnessFailure(`homedir escaped run root: ${homedir()}`);
  if (tmpdir() !== root) throw new HarnessFailure(`tmpdir escaped run root: ${tmpdir()}`);

  const envKeysPath = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys');
  const listed = (await readFile(envKeysPath, 'utf8'))
    .split(/\r?\n/)
    .map((v) => v.trim())
    .filter(Boolean);
  const exempt = new Set(['HOME', 'USERPROFILE']);
  for (const key of new Set([...listed, ...EXTRA_SCRUB])) {
    if (exempt.has(key)) continue;
    remember(key);
    delete process.env[key];
  }
  for (const key of Object.keys(process.env)) {
    if (exempt.has(key)) continue;
    if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/i.test(key)) { remember(key); delete process.env[key]; }
  }
  const surviving = Object.keys(process.env).filter(
    (key) => !exempt.has(key) && /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/i.test(key),
  );
  if (surviving.length > 0) throw new HarnessFailure(`surviving secret-like env keys: ${surviving.join(',')}`);

  globalThis.fetch = (async (input: unknown) => {
    throw new Error(`network disabled: ${String(input)}`);
  }) as unknown as typeof fetch;

  return {
    root,
    restore: () => {
      process.chdir(originalCwd);
      for (const [key, value] of saved) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    },
  };
}

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const { MemoryDatabase } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/memory/database.ts')).href
    );
    const { hybridSearch } = await import(pathToFileURL(join(REPO_ROOT, 'src/memory/search.ts')).href);
    mkdirSync(join(root, 'db'), { recursive: true });
    const db = await MemoryDatabase.create(join(root, 'db', 'search.sqlite'));
    db.upsertChunk({
      chunk: {
        filePath: 'MEMORY.md',
        startLine: 1,
        endLine: 1,
        content: 'revenue growth revenue growth revenue growth accelerated',
        contentHash: 'a',
      },
      embedding: null,
    });
    db.upsertChunk({
      chunk: {
        filePath: 'MEMORY.md',
        startLine: 2,
        endLine: 2,
        content:
          'revenue growth slowed modestly but the rest of this sentence is deliberately long and mentions many unrelated things like weather and transport and logistics',
        contentHash: 'b',
      },
      embedding: null,
    });

    const results = await hybridSearch({
      db,
      embeddingClient: null,
      query: 'revenue growth',
      options: { maxResults: 2, minScore: 0.1 },
      defaults: { maxResults: 6, minScore: 0.1, vectorWeight: 0.7, textWeight: 0.3 },
    });
    const observed = JSON.stringify(
      {
        keywordMatches: db.searchKeyword('revenue growth', 8).map((candidate) => candidate.score),
        returnedCount: results.length,
        returnedScores: results.map((entry) => entry.score),
      },
      null,
      2,
    );
    const expected = 'both keyword matches are returned when maxResults=2 and minScore=0.1';
    const actual = `only ${results.length} of 2 matching chunks were returned; the weakest BM25 hit scored 0 and was filtered`;

    try {
      assert.equal(results.length, 2, 'the weakest keyword match was dropped by min-max normalisation');
    } catch {
      console.log('OBSERVED:');
      for (const line of normalize(observed, root).split('\n')) console.log(`  ${line}`);
      console.log(`EXPECTED: ${expected}`);
      console.log(`ACTUAL: ${actual}`);
      process.exitCode = 42;
      return;
    }

    console.log('NO BUG');
  } catch (error) {
    if (error instanceof HarnessFailure) {
      console.log(`HARNESS-FAILURE: ${error.message}`);
      process.exitCode = 43;
      return;
    }
    throw error;
  } finally {
    prepared?.restore();
  }
}

await main();
