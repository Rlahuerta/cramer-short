/**
 * PI-28: Ticker-filtered insight search applies the FTS limit before filtering
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `FinancialMemoryStore.search` asks FTS for `maxResults * 2` rows across all
 * tickers and only then applies the ticker filter. When another ticker owns
 * those top-ranked rows, the requested ticker's matching insight is never
 * fetched and the search returns an empty list.
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
    const { FinancialMemoryStore } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/memory/financial-store.ts')).href
    );
    mkdirSync(join(root, 'db'), { recursive: true });
    const db = await MemoryDatabase.create(join(root, 'db', 'insights.sqlite'));
    const store = new FinancialMemoryStore(db);

    for (let i = 0; i < 20; i += 1) {
      await store.storeInsight({
        ticker: 'BBB',
        content: `revenue growth revenue growth revenue growth ${i}`,
        tags: ['ticker:BBB'],
        source: 'test',
      });
    }
    await store.storeInsight({
      ticker: 'AAPL',
      content: 'revenue growth mentioned once',
      tags: ['ticker:AAPL'],
      source: 'test',
    });

    const unfiltered = store.search('revenue growth', { maxResults: 3 });
    const aaplFiltered = store.search('revenue growth', { ticker: 'AAPL', maxResults: 3 });
    const observed = JSON.stringify(
      {
        unfilteredTickers: unfiltered.map((insight) => insight.ticker),
        tickerFilteredCount: aaplFiltered.length,
        tickerFilteredTickers: aaplFiltered.map((insight) => insight.ticker),
      },
      null,
      2,
    );
    const expected = 'the ticker-filtered search returns the stored AAPL insight';
    const actual = `the ticker-filtered search returned ${aaplFiltered.length} results although AAPL has a match`;

    try {
      assert.ok(aaplFiltered.some((insight) => insight.ticker === 'AAPL'), 'ticker-filtered search dropped the only AAPL match');
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
