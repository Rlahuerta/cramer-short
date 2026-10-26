/**
 * PI-47: geopolitics_search classifies "Strait of Hormuz" news as China-Taiwan
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `detectCategories` matches substrings, and the China-Taiwan entry carries the
 * bare keyword `'strait'` (intended as "Taiwan Strait"). Any Middle East story
 * about the Strait of Hormuz therefore also matches China-Taiwan, and
 * `getAssetImplications` then emits TSMC/NVIDIA/AMD/ASML/Intel as risk-down
 * implications for an oil-supply question — a wrong answer, not a missing one.
 * The same table maps `energy-supply` airline exposure to ticker `AIRL` while the
 * asset name says "Airline ETF (JETS)".
 */

import assert from 'node:assert/strict';
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
  'OLLAMA_BASE_URL',
  'DEXTER_GATEWAY_CONFIG',
  'DEXTER_PAIRING_PATH',
  'CRAMER_SHORT_SESSIONS_DIR',
  'KALSHI_API_KEY',
  'OPENROUTER_API_KEY',
  'E2E_MODEL',
  'FORECAST_LAB_OUTPUT_METRICS',
  'COVERAGE_LINES_MIN',
  'TEST_API_KEY',
];

async function prepare(): Promise<{ root: string; restore: () => void }> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) throw new HarnessFailure('CS_REPRO_ROOT is not set');

  const originalCwd = process.cwd();
  const saved = new Map<string, string | undefined>();
  const remember = (key: string) => {
    if (!saved.has(key)) saved.set(key, process.env[key]);
  };

  process.chdir(root);
  if (homedir() !== root) throw new HarnessFailure(`homedir escaped run root: ${homedir()}`);
  if (tmpdir() !== root) throw new HarnessFailure(`tmpdir escaped run root: ${tmpdir()}`);

  const envKeysPath = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys');
  const listed = (await readFile(envKeysPath, 'utf8')).split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const exempt = new Set(['HOME', 'USERPROFILE']);
  for (const key of new Set([...listed, ...EXTRA_SCRUB])) {
    if (exempt.has(key)) continue;
    remember(key);
    delete process.env[key];
  }
  for (const key of Object.keys(process.env)) {
    if (exempt.has(key)) continue;
    if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/i.test(key)) {
      remember(key);
      delete process.env[key];
    }
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

const CHINA_TAIWAN_TICKERS = ['TSM', 'NVDA', 'AMD', 'ASML', 'INTC'];

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const map = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/osint/event-asset-map.ts')).href
    );

    const categories = map.detectCategories('Strait of Hormuz closure');
    const tickers = map.getAssetImplications(categories).map((asset: { ticker: string }) => asset.ticker);
    const leakedTickers = tickers.filter((ticker: string) => CHINA_TAIWAN_TICKERS.includes(ticker));
    const taiwanCategories = map.detectCategories('Taiwan Strait military exercises');

    const observed = JSON.stringify(
      {
        categories,
        impliedTickers: tickers,
        chipTickersFromOilStory: leakedTickers,
        taiwanStoryCategories: taiwanCategories,
        energySupplyTickers: map
          .getAssetImplications(['energy-supply'])
          .map((asset: { ticker: string; name: string }) => `${asset.ticker} (${asset.name})`),
      },
      null,
      2,
    );
    const expected = 'a Strait of Hormuz story yields only Middle East / Iran / energy categories, with no chip-maker tickers';
    const actual = `categories=${JSON.stringify(categories)}, chipTickers=${JSON.stringify(leakedTickers)}`;

    try {
      assert.equal(categories.includes('china-taiwan'), false, 'Hormuz matched china-taiwan');
      assert.deepEqual(leakedTickers, [], 'oil story produced chip-supply tickers');
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
