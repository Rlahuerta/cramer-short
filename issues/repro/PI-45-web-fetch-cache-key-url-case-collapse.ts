/**
 * PI-45: web_fetch cache key lowercases the URL, serving another page's content
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `runWebFetch` builds its cache key as `fetch:${url}:${mode}:${maxChars}` and then
 * passes it through `normalizeCacheKey`, which lowercases the whole string — URL
 * path and query included. URLs that differ only by case therefore share one cache
 * entry, so fetching `https://host/Report` and then `https://host/report` returns
 * the first page's body (and reports the first URL) for the second request without
 * ever hitting the transport. Path- and value-case-sensitive servers return wrong
 * data; `?Ticker=AAPL` and `?ticker=aapl` collapse too.
 *
 * The transport is replaced through the module's own `_setHttpFetchForTest` seam,
 * so no network request is made.
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

const UPPER_URL = 'https://example.com/Report';
const LOWER_URL = 'https://example.com/report';

function page(marker: string): Response {
  return new Response(
    `<html><head><title>${marker}</title></head><body><p>${marker} body</p></body></html>`,
    { status: 200, headers: { 'content-type': 'text/html' } },
  );
}

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const mod = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/fetch/web-fetch.ts')).href
    );

    const transportCalls: string[] = [];
    mod._setHttpFetchForTest(async (url: string) => {
      transportCalls.push(url);
      return page(url === UPPER_URL ? 'UPPERCASE_PAGE' : 'lowercase_page');
    });

    const first = JSON.parse(String(await mod.webFetchTool.invoke({ url: UPPER_URL, maxChars: 1000 }))).data;
    const second = JSON.parse(String(await mod.webFetchTool.invoke({ url: LOWER_URL, maxChars: 1000 }))).data;

    const observed = JSON.stringify(
      {
        transportCalls,
        firstUrl: first.url,
        firstHasUppercaseBody: String(first.text).includes('UPPERCASE_PAGE'),
        secondUrl: second.url,
        secondCached: second.cached === true,
        secondHasLowercaseBody: String(second.text).includes('lowercase_page'),
        secondRepeatsUppercaseBody: String(second.text).includes('UPPERCASE_PAGE'),
      },
      null,
      2,
    );
    const expected = 'the case-distinct URL is fetched on its own and returns its own body';
    const actual = `transport calls=${JSON.stringify(transportCalls)}, second.cached=${String(second.cached)}, second.url=${String(second.url)}`;

    try {
      assert.deepEqual(transportCalls, [UPPER_URL, LOWER_URL], 'case-distinct URL reused the cached entry');
      assert.ok(String(second.text).includes('lowercase_page'), 'wrong page served');
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
