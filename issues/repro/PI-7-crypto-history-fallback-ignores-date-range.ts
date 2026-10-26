/**
 * PI-7: crypto history fallback ignores the requested date range
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * getCryptoPrices() asks for "start_date → end_date" but, when the primary
 * provider fails, it only forwards the WINDOW LENGTH (`days`) to
 * fetchBinanceDailyCloses(). That helper anchors its klines window to
 * `Date.now()` — so a request for 2021-01-01 → 2021-01-10 returns the last ten
 * days (2026) instead. The fallback payload is `{close, index}` with no dates,
 * so the caller cannot detect that the window was substituted.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');
const START = '2021-01-01';
const END = '2021-01-10';
const START_MS = Date.UTC(2021, 0, 1);
const END_MS = Date.UTC(2021, 0, 10) + 86_399_999;
const EXPECTED_CLOSE = 42300;
const WRONG_WINDOW_CLOSE = 99999;

class HarnessFailure extends Error {}

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  throw new HarnessFailure(reason);
}

function normalize(text: string, root: string): string {
  return text
    .replaceAll(root, '<TMP>')
    .replace(/\u001b\[[0-9;]*m/g, '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

const isoTimestamp = (ms: number): string => new Date(ms).toISOString();

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is unset');
  if (homedir() !== root) harnessFailure(`homedir() ${homedir()} escaped ${root}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() ${tmpdir()} escaped ${root}`);

  const savedCwd = process.cwd();
  const savedEnv: Record<string, string | undefined> = { ...process.env };
  const savedFetch = globalThis.fetch;
  const keys = readFileSync(join(EVIDENCE, 'pre-work.envkeys'), 'utf-8')
    .split(/\s+/)
    .filter(Boolean);
  const extra = [
    'OLLAMA_BASE_URL', 'DEXTER_GATEWAY_CONFIG', 'DEXTER_PAIRING_PATH',
    'CRAMER_SHORT_SESSIONS_DIR', 'KALSHI_API_KEY', 'OPENROUTER_API_KEY',
    'E2E_MODEL', 'FORECAST_LAB_OUTPUT_METRICS', 'COVERAGE_LINES_MIN', 'TEST_API_KEY',
  ];
  const secretPattern = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/;

  try {
    process.chdir(root);
    for (const key of [...keys, ...extra]) {
      if (key !== 'HOME' && key !== 'USERPROFILE') delete process.env[key];
    }
    for (const key of Object.keys(process.env)) {
      if (secretPattern.test(key)) delete process.env[key];
    }
    for (const key of Object.keys(process.env)) {
      if (secretPattern.test(key)) harnessFailure(`secret survived: ${key}`);
    }

    let klinesUrl = '';
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (url.includes('api.binance.com')) {
        klinesUrl = url;
        const parsed = new URL(url);
        const startTime = Number(parsed.searchParams.get('startTime'));
        // Only the requested 2021 window gets the real historical fixture.
        const inRequestedWindow = Math.abs(startTime - START_MS) <= 86_400_000;
        const close = inRequestedWindow ? EXPECTED_CLOSE : WRONG_WINDOW_CLOSE;
        return Response.json([
          [startTime, '42000', '42500', '41800', String(close), '10000', startTime + 86_399_999, '0', 0, '0', '0', '0'],
        ]);
      }
      if (url.includes('financialdatasets.ai')) {
        return new Response('payment required', { status: 402, statusText: 'Payment Required' });
      }
      throw new Error(`network disabled: ${url}`);
    }) as typeof fetch;

    const { getCryptoPrices } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/crypto.ts')).href
    );

    const raw = await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval: 'day',
      interval_multiplier: 1,
      start_date: START,
      end_date: END,
    });
    const parsed = JSON.parse(raw) as { data?: Array<{ close: number }> };
    const closes = Array.isArray(parsed.data) ? parsed.data.map((row) => row.close) : [];

    const requestParams = klinesUrl ? new URL(klinesUrl).searchParams : null;
    const requestStart = Number(requestParams?.get('startTime'));
    const requestEnd = Number(requestParams?.get('endTime'));

    const emit = (line: string): void => { console.log(normalize(line, root)); };
    const observed = [
      `  requested range ${START} → ${END}`,
      `  binance request starts at ${isoTimestamp(requestStart)} (requested ${isoTimestamp(START_MS)})`,
      `  binance request ends at ${isoTimestamp(requestEnd)} (requested ${isoTimestamp(END_MS)})`,
      `  returned closes = ${JSON.stringify(closes)}`,
    ];

    let failure: string | null = null;
    try {
      assert.ok(klinesUrl.length > 0, 'fallback never queried Binance');
      assert.ok(
        Math.abs(requestStart - START_MS) <= 86_400_000,
        `Binance startTime ${isoTimestamp(requestStart)} is not the requested ${isoTimestamp(START_MS)}`,
      );
      assert.ok(
        Math.abs(requestEnd - END_MS) <= 86_400_000,
        `Binance endTime ${isoTimestamp(requestEnd)} is not the requested ${isoTimestamp(END_MS)}`,
      );
      assert.deepEqual(closes, [EXPECTED_CLOSE]);
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      emit('OBSERVED:');
      for (const line of observed) emit(line);
      emit('EXPECTED:');
      emit(`  klines bounded by the requested range; closes = [${EXPECTED_CLOSE}]`);
      emit('ACTUAL:');
      emit(`  klines anchored to the run date; closes = ${JSON.stringify(closes)}`);
      process.exitCode = 42;
      return;
    }
    console.log('NO BUG');
  } finally {
    globalThis.fetch = savedFetch;
    process.chdir(savedCwd);
    for (const key of Object.keys(process.env)) delete process.env[key];
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value !== undefined) process.env[key] = value;
    }
  }
}

try {
  await main();
} catch (error) {
  if (error instanceof HarnessFailure) {
    process.exitCode = 43;
  } else {
    throw error;
  }
}
