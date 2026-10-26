/**
 * PI-23: CLOB price history requests a 1-hour lookback for 6h/24h stats
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * enrichStructuredMarketMicrostructure() asks the CLOB /prices-history endpoint
 * for interval=1h, then computes computePriceVelocityPpH(history) (6h window)
 * and computeMaxHourlyJump(history) (24h window).  Polymarket's `interval`
 * parameter is the lookback span, not a bar size, so both statistics see only
 * the last hour.  The fixture below ramps +6pp/hour over the previous six hours
 * and is flat in the last hour: a correct 6h regression is ~5.5pp/h, but the
 * requested 1h lookback yields exactly 0.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const FOUND_AT = 'e3e2088b21acc4963b3c20661e5d15c739244409';

const EXPLICIT_SCRUB = [
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
const SECRET_RE = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/;

const NEWEST_SEC = 1_800_000_000;
const requestedIntervals: string[] = [];

function fixtureHistory(interval: string): { history: Array<{ t: number; p: number }> } {
  const spanSeconds = interval === '1h' ? 3_600 : 6 * 3_600;
  const history: Array<{ t: number; p: number }> = [];
  for (let back = spanSeconds; back >= 0; back -= 300) {
    const t = NEWEST_SEC - back;
    // +6pp/hour for the six hours before the last hour, then flat at 0.50.
    const p = back > 3_600 ? 0.2 + ((spanSeconds - back) / 3_600) * 0.06 : 0.5;
    history.push({ t, p });
  }
  return { history };
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

function fixtureFetch(input: string | URL | Request): Promise<Response> {
  const url = String(typeof input === 'string' ? input : input instanceof URL ? input : input.url);
  if (url.includes('gamma-api.polymarket.com/events')) {
    return Promise.resolve(json([{
      id: 'event-1',
      title: 'Bitcoin price',
      markets: [{
        id: 'market-1',
        conditionId: 'market-1',
        question: 'Will Bitcoin be above $70,000 on 2026-05-07?',
        outcomes: '["Yes","No"]',
        outcomePrices: '["0.5","0.5"]',
        clobTokenIds: '["token-yes-1","token-no-1"]',
        endDateIso: '2026-05-07T00:00:00.000Z',
        volume24hr: 500_000,
        liquidityNum: 500_000,
        active: true,
        closed: false,
        enableOrderBook: true,
      }],
    }]));
  }
  if (url.includes('/spread/')) {
    return Promise.resolve(json({ spread: 0.01 }));
  }
  if (url.includes('/prices-history')) {
    const match = /[?&]interval=([^&]+)/.exec(url);
    const interval = match ? decodeURIComponent(match[1]!) : '1h';
    requestedIntervals.push(interval);
    return Promise.resolve(json(fixtureHistory(interval)));
  }
  return Promise.reject(new Error(`network disabled: ${url}`));
}

function normalize(text: string, root: string): string {
  return text
    .replaceAll(root, '<TMP>')
    .replace(/\u001b\[[0-9;]*m/g, '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) {
    console.log('HARNESS-FAILURE: CS_REPRO_ROOT unset');
    process.exitCode = 43;
    return;
  }

  const originalCwd = process.cwd();
  const savedEnv = new Map<string, string | undefined>();
  const realFetch = globalThis.fetch;

  const fail = (reason: string): never => {
    console.log(`HARNESS-FAILURE: ${reason}`);
    process.exitCode = 43;
    throw new Error(`HARNESS-FAILURE: ${reason}`);
  };

  try {
    process.chdir(root);
    if (homedir() !== root) fail(`homedir escape: ${homedir()}`);
    if (tmpdir() !== root) fail(`tmpdir escape: ${tmpdir()}`);

    // --- env scrub (before every dynamic import) -------------------------
    const envKeysRaw = await readFile(
      join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys'),
      'utf8',
    );
    const keys = new Set<string>([
      ...envKeysRaw.split(/\r?\n/).map((k) => k.trim()).filter(Boolean),
      ...EXPLICIT_SCRUB,
    ]);
    for (const k of Object.keys(process.env)) {
      if (SECRET_RE.test(k)) keys.add(k);
    }
    keys.delete('HOME');
    keys.delete('USERPROFILE');
    for (const k of keys) {
      savedEnv.set(k, process.env[k]);
      delete process.env[k];
    }
    for (const k of Object.keys(process.env)) {
      if (SECRET_RE.test(k)) fail(`secret key survived scrub: ${k}`);
    }

    // --- in-process fixture stub (no real network) -----------------------
    globalThis.fetch = fixtureFetch as unknown as typeof globalThis.fetch;

    // --- production import after chdir + scrub ---------------------------
    const polymarket = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/polymarket.ts')).href
    )) as typeof import('../../src/tools/finance/polymarket.ts');

    const markets = await polymarket.fetchPolymarketAnchorMarketsWithQueries(
      ['Bitcoin price'],
      1,
      { ticker: 'BTC', horizonDays: 7, enrichMicrostructure: true },
    );
    const velocity = markets[0]?.priceVelocityPpH;
    const observed =
      `priceVelocityPpH=${String(velocity)}; ` +
      `requested interval(s)=${JSON.stringify(requestedIntervals)}`;

    assert.equal(
      typeof velocity === 'number' && Math.abs(velocity) > 1,
      true,
      `${observed}; a 6-hour regression cannot see the six-hour ramp with a one-hour lookback`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log('EXPECTED: |priceVelocityPpH| > 1 for a six-hour +6pp/hour ramp');
      console.log('ACTUAL: priceVelocityPpH=0 because only the last 1h of history was requested');
      process.exitCode = 42;
    }
  } finally {
    globalThis.fetch = realFetch;
    for (const [k, v] of savedEnv) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    process.chdir(originalCwd);
  }
}

await main();
