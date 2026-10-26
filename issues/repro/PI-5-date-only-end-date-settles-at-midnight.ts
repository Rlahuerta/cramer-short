/**
 * PI-5: date-only market end dates settle replay labels at midnight UTC
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * Polymarket's endDateIso is date-only for daily markets ("2026-06-30").
 * Date.parse() reads that as 2026-06-30T00:00:00Z, so settlementPoint() takes
 * the first intraday observation on the settlement day instead of the end-of-day
 * close.  Here the 01:00 print says 120,000 (YES) while the 20:00 close says
 * 90,000 (NO); the labeler records YES.
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

    // --- network trap ----------------------------------------------------
    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as unknown as typeof globalThis.fetch;

    // --- production import after chdir + scrub ---------------------------
    const labeler = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/arbiter-replay-labeler.ts')).href
    )) as typeof import('../../src/tools/finance/arbiter-replay-labeler.ts');

    const bundle = {
      capturedAt: '2026-06-20T00:00:00.000Z',
      ticker: 'BTC',
      horizonDays: 10,
      currentPrice: 100_000,
      polymarket: {
        querySet: ['bitcoin price'],
        selectedMarketIds: ['pm-daily-1'],
        selectedMarkets: [{
          marketId: 'pm-daily-1',
          assetId: 'token-yes-1',
          question: 'Will BTC be above $100,000 on June 30?',
          probability: 0.6,
          volume24h: 500_000,
          // Daily market: the date is the settlement day, not 00:00 UTC.
          endDate: '2026-06-30',
          semantics: 'terminal' as const,
          extractedPriceLevels: [100_000],
        }],
        warnings: [],
      },
      warnings: [],
    };

    const history = {
      points: [
        { at: '2026-06-20T00:00:00.000Z', price: 100_000 },
        { at: '2026-06-30T01:00:00.000Z', price: 120_000 },
        { at: '2026-06-30T20:00:00.000Z', price: 90_000 },
      ],
    };

    const labeled = labeler.labelReplayBundle(bundle, history, '2026-07-01T00:00:00.000Z');
    const outcome = labeled.labels?.semantic?.[0]?.outcome;
    const observed =
      `semantic outcome=${String(outcome)}; settlement-day prints were ` +
      '01:00=120000 (above) and 20:00=90000 (below)';

    assert.equal(
      outcome,
      'no',
      `${observed}; the end-of-day close says NO but the midnight target picked the 01:00 print`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log('EXPECTED: semantic outcome=no (settlement uses the June 30 end-of-day close)');
      console.log('ACTUAL: semantic outcome=yes (date-only endDate parsed as 00:00 UTC)');
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
