/**
 * PI-6: replay bundles with ambiguous market semantics are dropped on read
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * freezePolymarketReplayBlock() stores the classifier output verbatim, and
 * classifyPolymarketQuestion() can return 'ambiguous'.  The reader's
 * isReplaySemantics() guard accepts only terminal/barrier_touch/range/
 * path_dependent/unknown, so a bundle containing an 'ambiguous' market is
 * rejected as malformed and silently skipped by readArbiterReplayBundles().
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
    const replay = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/arbiter-replay.ts')).href
    )) as typeof import('../../src/tools/finance/arbiter-replay.ts');

    const question = 'Will BTC close above $80,000 or hit $90,000 by May?';
    const polymarket = replay.freezePolymarketReplayBlock({
      querySet: ['btc price'],
      selectedMarkets: [{
        marketId: 'pm-ambiguous-1',
        assetId: 'token-yes-1',
        question,
        probability: 0.42,
        volume24h: 250_000,
        endDate: '2026-05-07T00:00:00.000Z',
      }],
    });

    const writtenSemantics = polymarket.selectedMarkets[0]?.semantics;
    const bundle = {
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      horizonDays: 7,
      currentPrice: 68_000,
      polymarket,
      warnings: [],
    };

    const parsed = replay.parseArbiterReplayBundleLine(JSON.stringify(bundle));
    const observed =
      `written semantics=${String(writtenSemantics)}; ` +
      `parseArbiterReplayBundleLine()=${parsed === null ? 'null (bundle dropped)' : 'bundle accepted'}`;

    assert.notEqual(
      parsed,
      null,
      `${observed}; an 'ambiguous' market is written by the same module but rejected on read`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log('EXPECTED: the written bundle round-trips through parseArbiterReplayBundleLine()');
      console.log('ACTUAL: the whole bundle is rejected because isReplaySemantics() omits ambiguous');
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
