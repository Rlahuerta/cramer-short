/**
 * PI-13: momentum velocity annualizes losses with |r|, producing returns below -100%
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `computeMomentumSignal` annualizes a total return r over L days as
 * sign(r)·((1+|r|)^(252/L) − 1). For r < 0 the geometric annualization is
 * (1+r)^(252/L) − 1; the mirrored form is ~3x larger and can exceed -100%,
 * which is impossible for a long-only price series. The inflated velocity is
 * then divided by 252 and clamped, so downtrend drift tilts saturate the
 * ±0.003/day clamp the lossy way.
 */

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const FOUND_AT = 'e3e2088b21acc4963b3c20661e5d15c739244409';

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

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const regime = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/markov-distribution/regime.ts')).href
    );

    const lookback = 20;
    // Steady -0.5%/day decline: total return = 0.995^20 - 1 ≈ -9.54%.
    const prices = Array.from({ length: lookback + 1 }, (_, i) => 100 * Math.pow(0.995, i));
    const totalReturn = prices[prices.length - 1] / prices[0] - 1;
    const correctVelocity = Math.pow(1 + totalReturn, 252 / lookback) - 1;
    const signal = regime.computeMomentumSignal(prices, lookback);

    const observed = JSON.stringify(
      {
        totalReturn,
        lookback,
        reportedVelocity: signal.velocity,
        geometricVelocity: correctVelocity,
        reportedAdjustmentPerDay: signal.adjustment,
        trendStrength: signal.trendStrength,
      },
      null,
      2,
    );
    const expected =
      'reported velocity equals the geometric annualization (1+r)^(252/lookback)-1 and stays above -100%';
    const actual = `reported velocity=${signal.velocity} (${(signal.velocity * 100).toFixed(1)}% annualized) vs geometric ${correctVelocity} (${(correctVelocity * 100).toFixed(1)}%)`;

    try {
      assert.ok(
        Math.abs(signal.velocity - correctVelocity) < 0.01 && signal.velocity > -1,
        'loss annualization used |r| instead of (1+r)',
      );
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
