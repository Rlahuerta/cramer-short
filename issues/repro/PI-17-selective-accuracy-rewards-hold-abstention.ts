/**
 * PI-17: selective directional accuracy counts HOLD abstentions as correct
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * f40143d made directionalAccuracy() score traded (BUY/SELL) steps only and call
 * HOLD an abstention. selectiveDirectionalAccuracy() was missed: inside the
 * confidence-selected set it still counts a HOLD as correct whenever
 * |actualReturn| < holdThreshold. The selective accuracy (and the risk-coverage
 * curve built on it) therefore rewards abstention: a model that never trades
 * reports 100% selective accuracy while directionalAccuracy() reports 0.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');

class HarnessFailure extends Error {}

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  throw new HarnessFailure(reason);
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is unset');
  if (homedir() !== root) harnessFailure(`homedir() ${homedir()} escaped ${root}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() ${tmpdir()} escaped ${root}`);

  const savedCwd = process.cwd();
  const savedEnv: Record<string, string | undefined> = { ...process.env };
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
    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as typeof fetch;

    const metrics = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/backtest/metrics.ts')).href
    );

    const makeStep = (overrides: Record<string, unknown>) => ({
      t: 0,
      predictedProb: 0.5,
      actualBinary: 1,
      predictedReturn: 0,
      actualReturn: 0.001,
      ciLower: 95,
      ciUpper: 105,
      realizedPrice: 100.1,
      recommendation: 'HOLD',
      gofPasses: null,
      confidence: 0.95,
      ...overrides,
    });

    // Selected set: 3 traded (2 correct) + 7 flat HOLD abstentions.
    const mixed = [
      makeStep({ t: 0, recommendation: 'BUY', actualReturn: 0.05, actualBinary: 1 }),
      makeStep({ t: 1, recommendation: 'BUY', actualReturn: 0.04, actualBinary: 1 }),
      makeStep({ t: 2, recommendation: 'SELL', actualReturn: 0.03, actualBinary: 0 }),
      ...Array.from({ length: 7 }, (_, index) => makeStep({ t: 3 + index, recommendation: 'HOLD', actualReturn: 0.001 })),
    ];
    const mixedResult = metrics.selectiveDirectionalAccuracy(mixed, 0.5);
    const neverTrade = Array.from({ length: 20 }, (_, index) => makeStep({ t: index }));
    const neverTradeResult = metrics.selectiveDirectionalAccuracy(neverTrade, 0.5);
    const rc = metrics.computeRCCurve(neverTrade).slice(0, 2);

    console.log('OBSERVED:');
    console.log(`  mixed selective accuracy = ${mixedResult.accuracy.toFixed(6)} (selected=${mixedResult.selected})`);
    console.log(`  directionalAccuracy(mixed) = ${metrics.directionalAccuracy(mixed).toFixed(6)}`);
    console.log(`  never-trading selective accuracy = ${neverTradeResult.accuracy.toFixed(6)} coverage=${neverTradeResult.coverage.toFixed(6)}`);
    console.log(`  never-trading directionalAccuracy = ${metrics.directionalAccuracy(neverTrade).toFixed(6)}`);
    console.log(`  RC curve first point accuracy = ${rc[0].accuracy.toFixed(6)}`);

    let failure: string | null = null;
    try {
      assert.ok(
        Math.abs(mixedResult.accuracy - 2 / 3) < 1e-9,
        `selective accuracy must score the 3 traded selected rows only (2/3), got ${mixedResult.accuracy.toFixed(6)}`,
      );
      assert.ok(
        neverTradeResult.accuracy === 0,
        `a model that never trades must not score 100% selective accuracy, got ${neverTradeResult.accuracy.toFixed(6)}`,
      );
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      console.log('EXPECTED:');
      console.log('  mixed selective accuracy = 0.666667; never-trading selective accuracy = 0.000000');
      console.log('ACTUAL:');
      console.log(`  mixed = ${mixedResult.accuracy.toFixed(6)}; never-trading = ${neverTradeResult.accuracy.toFixed(6)} (HOLD scored as correct)`);
      process.exitCode = 42;
      return;
    }
    console.log('NO BUG');
  } finally {
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
