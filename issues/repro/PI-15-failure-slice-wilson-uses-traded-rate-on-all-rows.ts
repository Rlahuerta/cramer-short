/**
 * PI-15: failure-slice Wilson CI multiplies a traded-only rate by every row
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * metrics.directionalAccuracy() is a hit rate over traded (BUY/SELL) rows only,
 * but btc-failure-analysis.toSuccessCount() treats it as a per-row success rate
 * and multiplies it by the bucket's total row count (HOLDs included). The Wilson
 * interval is then computed from a fabricated sample, so a bucket with k correct
 * out of n traded rows reports the interval for round(k/n * bucketCount) out of
 * bucketCount.
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

function wilson95(successes: number, total: number): { lower: number; upper: number } {
  const z = 1.959963984540054;
  const p = successes / total;
  const denominator = 1 + (z * z) / total;
  const centre = p + (z * z) / (2 * total);
  const margin = z * Math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total);
  return { lower: (centre - margin) / denominator, upper: (centre + margin) / denominator };
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

    const failureAnalysis = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/backtest/btc-failure-analysis.ts')).href
    );

    const makeStep = (overrides: Record<string, unknown>) => ({
      t: 0,
      predictedProb: 0.5,
      actualBinary: 1,
      predictedReturn: 0.02,
      actualReturn: 0.03,
      ciLower: 90,
      ciUpper: 110,
      realizedPrice: 100,
      recommendation: 'BUY',
      gofPasses: null,
      confidence: 0.5,
      ...overrides,
    });

    const steps: Array<ReturnType<typeof makeStep>> = [];
    let t = 0;
    // Bear bucket: 20 rows, only 5 traded (2 correct), 15 HOLD.
    for (let i = 0; i < 2; i++) steps.push(makeStep({ t: t++, regime: 'bear', recommendation: 'BUY', actualReturn: 0.05, actualBinary: 1, confidence: 0.8 }));
    for (let i = 0; i < 3; i++) steps.push(makeStep({ t: t++, regime: 'bear', recommendation: 'SELL', actualReturn: 0.05, actualBinary: 1, confidence: 0.8 }));
    for (let i = 0; i < 15; i++) steps.push(makeStep({ t: t++, regime: 'bear', recommendation: 'HOLD', actualReturn: 0.001, actualBinary: 1, confidence: 0.5 }));
    // Bull bucket: 20 traded rows, all correct.
    for (let i = 0; i < 20; i++) steps.push(makeStep({ t: t++, regime: 'bull', recommendation: 'BUY', actualReturn: 0.05, actualBinary: 1, confidence: 0.8 }));

    const report = failureAnalysis.rankFailureBuckets(steps, 7);
    const bucket = report.rankedBuckets.find(
      (candidate: { dimension: string; bucket: string }) => candidate.dimension === 'regime' && candidate.bucket === 'bear',
    );
    if (!bucket) harnessFailure('bear regime bucket missing from the failure ranking');

    const expected = wilson95(2, 5); // 2 correct out of 5 traded
    const actual = bucket.accuracyCI;

    console.log('OBSERVED:');
    console.log(`  bucket rows=${bucket.n} traded-only accuracy=${bucket.directionalAccuracy.toFixed(6)}`);
    console.log(`  reported CI = [${actual.lower.toFixed(6)}, ${actual.upper.toFixed(6)}]`);
    console.log(`  wilson(2, 5) = [${expected.lower.toFixed(6)}, ${expected.upper.toFixed(6)}]`);
    console.log(`  wilson(round(0.4*20)=8, 20) = [${wilson95(8, 20).lower.toFixed(6)}, ${wilson95(8, 20).upper.toFixed(6)}]`);

    let failure: string | null = null;
    try {
      assert.ok(
        Math.abs(actual.lower - expected.lower) < 1e-9 && Math.abs(actual.upper - expected.upper) < 1e-9,
        `Wilson CI must come from the traded sample (2/5), got [${actual.lower.toFixed(6)}, ${actual.upper.toFixed(6)}]`,
      );
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      console.log('EXPECTED:');
      console.log(`  accuracyCI = [${expected.lower.toFixed(6)}, ${expected.upper.toFixed(6)}] from 2/5 traded rows`);
      console.log('ACTUAL:');
      console.log(`  accuracyCI = [${actual.lower.toFixed(6)}, ${actual.upper.toFixed(6)}] from 8/20 rows (HOLDs counted as misses)`);
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
