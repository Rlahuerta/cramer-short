/**
 * PI-53: bundles with no markets are bucketed as 'terminal' semantics
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * primarySemantics()/summarizeSemantics() pick the highest-count entry of a
 * six-key counter and fall back with `?? 'unknown'`. That fallback is
 * unreachable: with zero markets every count is 0, the stable sort keeps
 * 'terminal' first, so a bundle that carries no Polymarket markets is reported
 * as 'terminal' and pollutes the terminal semantic-calibration bucket. The same
 * idiom appears in combined-short-horizon-benchmark.summarizeSemantics().
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

    const runner = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/backtest/arbiter-replay-runner.ts')).href
    );

    const bundle = {
      capturedAt: '2024-06-03T14:30:00.000Z',
      ticker: 'BTC-USD',
      horizonDays: 1,
      currentPrice: 68000,
      labels: {
        forecast: {
          realizedPrice: 68500,
          realizedReturn: 0.0074,
          actualBinary: 1,
          labeledAt: '2024-06-05T00:00:00.000Z',
        },
      },
      warnings: [],
      // no polymarket object at all
    };

    const stubEvaluator = {
      name: 'stub',
      evaluate: () => ({
        ticker: 'BTC-USD',
        horizonDays: 1,
        currentPrice: 68000,
        leverage: 1,
        verdict: 'LONG',
        preferredDirection: 'long',
        confidence: 'medium',
        shouldEnterNow: true,
        semanticSummary: { primaryPolymarketSemantics: 'unknown', counts: {}, barrierPrices: [], reconciliation: '' },
        disagreement: { markovDirection: 'long', polymarketDirection: 'neutral', whaleDirection: 'neutral', isDivergent: false, summary: '' },
        leverageAssessment: {
          long: { directionalEdgePct: 0, riskAdjustedScore: 0, leveragePnlPct: 0, rr: null, notes: [] },
          short: { directionalEdgePct: 0, riskAdjustedScore: 0, leveragePnlPct: 0, rr: null, notes: [] },
          warning: null,
        },
        conditionalPlan: { longTrigger: null, shortTrigger: null, invalidation: null },
        policy: { level: 'full', horizonEligible: true, tradeEligible: true, reasons: [] },
        rationale: [],
        rawEvidence: { markov: null, polymarket: null, whale: null },
      }),
    };

    const result = runner.runArbiterReplay({ bundles: [bundle], baselineEvaluator: stubEvaluator });
    const row = result.baseline.rows[0];
    const buckets = Object.keys(result.baseline.semanticBucketCalibration);

    console.log('OBSERVED:');
    console.log(`  row.primarySemantics = ${JSON.stringify(row?.primarySemantics)}`);
    console.log(`  semanticBucketCalibration keys = ${JSON.stringify(buckets)}`);

    let failure: string | null = null;
    try {
      assert.equal(row?.primarySemantics, 'unknown');
      assert.deepEqual(buckets, ['unknown']);
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      console.log('EXPECTED:');
      console.log("  row.primarySemantics = \"unknown\"; buckets = [\"unknown\"]");
      console.log('ACTUAL:');
      console.log(`  row.primarySemantics = ${JSON.stringify(row?.primarySemantics)}; buckets = ${JSON.stringify(buckets)}`);
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
