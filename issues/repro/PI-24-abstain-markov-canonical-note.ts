/**
 * PI-24: Canonical-Markov prompt guard fires for abstained payloads
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `buildIterationPrompt()` gates the canonical-Markov instructions on the mere
 * presence of `"_tool":"markov_distribution"` in the tool results — it never
 * checks `status`. For an ABSTAINED payload (canonical.scenarios === null,
 * distribution === null) it still tells the model to "Use the reported scenario
 * bucket probabilities ... verbatim", and when the query names N parts/buckets it
 * additionally appends "Preserve that N-part bucket granularity in the final
 * answer when presenting the canonical Markov distribution". The same prompt also
 * contains the abstain guard saying no distribution is available and that
 * synthesizing bucket probabilities is forbidden, so the model receives two
 * contradictory authoritative instructions on a path whose formatter
 * (ensureStructuredDensityTable) correctly emits no table.
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

const ABSTAIN_PAYLOAD = JSON.stringify({
  data: {
    _tool: 'markov_distribution',
    status: 'abstain',
    manualSynthesisForbidden: true,
    abstainReasons: ['No horizon-matched terminal threshold markets'],
    report: 'Diagnostics-only Markov output',
    canonical: {
      ticker: 'BTC-USD',
      currentPrice: 100000,
      horizon: 7,
      scenarios: null,
      actionSignal: null,
      diagnostics: {
        totalAnchors: 0,
        trustedAnchors: 0,
        anchorQuality: 'poor',
        predictionConfidence: 0.1,
        status: 'abstain',
        canEmitCanonical: false,
        manualSynthesisForbidden: true,
        abstainReasons: ['No horizon-matched terminal threshold markets'],
      },
    },
    distribution: null,
    decisionDistribution: null,
    trajectory: null,
  },
  sourceUrls: [],
});

const OK_PAYLOAD = JSON.stringify({
  data: {
    _tool: 'markov_distribution',
    status: 'ok',
    canonical: {
      ticker: 'BTC-USD',
      currentPrice: 100000,
      horizon: 7,
      scenarios: { pUp: 0.55, expectedReturn: 0.03, buckets: [{ label: 'flat', probability: 0.4 }] },
      actionSignal: { expectedReturn: 0.03, confidence: 'MEDIUM' },
      diagnostics: { predictionConfidence: 0.6 },
    },
    distribution: [
      { price: 90000, probability: 0.95 },
      { price: 100000, probability: 0.5 },
      { price: 110000, probability: 0.05 },
    ],
  },
  sourceUrls: [],
});

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
    }) as unknown as typeof fetch;

    const { buildIterationPrompt } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/agent/prompts.ts')).href
    );

    const query = 'Show the BTC-USD 7-day price probability distribution in 5 buckets';
    const build = (payload: string) => buildIterationPrompt(
      query,
      `### markov_distribution(ticker=BTC-USD, horizon=7)\n${payload}`,
      null,
      null,
      query,
    );

    const abstainPrompt = build(ABSTAIN_PAYLOAD);
    const okPrompt = build(OK_PAYLOAD);

    const checks = {
      abstainHasScenarioVerbatim: abstainPrompt.includes('Use the reported scenario bucket probabilities and diagnostics verbatim'),
      abstainHasBucketGranularity: abstainPrompt.includes('Preserve that 5-part bucket granularity'),
      abstainHasAbstainGuard: abstainPrompt.includes('markov_distribution explicitly abstained'),
      abstainHasProxyGuard: abstainPrompt.includes('Do NOT substitute another asset, proxy ticker'),
      okHasScenarioVerbatim: okPrompt.includes('Use the reported scenario bucket probabilities and diagnostics verbatim'),
      okHasBucketGranularity: okPrompt.includes('Preserve that 5-part bucket granularity'),
    };

    console.log('OBSERVED:');
    console.log(`  abstained payload: scenario-verbatim instruction = ${String(checks.abstainHasScenarioVerbatim)}`);
    console.log(`  abstained payload: 5-part bucket-granularity instruction = ${String(checks.abstainHasBucketGranularity)}`);
    console.log(`  abstained payload: abstain guard = ${String(checks.abstainHasAbstainGuard)}`);
    console.log(`  abstained payload: proxy/anchor guard = ${String(checks.abstainHasProxyGuard)}`);
    console.log(`  ok payload: scenario-verbatim instruction = ${String(checks.okHasScenarioVerbatim)}`);
    console.log(`  ok payload: 5-part bucket-granularity instruction = ${String(checks.okHasBucketGranularity)}`);

    let failure: string | null = null;
    try {
      // Control: a non-abstaining payload must keep both canonical instructions.
      assert.equal(checks.okHasScenarioVerbatim, true, 'control: ok payload must keep the scenario-verbatim instruction');
      assert.equal(checks.okHasBucketGranularity, true, 'control: ok payload must keep the bucket-granularity instruction');
      // The proxy/anchor paragraph is valid for abstain (diagnostics exist) and must stay.
      assert.equal(checks.abstainHasProxyGuard, true, 'abstained payload must keep the proxy/anchor guard');
      assert.equal(checks.abstainHasAbstainGuard, true, 'abstained payload must keep the abstain guard');
      // Defect: no distribution exists, so neither distribution instruction may fire.
      assert.equal(
        checks.abstainHasScenarioVerbatim,
        false,
        'abstained payload must not instruct the model to use non-existent scenario bucket probabilities',
      );
      assert.equal(
        checks.abstainHasBucketGranularity,
        false,
        'abstained payload must not instruct the model to preserve bucket granularity for a missing distribution',
      );
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      console.log('EXPECTED:');
      console.log('  abstained payload: scenario-verbatim = false; bucket-granularity = false; abstain guard = true; proxy guard = true');
      console.log('  ok payload: scenario-verbatim = true; bucket-granularity = true');
      console.log('ACTUAL:');
      console.log(`  abstained payload: scenario-verbatim = ${String(checks.abstainHasScenarioVerbatim)}; bucket-granularity = ${String(checks.abstainHasBucketGranularity)} (no distribution exists)`);
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
