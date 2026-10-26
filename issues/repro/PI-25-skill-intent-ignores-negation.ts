/**
 * PI-25: Negated or questioning "use the X skill" requests still trigger intent
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `detectExplicitSkillRequest()` matches `use [the] <name> skill` anywhere in the
 * query and never checks for a negation or a question cue, so "Don't use the
 * portfolio_risk skill" resolves to portfolio_risk and Agent.run() pre-executes
 * that skill (agent.ts:364,374-389), injecting the workflow the user declined and
 * marking sequential_thinking as satisfied. The sibling exclusion in
 * `isCryptoForecastQuery()` has the same omission: a negated/questioning mention
 * of the probability_assessment skill disables the whole crypto forecast pipeline.
 *
 * The repo already treats negated/questioning routing requests as non-intent for
 * forecast-lab promotion/reset (87dd643), so this is the remaining matcher that
 * ignores the same cues.
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
    }) as unknown as typeof fetch;

    const { detectExplicitSkillRequest, isCryptoForecastQuery } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/agent/query-router/classification.ts')).href
    );

    const skillCases = [
      { label: 'affirmative', query: 'Use the portfolio_risk skill to analyse AAPL risk', shouldFire: true },
      { label: 'negated (don\'t)', query: "Don't use the portfolio_risk skill for AAPL", shouldFire: false },
      { label: 'negated (do not)', query: 'Do not use the portfolio_risk skill; just summarise the risk', shouldFire: false },
      { label: 'question', query: 'Should I use the portfolio_risk skill for AAPL?', shouldFire: false },
    ] as const;
    const detected = skillCases.map((entry) => ({
      ...entry,
      actual: detectExplicitSkillRequest(entry.query),
    }));

    const routingCases = [
      { label: 'plain forecast', query: 'BTC-USD 7-day price forecast', shouldRoute: true },
      {
        label: 'negated skill mention',
        query: "Don't use the probability_assessment skill; BTC-USD 7-day price forecast",
        shouldRoute: true,
      },
      {
        label: 'questioned skill mention',
        query: 'Should I use the probability_assessment skill for a BTC-USD 7-day price forecast?',
        shouldRoute: true,
      },
    ] as const;
    const routed = routingCases.map((entry) => ({
      ...entry,
      actual: isCryptoForecastQuery(entry.query),
    }));

    console.log('OBSERVED:');
    for (const entry of detected) {
      console.log(`  ${entry.label}: detectExplicitSkillRequest(...) = ${JSON.stringify(entry.actual)} (expected ${entry.shouldFire ? 'a skill' : 'null'})`);
    }
    for (const entry of routed) {
      console.log(`  ${entry.label}: isCryptoForecastQuery(...) = ${String(entry.actual)} (expected ${String(entry.shouldRoute)})`);
    }

    let failure: string | null = null;
    try {
      for (const entry of detected) {
        if (entry.shouldFire) {
          assert.equal(entry.actual, 'portfolio_risk', `affirmative request must still resolve the skill, got ${String(entry.actual)}`);
        } else {
          assert.equal(
            entry.actual,
            null,
            `${entry.label} request must not resolve a skill, got ${String(entry.actual)}`,
          );
        }
      }
      for (const entry of routed) {
        assert.equal(
          entry.actual,
          entry.shouldRoute,
          `${entry.label} must not change crypto-forecast routing (expected ${String(entry.shouldRoute)})`,
        );
      }
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      const wrongSkills = detected.filter((entry) => !entry.shouldFire && entry.actual !== null);
      const wrongRouting = routed.filter((entry) => entry.shouldRoute && entry.actual !== true);
      console.log('EXPECTED:');
      console.log('  negated/questioned skill mentions resolve no skill; crypto forecast routing is unchanged');
      console.log('ACTUAL:');
      console.log(`  detected skills: ${wrongSkills.map((entry) => `${entry.label}=${String(entry.actual)}`).join(', ') || 'none'}`);
      console.log(`  disabled crypto routing: ${wrongRouting.map((entry) => entry.label).join(', ') || 'none'}`);
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
