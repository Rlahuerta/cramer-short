/**
 * PI-21: substring improvement-term matching diverts ordinary queries into forecast-lab
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * routeForecastLabQuery() treats an improvement term as matched whenever the raw
 * query CONTAINS it, so "fixed-income" matches "fix", "fixture" matches "fix",
 * and "fortune" matches "tune". getForecastLabRoutingHint() then returns a
 * non-null hint (shouldInvokeSkill=true) which the agent injects as a
 * "bounded forecast-workflow improvement task" and the forecast-lab skill is
 * called before ordinary forecast/data tools. Word-boundary matching is the
 * stated intent (docs/reviews/deep-code-review-2026-10-23.md §5 "Substring routing").
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

  try {
    process.chdir(root);
    if (homedir() !== root) {
      console.log(`HARNESS-FAILURE: homedir escape: ${homedir()}`);
      process.exitCode = 43;
      return;
    }
    if (tmpdir() !== root) {
      console.log(`HARNESS-FAILURE: tmpdir escape: ${tmpdir()}`);
      process.exitCode = 43;
      return;
    }

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
      if (SECRET_RE.test(k)) {
        console.log(`HARNESS-FAILURE: secret key survived scrub: ${k}`);
        process.exitCode = 43;
        return;
      }
    }

    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as unknown as typeof globalThis.fetch;

    const router = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/experiments/forecast-lab/router.ts')).href
    )) as typeof import('../../src/experiments/forecast-lab/router.ts');
    const queryRouter = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/experiments/forecast-lab/query-router.ts')).href
    )) as typeof import('../../src/experiments/forecast-lab/query-router.ts');

    const fixtures = [
      { query: 'What is the fixed-income outlook for the next year?', offending: 'fix (from fixed-income)' },
      { query: 'Compare the fixture data quality across providers.', offending: 'fix (from fixture)' },
      { query: 'What is the fortune 500 earnings trend?', offending: 'tune (from fortune)' },
    ];

    const observedRows = fixtures.map(({ query, offending }) => {
      const route = router.routeForecastLabQuery(query);
      const hint = queryRouter.getForecastLabRoutingHint(query);
      return {
        query,
        offending,
        intent: route.intent,
        preferredProfileId: route.preferredProfileId,
        routingHintInjected: hint !== null,
      };
    });

    const observed = observedRows
      .map((row) =>
        `query="${row.query}" matched ${row.offending}; intent=${row.intent}; ` +
        `preferredProfileId=${String(row.preferredProfileId)}; routingHintInjected=${row.routingHintInjected}`)
      .join('\n');

    try {
      for (const row of observedRows) {
        assert.equal(
          row.intent,
          'none',
          `"${row.query}" matched the improvement cue ${row.offending} and routed as ${row.intent}`,
        );
        assert.equal(
          row.routingHintInjected,
          false,
          `"${row.query}" injected a forecast-lab routing hint (skill diverted before ordinary forecast tools)`,
        );
      }
      console.log('NO BUG');
    } catch (error) {
      if (error instanceof assert.AssertionError) {
        console.log('OBSERVED:');
        for (const line of normalize(observed, root).split('\n')) {
          console.log(`  ${line}`);
        }
        console.log('EXPECTED: intent=none and routingHintInjected=false for every fixture (word-boundary improvement terms only)');
        console.log('ACTUAL: intent=improvement and routingHintInjected=true because "fix"/"tune" match inside fixed-income/fixture/fortune');
        process.exitCode = 42;
        return;
      }
      throw error;
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
