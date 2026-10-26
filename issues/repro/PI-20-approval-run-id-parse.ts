/**
 * PI-20: approval source-run-id parser matches an unanchored "run" substring
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * extractForecastLabSourceRunId() builds /(?:source\s+run|run)\s+(id)/ WITHOUT a
 * leading word boundary and takes only the first match. Any token ending in
 * "run" ("rerun", "overrun") therefore wins the first match: "Rerun the check
 * for run <id>" yields no id at all, and "the overrun 2026-01-01 ... promote run
 * <id>" yields the unrelated token "2026-01-01". The approval hint then carries
 * a missing or wrong sourceRunId, and agent.ts:817 forwards it to
 * forecast_lab_run promote-approved, which falls back to the latest kept run.
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

    const queryRouter = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/experiments/forecast-lab/query-router.ts')).href
    )) as typeof import('../../src/experiments/forecast-lab/query-router.ts');

    const requestedRunId = 'forecast-lab-sol-2026-01-01T00-00-00-000Z';
    const fixtures = [
      {
        label: 'plain approval',
        query: `Approve forecast-lab promotion for sol-markov-short-horizon run ${requestedRunId}`,
      },
      {
        label: 'approval that mentions rerun',
        query: `Approve forecast-lab promotion for sol-markov-short-horizon. Rerun the check for run ${requestedRunId}.`,
      },
      {
        label: 'approval that mentions an overrun date',
        query: `Approve forecast-lab promotion for sol-markov-short-horizon; the overrun 2026-01-01 was fine, promote run ${requestedRunId}`,
      },
    ];

    const observedRows = fixtures.map(({ label, query }) => {
      const hint = queryRouter.detectForecastLabPromotionApproval(query);
      return {
        label,
        profileId: hint?.profileId,
        sourceRunId: hint?.sourceRunId,
      };
    });

    const observed = observedRows
      .map((row) => `${row.label}: sourceRunId=${String(row.sourceRunId)} (profileId=${String(row.profileId)})`)
      .join('\n');

    try {
      assert.ok(observedRows[0]!.sourceRunId === requestedRunId, 'baseline approval lost the source run id');
      for (const row of observedRows.slice(1)) {
        assert.equal(
          row.sourceRunId,
          requestedRunId,
          `${row.label}: expected the explicitly named run id, got ${String(row.sourceRunId)}`,
        );
      }
      console.log('NO BUG');
    } catch (error) {
      if (error instanceof assert.AssertionError) {
        console.log('OBSERVED:');
        for (const line of normalize(observed, root).split('\n')) {
          console.log(`  ${line}`);
        }
        console.log(`EXPECTED: every approval hint carries sourceRunId=${requestedRunId}`);
        console.log('ACTUAL: "rerun" drops the id (undefined) and "overrun 2026-01-01" is captured as the id');
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
