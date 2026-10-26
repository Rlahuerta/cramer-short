/**
 * PI-22: named GOLD mutator comparison drops the "gold-" id prefix
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * forecast_lab_run action=compare-best-vs-shipped derives the named mutator id
 * from `query` with /\b(markov-...)/. Every GOLD catalog id is prefixed with
 * "gold-" (mutators/markov-parameters.ts buildGoldCatalog idPrefix), so the
 * extraction yields "markov-..." while the shipped/ledger id is
 * "gold-markov-...". The lookup therefore misses the kept GOLD run and the tool
 * returns a "No kept structured run matched mutationId" error. The sibling
 * extractor in query-router.ts already allows the optional gold- prefix.
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

    const { createForecastLabRunTool } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/forecast-lab-run.ts')).href
    )) as typeof import('../../src/tools/forecast-lab-run.ts');

    const requestedMutationId = 'gold-markov-shorter-reactive-window';
    const activeMutationId = 'gold-markov-longer-stability-window';
    const requestedRunId = 'forecast-lab-gold-markov-short-horizon-2026-01-01T00-00-00-000Z';
    const activeRunId = 'forecast-lab-gold-markov-short-horizon-2025-12-31T00-00-00-000Z';
    const requestedArtifactsPath = '/tmp/c2-gold-artifacts/requested';
    const activeArtifactsPath = '/tmp/c2-gold-artifacts/active';

    const entry = (runId: string, mutationId: string, artifactsPath: string) => ({
      runId,
      startedAt: '2026-01-01T00:00:00.000Z',
      profileId: 'gold-markov-short-horizon',
      targetSubsystem: 'markov-distribution',
      candidateBranch: `topic/forecast-lab-${runId}`,
      allowedGlobs: [],
      mutationMode: 'structured',
      mutationId,
      baselineSummary: { exitCode: 0 },
      candidateSummary: { exitCode: 0 },
      decision: 'keep',
      reason: 'kept',
      artifactsPath,
    });

    const ledgerEntries = [
      entry(activeRunId, activeMutationId, activeArtifactsPath),
      entry(requestedRunId, requestedMutationId, requestedArtifactsPath),
    ];

    const tool = createForecastLabRunTool({
      getLedgerPathFn: () => '/tmp/c2-gold-ledger.tsv',
      readLedgerEntriesFn: () => ledgerEntries as never,
      readRunManifestFn: () => {
        throw new Error('manifest read is not expected for this path');
      },
      existsSyncFn: (path: string) => path.endsWith('active-promotions/gold-markov-short-horizon.json'),
      readTextFileFn: (path: string) => {
        if (path.endsWith('active-promotions/gold-markov-short-horizon.json')) {
          return JSON.stringify({ sourceRunId: activeRunId });
        }
        if (path.endsWith('candidate.json')) {
          return JSON.stringify({ commands: [] });
        }
        throw new Error(`unexpected read: ${path}`);
      },
    });

    const raw = await tool.invoke({
      action: 'compare-best-vs-shipped',
      profileId: 'gold-markov-short-horizon',
      query: `Compare ${requestedMutationId} against the active run.`,
    });
    const payload = JSON.parse(raw) as {
      data: {
        status: string;
        error?: string;
        mutationId?: string;
        activeMutationId?: string;
        sourceRunId?: string;
      };
    };
    const data = payload.data;

    const observed =
      `status=${data.status}; mutationId=${String(data.mutationId)}; ` +
      `activeMutationId=${String(data.activeMutationId)}; ` +
      `sourceRunId=${String(data.sourceRunId)}` +
      (data.error ? `; error=${data.error}` : '');

    try {
      assert.equal(data.status, 'ok', `compare-best-vs-shipped failed: ${data.error ?? 'unknown error'}`);
      assert.equal(data.mutationId, requestedMutationId, 'the named GOLD mutator was not resolved');
      assert.equal(data.activeMutationId, activeMutationId, 'the active GOLD mutator was not resolved');
      console.log('NO BUG');
    } catch (error) {
      if (error instanceof assert.AssertionError) {
        console.log('OBSERVED:');
        for (const line of normalize(observed, root).split('\n')) {
          console.log(`  ${line}`);
        }
        console.log('EXPECTED: status=ok; mutationId=gold-markov-shorter-reactive-window; activeMutationId=gold-markov-longer-stability-window');
        console.log('ACTUAL: status=error because the query extractor strips the "gold-" prefix and looks up markov-shorter-reactive-window');
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
