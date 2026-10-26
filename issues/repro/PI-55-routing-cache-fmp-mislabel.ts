/**
 * PI-55: routing cache records FMP for Financial Datasets successes
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * get_financials persists the serving provider as `result.tool.includes('yahoo')
 * ? 'yahoo' : 'fmp'` (get-financials.ts:306). Every non-Yahoo sub-tool is a
 * Financial Datasets tool, so a successful Financial Datasets call writes a
 * bogus `fmp` preference into .cramer-short/api-routing.json for 30 days; the
 * next session injects "FMP free tier works for this ticker" into the router.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mock } from 'bun:test';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');
const TICKER = 'AAPL';

class HarnessFailure extends Error {}

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  throw new HarnessFailure(reason);
}

function normalize(text: string, root: string): string {
  return text
    .replaceAll(root, '<TMP>')
    .replace(/\u001b\[[0-9;]*m/g, '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is unset');
  if (homedir() !== root) harnessFailure(`homedir() ${homedir()} escaped ${root}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() ${tmpdir()} escaped ${root}`);

  const savedCwd = process.cwd();
  const savedEnv: Record<string, string | undefined> = { ...process.env };
  const savedFetch = globalThis.fetch;
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
    // Dummy credential for the stubbed Financial Datasets endpoint only.
    process.env.FINANCIAL_DATASETS_API_KEY = 'dummy-fd-key';

    const fetches: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      fetches.push(url);
      if (url.includes('financialdatasets.ai')) {
        return Response.json({
          income_statements: [
            { report_period: '2024-09-28', revenue: 391035000000, net_income: 93736000000 },
          ],
        });
      }
      throw new Error(`network disabled: ${url}`);
    }) as typeof fetch;

    mock.module(pathToFileURL(join(REPO_ROOT, 'src/model/llm.ts')).href, () => ({
      callLlm: async () => ({
        response: {
          content: '',
          tool_calls: [
            {
              name: 'get_income_statements',
              args: { ticker: TICKER, period: 'annual', limit: 1 },
              id: 'call-1',
              type: 'tool_call',
            },
          ],
        },
      }),
    }));

    const { createGetFinancials } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/get-financials.ts')).href
    );
    const routing = await import(
      pathToFileURL(join(REPO_ROOT, 'src/utils/api-routing-cache.ts')).href
    );

    const tool = createGetFinancials('stub-model');
    const raw = await tool.invoke({ query: `${TICKER} income statement` });
    const parsed = JSON.parse(raw as string) as { data?: Record<string, unknown> };
    const servedByFd = fetches.some((url) => url.includes('financialdatasets.ai'));
    const servedByFmp = fetches.some((url) => url.includes('financialmodelingprep.com'));

    // setPreferredApi is fire-and-forget; let its microtask/persist land.
    await new Promise((resolve) => setTimeout(resolve, 100));
    const preferred = await routing.getPreferredApi(TICKER);

    const emit = (line: string): void => { console.log(normalize(line, root)); };
    const observed = [
      `  served by financialdatasets = ${servedByFd}, served by FMP = ${servedByFmp}`,
      `  result.data keys = ${JSON.stringify(Object.keys(parsed.data ?? {}))}`,
      `  cached preferredApi(${TICKER}) = ${String(preferred)}`,
    ];

    let failure: string | null = null;
    try {
      assert.ok(servedByFd, 'Financial Datasets was not the serving provider');
      assert.ok(!servedByFmp, 'FMP was unexpectedly queried');
      assert.equal(preferred, 'financial-datasets');
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      emit('OBSERVED:');
      for (const line of observed) emit(line);
      emit('EXPECTED:');
      emit(`  cached preferredApi(${TICKER}) = financial-datasets`);
      emit('ACTUAL:');
      emit(`  cached preferredApi(${TICKER}) = ${String(preferred)}`);
      process.exitCode = 42;
      return;
    }
    console.log('NO BUG');
  } finally {
    globalThis.fetch = savedFetch;
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
