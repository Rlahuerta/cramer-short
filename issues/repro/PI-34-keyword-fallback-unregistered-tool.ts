/**
 * PI-34: get_financials keyword fallback routes to an unregistered tool
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * When the router LLM returns no tool calls twice, get_financials falls back to
 * deterministic keyword routing. The ratio/default branch emits the tool name
 * `get_financial_metrics_snapshot`, but FINANCE_TOOLS registers that capability
 * as `get_key_ratios` — no tool in the map has the fallback name, so the
 * fallback always fails with "Tool ... not found" for those queries.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mock } from 'bun:test';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');
const QUERY = 'AAPL valuation metrics';

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
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      throw new Error(`network disabled: ${url}`);
    }) as typeof fetch;

    // Router model returns no tool calls on both attempts, forcing the keyword fallback.
    mock.module(pathToFileURL(join(REPO_ROOT, 'src/model/llm.ts')).href, () => ({
      callLlm: async () => ({ response: { content: '', tool_calls: [] } }),
    }));

    const { createGetFinancials } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/get-financials.ts')).href
    );
    const tool = createGetFinancials('stub-model');
    const raw = await tool.invoke({ query: QUERY });
    const parsed = JSON.parse(raw as string) as {
      data?: Record<string, unknown> & { _errors?: Array<{ tool: string; error: string }> };
    };
    const errors = parsed.data?._errors ?? [];
    const emittedTool = errors[0]?.tool ?? null;
    const hasRatioPayload = Boolean(parsed.data && 'get_key_ratios_AAPL' in parsed.data);

    const emit = (line: string): void => { console.log(normalize(line, root)); };
    const observed = [
      `  query = ${JSON.stringify(QUERY)}`,
      `  result.data keys = ${JSON.stringify(Object.keys(parsed.data ?? {}))}`,
      `  result.data._errors = ${JSON.stringify(errors)}`,
    ];

    let failure: string | null = null;
    try {
      assert.equal(errors.length, 0, `fallback produced errors: ${JSON.stringify(errors)}`);
      assert.ok(hasRatioPayload, 'fallback did not produce a get_key_ratios payload');
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      emit('OBSERVED:');
      for (const line of observed) emit(line);
      emit('EXPECTED:');
      emit('  fallback emits get_key_ratios_AAPL with no _errors');
      emit('ACTUAL:');
      emit(`  fallback emits ${emittedTool ?? 'nothing'} and reports "Tool '${emittedTool ?? ''}' not found"`);
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
