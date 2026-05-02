/**
 * E2E tests — forecast-lab skill with real Ollama model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * This dry-run prompt proves the skill can be invoked without allowing the
 * agent to mutate source files during the E2E check.
 */
import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { E2EResult } from '@/utils/e2e-helpers.js';

const FORECAST_LAB_QUERY =
  'Use the forecast-lab skill in dry-run explanation mode for a BTC Markov forecast experiment. Do not edit files, run shell commands, or write artifacts; just describe the bounded baseline-first workflow.';

let result: E2EResult;
let tools: string[];
let answer: string;

describe('forecast-lab skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return;
    result = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_QUERY, { maxIterations: 6 });
    tools = result.toolsCalled;
    answer = result.answer;
  }, E2E_TIMEOUT_MS);

  e2eIt('invokes the skill tool', () => {
    expect(
      tools.some((t) => t === 'skill'),
      `skill tool must be called for forecast-lab. Tools: [${tools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('describes the baseline-first candidate comparison', () => {
    const lower = answer.toLowerCase();
    expect(lower, 'answer must mention baseline').toMatch(/baseline/);
    expect(lower, 'answer must mention candidate comparison').toMatch(/candidate/);
    expect(lower, 'answer must mention fixed gates').toMatch(/gate|harness|metric/);
  });

  e2eIt('includes bounded mutation and drop/revert rules', () => {
    const lower = answer.toLowerCase();
    expect(lower, 'answer must mention allowlisted forecast files').toMatch(/allowlist|approved|editable/);
    expect(lower, 'answer must mention dropping or reverting failed candidates').toMatch(/drop|revert|discard/);
  });

  e2eIt('points experiment records at .cramer-short/experiments', () => {
    expect(answer).toMatch(/\.cramer-short\/experiments/);
  });

  e2eIt('does not mutate files in dry-run mode', () => {
    expect(
      tools.some((t) => t === 'write_file' || t === 'edit_file'),
      `dry-run skill invocation must not write or edit files. Tools: [${tools.join(', ')}]`,
    ).toBe(false);
  });
});
