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
import type { ToolStartEvent } from '@/agent/types.js';

const FORECAST_LAB_OPTIMIZATION_QUERY =
  'Optimize the BTC 1d/2d/3d Markov forecast workflow in a bounded baseline-first way. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.';

const ORDINARY_BTC_FORECAST_QUERY =
  'Give me a BTC forecast for the next 7 days and explain the main drivers.';

let optimizationResult: E2EResult;
let optimizationTools: string[];
let optimizationAnswer: string;
let ordinaryForecastResult: E2EResult;

function findSkillCall(result: E2EResult, skillName: string): ToolStartEvent | undefined {
  return result.events.find((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    if (event.type !== 'tool_start' || event.tool !== 'skill') return false;
    return event.args?.skill === skillName;
  });
}

describe('forecast-lab skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return;
    optimizationResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_OPTIMIZATION_QUERY, {
      maxIterations: 6,
    });
    optimizationTools = optimizationResult.toolsCalled;
    optimizationAnswer = optimizationResult.answer;
    ordinaryForecastResult = await runAgentE2EWithTimeoutRetry(ORDINARY_BTC_FORECAST_QUERY, {
      maxIterations: 6,
    });
  }, E2E_TIMEOUT_MS);

  e2eIt('invokes the forecast-lab skill for optimization queries', () => {
    expect(
      Boolean(findSkillCall(optimizationResult, 'forecast-lab')),
      `skill(forecast-lab) must be called for optimization routing. Tools: [${optimizationTools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('describes the baseline-first candidate comparison', () => {
    const lower = optimizationAnswer.toLowerCase();
    expect(lower, 'answer must mention baseline').toMatch(/baseline/);
    expect(lower, 'answer must mention candidate comparison').toMatch(/candidate/);
    expect(lower, 'answer must mention fixed gates').toMatch(/gate|harness|metric/);
  });

  e2eIt('includes bounded mutation and drop/revert rules', () => {
    const lower = optimizationAnswer.toLowerCase();
    expect(lower, 'answer must mention allowlisted forecast files').toMatch(/allowlist|approved|editable/);
    expect(lower, 'answer must mention dropping or reverting failed candidates').toMatch(/drop|revert|discard/);
  });

  e2eIt('points experiment records at .cramer-short/experiments', () => {
    expect(optimizationAnswer).toMatch(/\.cramer-short\/experiments/);
  });

  e2eIt('does not mutate files in dry-run mode', () => {
    expect(
      optimizationTools.some((t) => t === 'write_file' || t === 'edit_file'),
      `dry-run skill invocation must not write or edit files. Tools: [${optimizationTools.join(', ')}]`,
    ).toBe(false);
  });

  e2eIt('does not auto-enter forecast-lab for ordinary BTC forecast queries', () => {
    expect(findSkillCall(ordinaryForecastResult, 'forecast-lab')).toBeUndefined();
  });
});
