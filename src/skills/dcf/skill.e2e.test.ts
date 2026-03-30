/**
 * E2E tests — DCF valuation skill with real Ollama model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * The agent is invoked ONCE via beforeAll; all tests share the same result
 * to avoid paying the LLM cost multiple times.
 *
 * Model: ollama:nemotron-3-nano:30b-cloud (override via E2E_MODEL env var)
 * Timeout: E2E_TIMEOUT_MS (default 300 s)
 */
import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { runAgentE2E, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { E2EResult } from '@/utils/e2e-helpers.js';

// Financial data tool names registered in src/tools/registry.ts
const FINANCIAL_TOOL_NAMES = [
  'get_financials',
  'get_market_data',
  'read_filings',
  'wacc_inputs',
  'stock_screener',
  'portfolio_risk',
  'get_earnings_transcript',
  'get_fixed_income',
  'get_options_chain',
];

let result: E2EResult;
let tools: string[];
let answer: string;

describe('DCF skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return; // guard — tests will be skipped via e2eIt
    result = await runAgentE2E('Use the DCF skill to value Apple (AAPL)');
    tools = result.toolsCalled;
    answer = result.answer;
  }, E2E_TIMEOUT_MS);

  e2eIt('invokes skill tool', () => {
    expect(
      tools.some((t) => t === 'skill'),
      `skill tool must be called. Tools called: [${tools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('calls at least one financial data tool', () => {
    const calledFinancial = tools.some(
      (t) =>
        FINANCIAL_TOOL_NAMES.includes(t) ||
        t.includes('financial') ||
        t.includes('market') ||
        t.includes('filing') ||
        t.includes('wacc') ||
        t.includes('screener'),
    );
    expect(
      calledFinancial,
      `at least one financial tool must be called. Tools called: [${tools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('answer contains a DCF valuation figure and methodology', () => {
    expect(answer.toLowerCase()).toMatch(/dcf|discounted cash flow|intrinsic value|valuation/);
    expect(answer).toMatch(/\$[\d,]+(\.\d+)?|\d+\.?\d*/);
    expect(answer.toLowerCase()).toMatch(/apple|aapl/);
  });
});
