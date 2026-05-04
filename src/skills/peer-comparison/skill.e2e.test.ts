/**
 * E2E tests — peer-comparison skill with real Ollama model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * The agent is invoked ONCE via beforeAll; all tests share the same result
 * to avoid paying the LLM cost multiple times.
 *
 * Uses explicit multi-ticker query (NVDA vs AMD, INTC, QCOM) to avoid
 * auto-discovery dependency on stock_screener results.
 */
import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { E2EResult } from '@/utils/e2e-helpers.js';

const PEER_COMPARISON_QUERY =
  'Use the peer-comparison skill to compare NVDA against AMD, INTC, and QCOM on valuation, growth, and quality metrics';
const PEER_COMPARISON_TIMEOUT_MS = Math.max(E2E_TIMEOUT_MS, 600_000);

let result: E2EResult;
let tools: string[];
let answer: string;

describe('peer-comparison skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return;
    result = await runAgentE2EWithTimeoutRetry(PEER_COMPARISON_QUERY);
    tools = result.toolsCalled;
    answer = result.answer;
  }, PEER_COMPARISON_TIMEOUT_MS);

  e2eIt('invokes the skill tool or executes the peer-comparison workflow directly', () => {
    const usedSkillTool = tools.some((t) => t === 'skill');
    expect(
      usedSkillTool,
      `skill tool must be called for peer-comparison skill quality validation. Tools: [${tools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('calls at least one financial data tool', () => {
    const financialTools = [
      'get_financials', 'get_market_data', 'stock_screener',
      'get_robinhood_quote', 'get_robinhood_fundamentals',
    ];
    const calledFinancial = tools.some((t) => financialTools.includes(t));
    expect(
      calledFinancial,
      `at least one financial tool must be called. Tools: [${tools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('all target tickers appear in the answer', () => {
    const lower = answer.toLowerCase();
    expect(lower, 'NVDA must appear in answer').toMatch(/nvda/);
    expect(lower, 'AMD must appear in answer').toMatch(/amd/);
    expect(lower, 'INTC must appear in answer').toMatch(/intc/);
    expect(lower, 'QCOM must appear in answer').toMatch(/qcom/);
  });

  e2eIt('answer contains a comparison table or structured comparison', () => {
    const hasTable = /\|.*\|.*\|/.test(answer) || answer.includes('vs.');
    const hasValuation = /p\/e|ev\/ebitda|peg|premium|discount/i.test(answer);
    const hasGrowth = /revenue growth|growth/i.test(answer);
    const hasQuality = /roic|gross margin|margin|quality/i.test(answer);
    expect(
      hasTable,
      'answer must contain a comparison table or other structured comparison output',
    ).toBe(true);
    expect(
      hasValuation && hasGrowth && hasQuality,
      'answer must cover valuation, growth, and quality dimensions',
    ).toBe(true);
  });

  e2eIt('answer includes a verdict or relative assessment', () => {
    const lower = answer.toLowerCase();
    const hasVerdict =
      /verdict|conclusion|overall|relative|attractive|fairly valued|expensive|cheap|premium|discount/i.test(lower);
    expect(hasVerdict, 'answer must include a verdict or relative assessment paragraph').toBe(true);
  });
});
