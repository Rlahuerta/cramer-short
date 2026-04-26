/**
 * E2E tests — portfolio_risk skill with real Ollama model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * Uses explicit tickers (AAPL, MSFT, GOOGL) to avoid watchlist dependence.
 */
import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { E2EResult } from '@/utils/e2e-helpers.js';

const PORTFOLIO_RISK_QUERY =
  'Use the portfolio_risk skill to analyse risk for AAPL, MSFT, GOOGL — VaR, Sharpe, correlation, max drawdown';

let result: E2EResult;
let tools: string[];
let answer: string;

describe('portfolio_risk skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return;
    result = await runAgentE2EWithTimeoutRetry(PORTFOLIO_RISK_QUERY);
    tools = result.toolsCalled;
    answer = result.answer;
  }, E2E_TIMEOUT_MS);

  e2eIt('invokes the skill tool or calls portfolio_risk directly', () => {
    const usedSkillTool = tools.some((t) => t === 'skill');
    const usedPortfolioRiskDirectly = tools.some((t) => t === 'portfolio_risk');
    expect(
      usedSkillTool || usedPortfolioRiskDirectly,
      `skill tool or portfolio_risk must be called. Tools: [${tools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('calls at least one financial data tool', () => {
    const financialTools = [
      'get_financials', 'get_market_data', 'portfolio_risk',
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
    expect(lower, 'AAPL must appear in answer').toMatch(/aapl/);
    expect(lower, 'MSFT must appear in answer').toMatch(/msft/);
    expect(lower, 'GOOGL must appear in answer').toMatch(/googl/);
  });

  e2eIt('answer contains key risk concepts (VaR, Sharpe, drawdown, or correlation)', () => {
    const hasVaR = /var|value at risk|cvar/i.test(answer);
    const hasSharpe = /sharpe/i.test(answer);
    const hasDrawdown = /drawdown|max drawdown|draw\s*down/i.test(answer);
    const hasCorrelation = /correl/i.test(answer);
    expect(
      hasVaR && hasSharpe && hasDrawdown && hasCorrelation,
      'answer must contain VaR, Sharpe, max drawdown, and correlation concepts',
    ).toBe(true);
  });

  e2eIt('answer provides actionable guidance or risk assessment', () => {
    const lower = answer.toLowerCase();
    const hasActionable =
      /recommend|suggest|consider|reduce|hedge|diversi|position|concentrat|risk/i.test(lower);
    const hasAssessment =
      /low risk|moderate risk|high risk|portfolio.*(risk|level|score)/i.test(lower);
    expect(
      hasActionable || hasAssessment,
      'answer must include actionable recommendations or a risk-level assessment',
    ).toBe(true);
  });
});
