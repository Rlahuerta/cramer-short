/**
 * E2E tests — basic agent flows with the configured Ollama model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * Model: ollama:minimax-m2.7:cloud (override via E2E_MODEL env var)
 * Timeout: 360 s per test
 */
import { describe, expect } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
import { runAgentE2E, runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { ToolStartEvent, ToolEndEvent } from '@/agent/types.js';

function findToolStartEvent(result: { events: unknown[] }, tool: string): ToolStartEvent | undefined {
  return result.events.find((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_start' && candidate.tool === tool;
  });
}

function findToolEndEvent(result: { events: unknown[] }, tool: string): ToolEndEvent | undefined {
  return result.events.find((event): event is ToolEndEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_end' && candidate.tool === tool;
  });
}

describe('Agent E2E — basic financial query flows', () => {
  e2eIt(
    'looks up AAPL stock price and returns a numeric value',
    async () => {
      const result = await runAgentE2E('What is the current stock price of Apple (AAPL)?');

      // Agent must have called at least one financial tool
      expect(result.toolsCalled.length).toBeGreaterThan(0);
      const calledFinancial = result.toolsCalled.some(
        (t: string) => t.includes('financial') || t.includes('market') || t.includes('price'),
      );
      expect(calledFinancial).toBe(true);

      // Answer must contain a dollar amount or a clear price figure
      expect(result.answer).toMatch(/\$[\d,]+(\.\d+)?|\d+\.\d{2}/);

      // Answer must mention AAPL or Apple
      expect(result.answer.toLowerCase()).toMatch(/aapl|apple/);

      // Should complete in a reasonable time
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'searches for Federal Reserve news and returns a non-trivial answer',
    async () => {
      const result = await runAgentE2E('Find recent news about Federal Reserve interest rate decisions');

      // Agent must have called a search tool
      expect(result.toolsCalled.length).toBeGreaterThan(0);
      const calledSearch = result.toolsCalled.some(
        (t: string) => t.includes('search') || t.includes('web') || t.includes('news'),
      );
      expect(calledSearch).toBe(true);

      // Answer must be substantive (not just an error or placeholder)
      expect(result.answer.length).toBeGreaterThan(200);

      // Answer must mention Federal Reserve or interest rates
      expect(result.answer.toLowerCase()).toMatch(/federal reserve|interest rate|fed|fomc/);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'routes BTC 7-day forecast through full six-tool stack',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry('Provide a BTC forecast for the next 7 days');

      const required = ['get_market_data', 'social_sentiment', 'polymarket_forecast', 'get_onchain_crypto', 'get_fixed_income', 'markov_distribution'];
      for (const tool of required) {
        expect(result.toolsCalled).toContain(tool);
      }

      expect(result.answer.toLowerCase()).toMatch(/btc|bitcoin/);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'routes the open-ended GOLD markov prompt through the commodity proxy path',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(
        '--deep Provide a GOLD forecast based on markov chain for the next 30 days',
        { model: 'ollama:minimax-m2.7:cloud' },
      );

      expect(result.toolsCalled).toContain('markov_distribution');
      const markovStart = findToolStartEvent(result, 'markov_distribution');
      expect(markovStart).toBeDefined();
      expect(markovStart?.args.ticker).toBe('GLD');
      expect(markovStart?.args.horizon).toBe(30);
      expect(result.answer.toLowerCase()).toMatch(/gold|gld/);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'routes the open-ended SILVER markov prompt through the silver proxy path',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(
        '--deep Provide a SILVER forecast based on markov chain for the next 30 days',
        { model: 'ollama:minimax-m2.7:cloud' },
      );

      expect(result.toolsCalled).toContain('markov_distribution');
      const markovStart = findToolStartEvent(result, 'markov_distribution');
      expect(markovStart).toBeDefined();
      expect(markovStart?.args.ticker).toBe('SLV');
      expect(markovStart?.args.horizon).toBe(30);
      expect(result.answer.toLowerCase()).toMatch(/silver|slv/);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'uses deeper fallback tools after a non-crypto Markov abstain path for NVDA forecasts',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(
        '--deep Provide an NVDA forecast based on markov chain for the next 7 days',
        { model: 'ollama:minimax-m2.7:cloud' },
      );

      expect(result.toolsCalled).toContain('markov_distribution');
      expect(result.toolsCalled).toContain('get_market_data');
      expect(result.toolsCalled).toContain('polymarket_forecast');

      const markovStart = findToolStartEvent(result, 'markov_distribution');
      expect(markovStart).toBeDefined();
      expect(markovStart?.args.ticker).toBe('NVDA');
      expect(markovStart?.args.horizon).toBe(7);

      const markovEnd = findToolEndEvent(result, 'markov_distribution');
      expect(markovEnd).toBeDefined();
      const payload = JSON.parse(markovEnd!.result) as { data?: { status?: string } };
      expect(payload?.data?.status).toBe('abstain');

      expect(result.answer.toLowerCase()).toMatch(/nvda|nvidia/);
      const mentionsAbstainLimit = /abstain|no calibrated markov|confidence interval|point estimate/i.test(result.answer);
      const hasPriceFigure = /\$[\d,]+(\.\d+)?|\d+\.\d{2}/.test(result.answer);
      expect(mentionsAbstainLimit || hasPriceFigure).toBe(true);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );
});
