/**
 * Integration tests — direct tool chain verification with real external APIs.
 * No LLM involved; tests the data layer independently.
 *
 * Run with:  RUN_INTEGRATION=1 bun test --filter integration
 * Skipped in normal `bun test` / CI runs.
 */
import { afterEach, describe, expect } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { polymarketTool } from './polymarket.js';
import { socialSentimentTool } from './social-sentiment.js';

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('Financial tools integration — Polymarket', () => {
  integrationIt(
    'Polymarket search returns structured text for a crypto query',
    async () => {
      const result = await polymarketTool.invoke({ query: 'Bitcoin price', limit: 5 });
      const text = typeof result === 'string' ? result : (result as { data: { result: string } }).data.result;

      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(10);

      // Should contain either market data (probabilities with %) or a clear
      // empty/error message.  Polymarket markets use varying outcome labels
      // (Yes/No, Above/Below, Trump/Biden, etc.) so we check for % signs,
      // which appear in every probability line, rather than a specific label.
      const hasContent = text.includes('%') || text.includes('No active Polymarket') || text.includes('polymarket.com') || text.includes('search failed');
      expect(hasContent).toBe(true);
    },
    20_000,
  );

  integrationIt(
    'Polymarket search returns markets for a macro query',
    async () => {
      const result = await polymarketTool.invoke({ query: 'US recession 2026', limit: 3 });
      const text = typeof result === 'string' ? result : (result as { data: { result: string } }).data.result;

      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(10);
    },
    20_000,
  );
});

describe('Financial tools integration — Social Sentiment', () => {
  integrationIt(
    'social sentiment returns sentiment fields for BTC',
    async () => {
      const result = await socialSentimentTool.invoke({
        ticker: 'BTC',
        include_fear_greed: true,
        limit: 5,
      });
      const parsed = typeof result === 'string' ? JSON.parse(result) as { data?: { result?: string } } : result as { data?: { result?: string } };
      const text = parsed.data?.result ?? '';

      expect(text.length).toBeGreaterThan(20);

      const normalized = text.toLowerCase();
      const hasStableSentimentPayload =
        normalized.includes('social sentiment:')
        || normalized.includes('no social media posts found')
        || normalized.includes('fear & greed');
      expect(hasStableSentimentPayload).toBe(true);
    },
    30_000,
  );
});
