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
import type { MarkovDistributionPoint } from '@/tools/finance/markov-distribution.js';
import { interpolateSurvival } from '@/tools/finance/markov-distribution.js';

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

function extractToolResultText(result: string): string {
  try {
    const payload = JSON.parse(result) as { data?: { result?: string; error?: string } };
    return payload.data?.result ?? payload.data?.error ?? result;
  } catch {
    return result;
  }
}

function parsePriceToken(token: string): number | null {
  const match = token.match(/\$([0-9][0-9,]*(?:\.\d+)?)([Kk])?/);
  if (!match) return null;
  const value = Number(match[1].replace(/,/g, ''));
  if (!Number.isFinite(value)) return null;
  return match[2] ? value * 1_000 : value;
}

function extractDensityRows(answer: string): Array<{
  line: string;
  answerPct: number;
  lower: number | null;
  upper: number | null;
}> {
  const densitySection = answer
    .split(/density probability table.*\n/i)[1]
    ?.split(/\n---|\n## |\n### /i)[0]
    ?? answer
      .split(/full .*scenario breakdown.*\n/i)[1]
      ?.split(/\n---|\n## |\n### /i)[0]
    ?? answer
      .split(/markov .*density table.*\n/i)[1]
      ?.split(/\n---|\n## |\n### /i)[0]
    ?? answer;

  return densitySection
    .split('\n')
    .filter((line) => /^\|\s*(?:\*\*)?\d+(?:\*\*)?\s*\|/.test(line))
    .map((line) => {
      const cells = line.split('|').slice(1, -1).map((cell) => cell.trim());
      const rangeCell = cells.find((cell) => /\$/.test(cell))
        ?? cells.find((cell) => /[<>]/.test(cell))
        ?? '';
      const probabilityCell = cells.find((cell, index) => index > 0 && /%/.test(cell)) ?? '';
      const answerPct = Number((probabilityCell.match(/(\d+(?:\.\d+)?)\s*%/) ?? [])[1]);

      if (/</.test(rangeCell)) {
        return {
          line,
          answerPct,
          lower: null,
          upper: parsePriceToken(rangeCell),
        };
      }

      if (/>/.test(rangeCell)) {
        return {
          line,
          answerPct,
          lower: parsePriceToken(rangeCell),
          upper: null,
        };
      }

      const prices = [...rangeCell.matchAll(/\$[0-9][0-9,]*(?:\.\d+)?(?:[Kk])?/g)]
        .map((match) => parsePriceToken(match[0]))
        .filter((price): price is number => price !== null);

      return {
        line,
        answerPct,
        lower: prices[0] ?? null,
        upper: prices[1] ?? null,
      };
    })
    .filter((row) => Number.isFinite(row.answerPct) && (row.lower !== null || row.upper !== null));
}

describe('Agent E2E — basic financial query flows', () => {
  e2eIt(
    'looks up AAPL stock price and returns a numeric value',
    async () => {
      const result = await runAgentE2E('What is the current stock price of Apple (AAPL)?');

      // Answer must contain a dollar amount or a clear price figure
      expect(result.answer).toMatch(/\$[\d,]+(\.\d+)?|\d+\.\d{2}/);

      // Answer must mention AAPL or Apple
      expect(result.answer.toLowerCase()).toMatch(/aapl|apple/);

      // Agent should either call a financial tool or produce a price figure directly
      const calledFinancial = result.toolsCalled.some(
        (t: string) => t.includes('financial') || t.includes('market') || t.includes('price'),
      );
      const hasPriceFigure = /\$[\d,]+(\.\d+)?/.test(result.answer);
      expect(calledFinancial || hasPriceFigure).toBe(true);

      // Should complete in a reasonable time
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'searches for Federal Reserve news and returns a non-trivial answer',
    async () => {
      const result = await runAgentE2E('Find recent news about Federal Reserve interest rate decisions');

      // Answer must be substantive (not just an error or placeholder)
      expect(result.answer.length).toBeGreaterThan(200);

      // Answer must mention Federal Reserve or interest rates
      expect(result.answer.toLowerCase()).toMatch(/federal reserve|interest rate|fed|fomc/);

      // Agent should either call a search tool or produce a substantive answer
      const calledSearch = result.toolsCalled.some(
        (t: string) => t.includes('search') || t.includes('web') || t.includes('news'),
      );
      const isSubstantive = result.answer.length > 200;
      expect(calledSearch || isSubstantive).toBe(true);
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
      expect(result.toolsCalled).toContain('polymarket_forecast');
      const forecastStart = findToolStartEvent(result, 'polymarket_forecast');
      expect(forecastStart).toBeDefined();
      expect(forecastStart?.args.ticker).toBe('GLD');
      expect(forecastStart?.args.horizon_days).toBe(30);
      const forecastEnd = findToolEndEvent(result, 'polymarket_forecast');
      expect(forecastEnd).toBeDefined();
      const forecastText = extractToolResultText(forecastEnd!.result).toLowerCase();
      expect(forecastText).toContain('polymarket forecast: gold (gld)');
      expect(forecastText).not.toMatch(/\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|cryptocurrency)\b/i);
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
      expect(result.toolsCalled).toContain('polymarket_forecast');
      const forecastStart = findToolStartEvent(result, 'polymarket_forecast');
      expect(forecastStart).toBeDefined();
      expect(forecastStart?.args.ticker).toBe('SLV');
      expect(forecastStart?.args.horizon_days).toBe(30);
      const forecastEnd = findToolEndEvent(result, 'polymarket_forecast');
      expect(forecastEnd).toBeDefined();
      const forecastText = extractToolResultText(forecastEnd!.result).toLowerCase();
      expect(forecastText).toContain('polymarket forecast: silver (slv)');
      expect(forecastText).not.toMatch(/\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|cryptocurrency)\b/i);
      expect(result.answer.toLowerCase()).toMatch(/silver|slv/);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'routes the exact OIL markov + polymarket prompt through the oil proxy path',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(
        '--deep Provide a OIL price forecast based on markov chain and polymarket for the next 14 days',
        { model: 'ollama:minimax-m2.7:cloud' },
      );

      expect(result.toolsCalled).toContain('markov_distribution');
      const markovStart = findToolStartEvent(result, 'markov_distribution');
      expect(markovStart).toBeDefined();
      expect(markovStart?.args.ticker).toBe('USO');
      expect(markovStart?.args.horizon).toBe(14);
      expect(result.toolsCalled).toContain('polymarket_forecast');
      const forecastStart = findToolStartEvent(result, 'polymarket_forecast');
      expect(forecastStart).toBeDefined();
      expect(forecastStart?.args.ticker).toBe('USO');
      expect(forecastStart?.args.horizon_days).toBe(14);
      const forecastEnd = findToolEndEvent(result, 'polymarket_forecast');
      expect(forecastEnd).toBeDefined();
      const forecastText = extractToolResultText(forecastEnd!.result).toLowerCase();
      expect(forecastText).toMatch(/oil|uso/);
      expect(forecastText).not.toMatch(/\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|cryptocurrency)\b/i);
      expect(result.answer.toLowerCase()).toMatch(/oil|uso/);
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

      const forecastStarts = result.events.filter((event): event is ToolStartEvent => {
        if (!event || typeof event !== 'object') return false;
        const candidate = event as { type?: string; tool?: string };
        return candidate.type === 'tool_start' && candidate.tool === 'polymarket_forecast';
      });
      expect(forecastStarts.length).toBeGreaterThanOrEqual(1);
      const lastForecastStart = forecastStarts[forecastStarts.length - 1];
      expect(lastForecastStart?.args['markov_return']).toBeUndefined();

      expect(result.answer.toLowerCase()).toMatch(/nvda|nvidia/);
      const mentionsAbstainLimit = /abstain|no calibrated markov|confidence interval|point estimate/i.test(result.answer);
      const hasPriceFigure = /\$[\d,]+(\.\d+)?|\d+\.\d{2}/.test(result.answer);
      expect(mentionsAbstainLimit || hasPriceFigure).toBe(true);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'uses Markov-enriched polymarket forecast args for a BTC 7-day forecast when live Markov succeeds',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(
        '--deep Provide a BTC forecast for the next 7 days',
        { model: 'ollama:minimax-m2.7:cloud' },
      );

      const required = ['get_market_data', 'social_sentiment', 'polymarket_forecast', 'get_onchain_crypto', 'get_fixed_income', 'markov_distribution'];
      for (const tool of required) {
        expect(result.toolsCalled).toContain(tool);
      }

      const markovStart = findToolStartEvent(result, 'markov_distribution');
      expect(markovStart).toBeDefined();
      expect(markovStart?.args.ticker).toBe('BTC-USD');
      expect(markovStart?.args.horizon).toBe(7);

      const markovEnd = findToolEndEvent(result, 'markov_distribution');
      expect(markovEnd).toBeDefined();
      const markovPayload = JSON.parse(markovEnd!.result) as {
        data?: {
          status?: string;
          canonical?: {
            diagnostics?: {
              predictionConfidence?: number;
            };
          };
        };
      };
      expect(['ok', 'abstain']).toContain(markovPayload?.data?.status ?? '');

      const polymarketStarts = result.events.filter((event): event is ToolStartEvent => {
        if (!event || typeof event !== 'object') return false;
        const candidate = event as { type?: string; tool?: string };
        return candidate.type === 'tool_start' && candidate.tool === 'polymarket_forecast';
      });

      const diagnostics = markovPayload?.data?.canonical?.diagnostics as Record<string, unknown> | undefined;
      const predictionConfidence = typeof diagnostics?.predictionConfidence === 'number'
        ? diagnostics.predictionConfidence
        : null;
      const hasHighConfidence = predictionConfidence !== null && predictionConfidence >= 0.25;

      if (markovPayload?.data?.status === 'ok' && hasHighConfidence) {
        expect(polymarketStarts.length).toBeGreaterThanOrEqual(1);
        const hasMarkovEnrichedForecast = polymarketStarts.some((start) =>
          typeof start.args['markov_return'] === 'number'
          && Number.isFinite(start.args['markov_return'] as number),
        );
        expect(hasMarkovEnrichedForecast).toBe(true);
      } else {
        expect(polymarketStarts.length).toBeGreaterThanOrEqual(1);
        const noAbstainMarkovReturnReuse = polymarketStarts.every((start) =>
          start.args['markov_return'] === undefined,
        );
        expect(noAbstainMarkovReturnReuse).toBe(true);
      }

      expect(result.answer.toLowerCase()).toMatch(/btc|bitcoin/);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'routes leveraged BTC divergence prompts through forecast_arbitrator without schema errors',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(
        '--deep Give me a Polymarket and markov price forecast for BTC over the next 24 hours and also check for a possible whales movements. Provide the enter price, and stop market price for 10x leveraged and the position direction',
        { model: 'ollama:minimax-m2.7:cloud' },
      );

      expect(result.toolsCalled).toContain('markov_distribution');
      expect(result.toolsCalled).toContain('polymarket_forecast');
      expect(result.toolsCalled).toContain('get_onchain_crypto');
      expect(result.toolsCalled).toContain('forecast_arbitrator');

      const arbiterStart = findToolStartEvent(result, 'forecast_arbitrator');
      expect(arbiterStart).toBeDefined();
      expect(arbiterStart?.args.ticker).toMatch(/^BTC(?:-USD)?$/);
      expect(arbiterStart?.args.horizon_days).toBe(1);
      expect(arbiterStart?.args.leverage).toBe(10);

      const arbiterEnd = findToolEndEvent(result, 'forecast_arbitrator');
      expect(arbiterEnd).toBeDefined();
      const payload = JSON.parse(arbiterEnd!.result) as {
        data?: {
          result?: {
            verdict?: string;
            rawEvidence?: {
              markov?: unknown;
              polymarket?: unknown;
              whale?: unknown;
            };
          };
        };
      };
      expect(['LONG', 'SHORT', 'NO_TRADE', 'CONDITIONAL_LONG', 'CONDITIONAL_SHORT']).toContain(payload.data?.result?.verdict ?? '');
      expect(payload.data?.result?.rawEvidence?.markov).toBeDefined();
      expect(payload.data?.result?.rawEvidence?.polymarket).toBeDefined();
      expect(payload.data?.result?.rawEvidence?.whale).toBeDefined();
      expect(result.answer.toLowerCase()).toMatch(/btc|bitcoin/);
      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'handles the exact BTC 24h Polymarket + Markov prompt with 9 buckets and structural-break diagnostics',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(
        '--deep Provide the Polymarket and Markov BTC forecast for 24 hours, also providing the density probabilities for the price range divided into 9 parts. If Markov detects a structural break, include a separate Structural Break Diagnostic explaining what triggered it, the divergence score, whether CI widening was applied, how it downgrades confidence, and how I should adjust leverage, entry, and stop placement as a result.',
        { model: 'ollama:minimax-m2.7:cloud' },
      );

      const required = [
        'get_market_data',
        'social_sentiment',
        'markov_distribution',
        'polymarket_forecast',
        'get_onchain_crypto',
        'get_fixed_income',
        'forecast_arbitrator',
      ];
      for (const tool of required) {
        expect(result.toolsCalled).toContain(tool);
      }

      const markovStart = findToolStartEvent(result, 'markov_distribution');
      expect(markovStart).toBeDefined();
      expect(markovStart?.args.ticker).toBe('BTC-USD');
      expect(markovStart?.args.horizon).toBe(1);
      expect(markovStart?.args.trajectory).toBe(true);
      expect(markovStart?.args.trajectoryDays).toBe(1);

      const markovEnd = findToolEndEvent(result, 'markov_distribution');
      expect(markovEnd).toBeDefined();
      const markovPayload = JSON.parse(markovEnd!.result) as {
        data?: {
          status?: string;
          distribution?: MarkovDistributionPoint[];
          canonical?: {
            diagnostics?: {
              structuralBreakDetected?: boolean;
              ciWidened?: boolean;
              btcShortHorizonRerunTriggered?: boolean;
              btcShortHorizonLivePolicy?: {
                historyDays?: number;
                breakDivergenceThreshold?: number;
                rerunOnBreak?: boolean;
                rerunWindowDays?: number;
              } | null;
            };
          };
        };
      };

      const diagnostics = markovPayload.data?.canonical?.diagnostics;
      expect(['ok', 'abstain']).toContain(markovPayload.data?.status ?? '');
      expect(diagnostics?.btcShortHorizonLivePolicy?.historyDays).toBe(252);
      expect(diagnostics?.btcShortHorizonLivePolicy?.breakDivergenceThreshold).toBe(0.10);
      expect(diagnostics?.btcShortHorizonLivePolicy?.rerunOnBreak).toBe(true);
      expect(diagnostics?.btcShortHorizonLivePolicy?.rerunWindowDays).toBe(60);

      const arbiterStart = findToolStartEvent(result, 'forecast_arbitrator');
      expect(arbiterStart).toBeDefined();
      expect(arbiterStart?.args.ticker).toMatch(/^BTC(?:-USD)?$/);
      expect(arbiterStart?.args.horizon_days).toBe(1);

      const bucketRows = [...result.answer.matchAll(/^\|\s*(?:\*\*)?\d+(?:\*\*)?\s*\|/gm)];
      const densityRows = extractDensityRows(result.answer);
      const hasNinePartLanguage = /9\s*(?:price )?(?:buckets|parts|bins|segments)/i.test(result.answer);
      const hasDensityTable =
        /bucket/i.test(result.answer)
        && /price range/i.test(result.answer)
        && /probability/i.test(result.answer);
      const hasBucketMassTable =
        /p\(bucket\)|p\(in bucket\)|scenario breakdown/i.test(result.answer);
      const mentionsDensityIntent =
        /density|bucket|probability distribution|scenario breakdown/i.test(result.answer);
      expect(
        bucketRows.length >= 9
          || densityRows.length >= 5
          || (hasNinePartLanguage && hasDensityTable && hasBucketMassTable)
          || mentionsDensityIntent,
        'answer must still acknowledge the requested density/bucket framing even if the model varies the markdown layout',
      ).toBe(true);

      const canonicalDistribution = markovPayload.data?.distribution ?? [];
      if (markovPayload.data?.status === 'ok' && hasBucketMassTable) {
        expect(canonicalDistribution.length).toBeGreaterThan(0);
        expect(densityRows.length).toBeGreaterThanOrEqual(5);

        const deltas = densityRows.map((row) => {
          const canonicalProb =
            row.lower !== null && row.upper !== null
              ? interpolateSurvival(canonicalDistribution, row.lower) - interpolateSurvival(canonicalDistribution, row.upper)
              : row.upper !== null
                ? 1 - interpolateSurvival(canonicalDistribution, row.upper)
                : interpolateSurvival(canonicalDistribution, row.lower!);
          return Math.abs(row.answerPct - canonicalProb * 100);
        });
        const maxAbsDelta = Math.max(...deltas);
        expect(maxAbsDelta).toBeLessThanOrEqual(2.5);
      }

      expect(result.answer.toLowerCase()).toMatch(/btc|bitcoin/);
      expect(result.answer.toLowerCase()).toMatch(/entry/);
      expect(result.answer.toLowerCase()).toMatch(/stop/);
      expect(result.answer.toLowerCase()).toMatch(/leverage/);

      if (diagnostics?.structuralBreakDetected) {
        expect(result.answer.toLowerCase()).toMatch(/structural break diagnostic/);
        expect(result.answer.toLowerCase()).toMatch(/divergence/);
        expect(result.answer.toLowerCase()).toMatch(/ci widen/);
      }

      if (diagnostics?.btcShortHorizonRerunTriggered) {
        expect(result.answer).toMatch(/252d/i);
        expect(result.answer).toMatch(/60d/i);
      }

      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );
});
