import { describe, expect, it } from 'bun:test';
import type { ToolCallRecord } from './scratchpad.js';
import * as queryRouter from './query-router.js';
import {
  extractCurrentPriceFromToolCalls,
  hasPolymarketForecastCoverage,
  hasSuccessfulMarkovDistributionForQuery,
  inferDistributionHorizon,
  inferDistributionHorizon as inferDistributionHorizonFromBarrel,
  shouldForceGoldCombinedForecastTools,
  shouldForceNonCryptoForecastFallback,
  shouldForceMarkovDistribution,
} from './query-router.js';
import { inferDistributionHorizon as inferDistributionHorizonFromModule } from './query-router/distribution.js';

describe('query-router modular barrel', () => {
  it('does not widen the legacy runtime export surface', () => {
    expect(Object.keys(queryRouter).sort()).toEqual([
      'buildForcedCryptoForecastMarkovArgs',
      'buildForcedFixedIncomeArgs',
      'buildForcedForecastArbiterArgs',
      'buildForcedGoldCombinedForecastArbiterArgs',
      'buildForcedMarketDataArgs',
      'buildForcedMarkovArgs',
      'buildForcedNonCryptoMarketDataArgs',
      'buildForcedNonCryptoPolymarketForecastArgs',
      'buildForcedOnchainArgs',
      'buildForcedPolymarketForecastArgs',
      'buildForcedSocialSentimentArgs',
      'detectBtcShortHorizonDisagreement',
      'detectExplicitSkillRequest',
      'extractCurrentPriceFromMarketDataQuery',
      'extractCurrentPriceFromToolCalls',
      'extractMarkovPredictionConfidenceForQuery',
      'extractMarkovReturnFromToolCalls',
      'extractSentimentScoreFromToolCalls',
      'getBtcSelectiveMarkovConfidenceThreshold',
      'hasAbstainingMarkovDistributionForQuery',
      'hasCompletedMarkovDistributionForQuery',
      'hasCryptoPolymarketForecastCoverage',
      'hasForecastArbitratorForQuery',
      'hasLowConfidenceBtcShortHorizonMarkov',
      'hasMarketDataQuery',
      'hasPolymarketForecastCoverage',
      'hasSuccessfulMarkovDistributionForQuery',
      'hasUsableFixedIncomeResult',
      'hasUsableOnchainResultForCryptoQuery',
      'inferBtcShortHorizonForecastHorizon',
      'inferDistributionHorizon',
      'inferDistributionTicker',
      'inferMarkovQueryHorizon',
      'inferTrajectoryRequest',
      'isBtcShortHorizonForecastQuery',
      'isCryptoForecastQuery',
      'isDistributionQuery',
      'isExplicitCombinedMarkovPolymarketRequest',
      'isExplicitGoldCombinedMarkovPolymarketRequest',
      'isExplicitPolymarketForecastRequest',
      'isExplicitTerminalDistributionQuery',
      'isForecastLabImprovementQuery',
      'isForecastLabPlanOnlyQuery',
      'isNonCryptoForecastQuery',
      'shouldForceCryptoForecastTools',
      'shouldForceForecastArbitrator',
      'shouldForceGoldCombinedForecastArbitrator',
      'shouldForceGoldCombinedForecastTools',
      'shouldForceMarkovDistribution',
      'shouldForceNonCryptoForecastFallback',
      'shouldInjectBtcShortHorizonLowConfidencePrompt',
      'shouldInjectBtcShortHorizonMixedEvidencePrompt',
      'shouldRerunPolymarketForecastWithMarkov',
    ].sort());
  });

  it('preserves legacy barrel exports from ./query-router.js', () => {
    expect(inferDistributionHorizonFromBarrel).toBe(inferDistributionHorizonFromModule);
    expect(inferDistributionHorizon('SPY forecast over the next 2 weeks')).toBe(10);
  });

  it('keeps Markov distribution coverage behavior through the compatibility barrel', () => {
    const toolCalls: ToolCallRecord[] = [
      {
        tool: 'markov_distribution',
        args: { ticker: 'BTC-USD', horizon: 3 },
        result: JSON.stringify({
          data: {
            _tool: 'markov_distribution',
            status: 'ok',
          },
        }),
      },
    ];

    const query = 'BTC markov distribution over the next 3 days';

    expect(hasSuccessfulMarkovDistributionForQuery(query, toolCalls)).toBe(true);
    expect(shouldForceMarkovDistribution(query, toolCalls)).toBe(false);
    expect(shouldForceMarkovDistribution(query, [])).toBe(true);
  });

  it('keeps tool-call extraction behavior after splitting modules', () => {
    const toolCalls: ToolCallRecord[] = [
      {
        tool: 'get_market_data',
        args: { query: 'BTC current price' },
        result: JSON.stringify({
          data: {
            get_crypto_price_snapshot_BTC: {
              ticker: 'BTC',
              price: 101_250,
            },
          },
        }),
      },
    ];

    expect(extractCurrentPriceFromToolCalls(toolCalls)).toBe(101_250);
  });

  it('does not re-force a failed non-crypto polymarket_forecast while preserving missing-coverage detection', () => {
    const query = 'Provide an NVDA forecast for the next 7 days';
    const desired = { ticker: 'NVDA', horizon_days: 7, current_price: 921.13 };
    const baseCalls: ToolCallRecord[] = [
      {
        tool: 'markov_distribution',
        args: { ticker: 'NVDA', horizon: 7 },
        result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }),
      },
      {
        tool: 'get_market_data',
        args: { query: 'NVDA current price' },
        result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }),
      },
    ];

    expect(hasPolymarketForecastCoverage(baseCalls, desired)).toBe(false);
    expect(shouldForceNonCryptoForecastFallback(query, baseCalls)).toBe(true);

    const failedCalls: ToolCallRecord[] = [
      ...baseCalls,
      {
        tool: 'polymarket_forecast',
        args: desired,
        result: 'Error: upstream failed',
      },
    ];

    expect(hasPolymarketForecastCoverage(failedCalls, desired)).toBe(false);
    expect(shouldForceNonCryptoForecastFallback(query, failedCalls)).toBe(false);
  });

  it('does not re-force a failed GOLD combined polymarket_forecast', () => {
    const query = 'Provide a GOLD price forecast based on markov chain and polymarket for the next 30 days';
    const desired = { ticker: 'GLD', horizon_days: 30, current_price: 294.87 };
    const baseCalls: ToolCallRecord[] = [
      {
        tool: 'markov_distribution',
        args: { ticker: 'GLD', horizon: 30 },
        result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }),
      },
      {
        tool: 'get_market_data',
        args: { query: 'GLD current price' },
        result: JSON.stringify({ data: { get_stock_price_GLD: { price: 294.87 } } }),
      },
    ];

    expect(shouldForceGoldCombinedForecastTools(query, baseCalls)).toBe(true);
    expect(shouldForceGoldCombinedForecastTools(query, [
      ...baseCalls,
      {
        tool: 'polymarket_forecast',
        args: desired,
        result: 'Error: upstream failed',
      },
    ])).toBe(false);
  });
});
