import type { AIMessage } from '@langchain/core/messages';
import type { ForecastLabRoutingHint } from '../experiments/forecast-lab/query-router.js';
import type { ToolCallRecord } from './scratchpad.js';
import { resolveForecastToolTicker } from './query-router/forecast-ticker.js';
import {
  buildForcedCryptoForecastMarkovArgs,
  buildForcedGoldCombinedForecastArbiterArgs,
  buildForcedMarkovArgs,
  buildForcedNonCryptoMarketDataArgs,
  buildForcedNonCryptoPolymarketForecastArgs,
  hasCompletedMarkovDistributionForQuery,
  inferDistributionHorizon,
  inferDistributionTicker,
  isExplicitGoldCombinedMarkovPolymarketRequest,
} from './query-router.js';

export function normalizeExplicitGoldCombinedToolCalls(
  response: AIMessage,
  query: string,
  toolCalls: ToolCallRecord[],
): void {
  if (!isExplicitGoldCombinedMarkovPolymarketRequest(query) || !response.tool_calls?.length) return;

  const forcedMarkovArgs = buildForcedMarkovArgs(query);
  const forcedMarketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  const forcedForecastArgs = buildForcedNonCryptoPolymarketForecastArgs(query, []);
  const forcedArbitratorArgs = buildForcedGoldCombinedForecastArbiterArgs(query, toolCalls);
  const forcedTicker = forcedMarkovArgs?.ticker ?? forcedForecastArgs?.ticker ?? inferDistributionTicker(query);
  const forcedHorizon = forcedMarkovArgs?.horizon ?? forcedForecastArgs?.horizon_days ?? inferDistributionHorizon(query) ?? 7;

  response.tool_calls = response.tool_calls.flatMap((toolCall) => {
    if (toolCall.name === 'social_sentiment' || toolCall.name === 'get_onchain_crypto') {
      return [];
    }

    if (toolCall.name === 'get_market_data' && forcedMarketDataArgs) {
      return [{ ...toolCall, args: forcedMarketDataArgs }];
    }

    if (toolCall.name === 'markov_distribution' && forcedMarkovArgs) {
      const args: Record<string, unknown> = {
        ...(toolCall.args as Record<string, unknown>),
        ...forcedMarkovArgs,
      };
      if (!('trajectory' in forcedMarkovArgs)) {
        delete args['trajectory'];
        delete args['trajectoryDays'];
      }
      return [{ ...toolCall, args }];
    }

    if (toolCall.name === 'polymarket_forecast' && forcedForecastArgs) {
      return [{
        ...toolCall,
        args: {
          ...(toolCall.args as Record<string, unknown>),
          ...forcedForecastArgs,
        },
      }];
    }

    if (toolCall.name === 'forecast_arbitrator') {
      if (forcedArbitratorArgs) {
        return [{ ...toolCall, args: forcedArbitratorArgs }];
      }
      if (!forcedTicker) return [];
      return [{
        ...toolCall,
        args: {
          ...(toolCall.args as Record<string, unknown>),
          ticker: forcedTicker,
          horizon_days: forcedHorizon,
        },
      }];
    }

    return [toolCall];
  });
}

/**
 * Corrects the ticker of the model's own forecast tool calls to the query's
 * intended asset, so a stray symbol (e.g. markov_distribution(ticker="PLAN")
 * on a "BTC-USD only" query) can't fail and trap the agent in a re-emit loop.
 *
 * Covers crypto, equities, and commodities via `inferDistributionTicker`
 * (which already maps GOLD/GLD→GLD, SILVER→SLV, OIL→USO, BTC→BTC-USD, NVDA→NVDA).
 * Per-tool conventions are honored: markov_distribution uses the markov form
 * (hyphenated crypto or ETF proxy) while polymarket_forecast and
 * forecast_arbitrator use the bare root (the crypto root, or the same proxy for
 * commodities/equities).
 *
 * Fires for any single-asset forecast query. Without an explicit "X only"
 * override it stays conservative: ambiguous multi-asset queries (e.g. "compare
 * BTC and ETH") are skipped so a legitimate comparison ticker is never
 * clobbered. Gold-combined requests keep their dedicated normalizer.
 */
export function normalizeForecastToolTickers(response: AIMessage, query: string): void {
  if (!response.tool_calls?.length) return;

  for (const toolCall of response.tool_calls) {
    const canonical = resolveForecastToolTicker(query, toolCall.name);
    if (!canonical) continue;

    const current = typeof toolCall.args?.['ticker'] === 'string'
      ? (toolCall.args['ticker'] as string)
      : '';
    if (current.toUpperCase() === canonical.toUpperCase()) continue;

    toolCall.args = { ...(toolCall.args as Record<string, unknown>), ticker: canonical };
  }
}

export function hasPrematureForecastArbitratorCall(response: AIMessage, query: string, toolCalls: ToolCallRecord[]): boolean {
  const requestedArbiter = response.tool_calls?.some((call) => call.name === 'forecast_arbitrator') ?? false;
  return requestedArbiter
    && buildForcedCryptoForecastMarkovArgs(query) !== null
    && !hasCompletedMarkovDistributionForQuery(query, toolCalls);
}

export function isAcceptedFirstPlanningToolCall(
  response: AIMessage,
  forecastLabRoutingHint?: ForecastLabRoutingHint | null,
  explicitlyRequestedSkill?: string | null,
): boolean {
  const firstToolCall = response.tool_calls?.[0];
  if (!firstToolCall) return true;
  if (firstToolCall.name === 'sequential_thinking') return true;
  if (
    explicitlyRequestedSkill
    && firstToolCall.name === 'skill'
    && firstToolCall.args?.skill === explicitlyRequestedSkill
  ) {
    return true;
  }
  return Boolean(
    forecastLabRoutingHint?.shouldInvokeSkill
      && firstToolCall.name === 'skill'
      && firstToolCall.args?.skill === 'forecast-lab',
  );
}
