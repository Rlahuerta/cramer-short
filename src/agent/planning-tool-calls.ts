import type { AIMessage } from '@langchain/core/messages';
import type { ForecastLabRoutingHint } from '../experiments/forecast-lab/query-router.js';
import type { ToolCallRecord } from './scratchpad.js';
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
