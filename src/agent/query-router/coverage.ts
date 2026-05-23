import type { ToolCallRecord } from '../scratchpad.js';
import {
  type ForecastCoverageArgs,
  hasErrorLikeToolResult,
  hasNonEmptyParsedToolData,
  numbersApproximatelyMatch,
  getForecastHorizonArg,
  matchesTickerAndOptionalHorizon,
  parseToolCallData,
  isFiniteNumber,
} from './types.js';
import {
  inferDistributionTicker,
  inferMarkovQueryHorizon,
} from './distribution.js';
import {
  isCryptoForecastQuery,
} from './classification.js';
import {
  buildForcedPolymarketForecastArgs,
  buildForcedCryptoForecastMarkovArgs,
  buildForcedOnchainArgs,
  buildForcedFixedIncomeArgs,
} from './forced-tool-args.js';
import {
  extractMarkovReturnForQuery,
} from './tool-call-extractors.js';

function hasMarkovDistributionStatusForQuery(
  query: string,
  toolCalls: ToolCallRecord[],
  status: 'ok' | 'abstain',
): boolean {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  return toolCalls.some((call) => {
    if (call.tool !== 'markov_distribution') return false;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) return false;

    const data = parseToolCallData(call);
    return data?._tool === 'markov_distribution' && data.status === status;
  });
}

export function hasSuccessfulMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'ok');
}

export function hasAbstainingMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'abstain');
}

export function hasCompletedMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasSuccessfulMarkovDistributionForQuery(query, toolCalls)
    || hasAbstainingMarkovDistributionForQuery(query, toolCalls);
}

export function hasUsableOnchainResultForCryptoQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  const desired = buildForcedOnchainArgs(query);
  if (!desired) return false;

  return toolCalls.some((call) =>
    call.tool === 'get_onchain_crypto'
    && call.args['ticker'] === desired.ticker
    && !hasErrorLikeToolResult(call.result)
    && hasNonEmptyParsedToolData(call),
  );
}

export function hasUsableFixedIncomeResult(toolCalls: ToolCallRecord[]): boolean {
  const desired = buildForcedFixedIncomeArgs();
  return toolCalls.some((call) =>
    call.tool === 'get_fixed_income'
    && JSON.stringify(call.args['series']) === JSON.stringify(desired.series)
    && !hasErrorLikeToolResult(call.result)
    && hasNonEmptyParsedToolData(call),
  );
}

function hasUsableStructuredToolResult(toolCalls: ToolCallRecord[], toolName: string): boolean {
  return toolCalls.some((call) =>
    call.tool === toolName
    && !hasErrorLikeToolResult(call.result)
    && hasNonEmptyParsedToolData(call),
  );
}

function polymarketForecastArgsMatch(call: ToolCallRecord, desired: ForecastCoverageArgs): boolean {
  const existingTicker = typeof call.args['ticker'] === 'string'
    ? call.args['ticker'].toUpperCase()
    : null;
  if (existingTicker !== desired.ticker.toUpperCase()) return false;

  if (desired.horizon_days !== undefined && getForecastHorizonArg(call.args) !== desired.horizon_days) return false;

  if (desired.current_price !== undefined && !numbersApproximatelyMatch(call.args['current_price'], desired.current_price)) {
    return false;
  }

  if (desired.sentiment_score !== undefined && !numbersApproximatelyMatch(call.args['sentiment_score'], desired.sentiment_score)) {
    return false;
  }

  if (desired.markov_return !== undefined && !numbersApproximatelyMatch(call.args['markov_return'], desired.markov_return)) {
    return false;
  }

  return true;
}

export function hasPolymarketForecastCoverage(
  toolCalls: ToolCallRecord[],
  desired: ForecastCoverageArgs,
): boolean {
  return toolCalls.some((call) =>
    call.tool === 'polymarket_forecast'
    && !hasErrorLikeToolResult(call.result)
    && polymarketForecastArgsMatch(call, desired)
  );
}

export function hasPolymarketForecastErrorForCoverage(
  toolCalls: ToolCallRecord[],
  desired: ForecastCoverageArgs,
): boolean {
  return toolCalls.some((call) =>
    call.tool === 'polymarket_forecast'
    && hasErrorLikeToolResult(call.result)
    && polymarketForecastArgsMatch(call, desired)
  );
}

export function hasCryptoPolymarketForecastCoverage(query: string, toolCalls: ToolCallRecord[]): boolean {
  const desired = buildForcedPolymarketForecastArgs(query, toolCalls);
  return desired !== null && hasPolymarketForecastCoverage(toolCalls, desired);
}

function hasPolymarketForecastWithMarkovReturn(toolCalls: ToolCallRecord[]): boolean {
  return toolCalls.some((call) => call.tool === 'polymarket_forecast'
    && isFiniteNumber(call.args['markov_return']));
}

export function shouldRerunPolymarketForecastWithMarkov(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  if (buildForcedCryptoForecastMarkovArgs(query) === null) return false;
  if (!toolCalls.some((call) => call.tool === 'polymarket_forecast')) return false;
  if (extractMarkovReturnForQuery(query, toolCalls) === null) return false;
  return !hasCryptoPolymarketForecastCoverage(query, toolCalls);
}
