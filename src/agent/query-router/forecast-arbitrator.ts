import type { ToolCallRecord } from '../scratchpad.js';
import {
  type ForcedForecastArbiterArgs,
  type ForcedNonCryptoPolymarketForecastArgs,
  hasErrorLikeToolResult,
  hasNonEmptyParsedToolData,
  matchesTickerAndOptionalHorizon,
} from './types.js';
import {
  inferDistributionTicker,
  inferDistributionHorizon,
  inferBtcShortHorizonForecastHorizon,
} from './distribution.js';
import {
  isCryptoForecastQuery,
  isNonCryptoForecastQuery,
  isExplicitGoldCombinedMarkovPolymarketRequest,
} from './classification.js';
import {
  buildForcedMarketDataArgs,
  buildForcedNonCryptoMarketDataArgs,
  buildForcedNonCryptoPolymarketForecastArgs,
  buildForcedCryptoForecastMarkovArgs,
  buildForcedPolymarketForecastArgs,
  buildForcedSocialSentimentArgs,
  buildForcedOnchainArgs,
  buildForcedFixedIncomeArgs,
} from './forced-tool-args.js';
import {
  hasMarketDataQuery,
  extractCurrentPriceFromMarketDataQuery,
  extractCurrentPriceForCryptoQuery,
  extractCurrentPriceForNonCryptoQuery,
  extractSentimentScoreForCryptoQuery,
  extractMarkovArbiterEvidence,
  extractPolymarketArbiterEvidence,
  extractPolymarketForecastReturnForQuery,
  extractMarkovReturnForQuery,
} from './tool-call-extractors.js';
import {
  hasSuccessfulMarkovDistributionForQuery,
  hasAbstainingMarkovDistributionForQuery,
  hasCompletedMarkovDistributionForQuery,
  hasUsableOnchainResultForCryptoQuery,
  hasPolymarketForecastCoverage,
  hasPolymarketForecastErrorForCoverage,
  shouldRerunPolymarketForecastWithMarkov,
  hasUsableFixedIncomeResult,
} from './coverage.js';
import { detectAssetType } from '../../tools/finance/signal-extractor.js';

function toolArgMatches(actual: unknown, expected: unknown): boolean {
  if (Array.isArray(expected)) {
    return Array.isArray(actual)
      && actual.length === expected.length
      && expected.every((value, index) => toolArgMatches(actual[index], value));
  }

  if (expected !== null && typeof expected === 'object') {
    if (actual === null || typeof actual !== 'object' || Array.isArray(actual)) return false;
    return Object.entries(expected).every(([key, value]) =>
      toolArgMatches(Reflect.get(actual, key), value),
    );
  }

  return Object.is(actual, expected);
}

function hasErrorAttemptForArgs(
  toolCalls: ToolCallRecord[],
  toolName: string,
  expectedArgs: object,
): boolean {
  return toolCalls.some((call) =>
    call.tool === toolName
    && hasErrorLikeToolResult(call.result)
    && Object.entries(expectedArgs).every(([key, value]) => toolArgMatches(call.args[key], value)),
  );
}

/**
 * Returns true when a non-crypto forecast query has had markov_distribution
 * abstain and is still missing get_market_data and/or
 * polymarket_forecast — the two tools that should be forced as a fallback.
 */
export function shouldForceNonCryptoForecastFallback(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isNonCryptoForecastQuery(query)) return false;

  // If Markov already produced a non-abstain result, don't force fallback.
  if (hasSuccessfulMarkovDistributionForQuery(query, toolCalls)) return false;

  // This fallback is only for the post-abstain path. If Markov never ran,
  // let the normal model/tool flow decide whether to invoke it first.
  const hasAbstainingMarkov = hasAbstainingMarkovDistributionForQuery(query, toolCalls);
  if (!hasAbstainingMarkov) return false;

  const marketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  const forecastArgs = buildForcedNonCryptoPolymarketForecastArgs(query, toolCalls);
  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  const marketDataAttempted = marketDataArgs !== null && hasMarketDataQuery(toolCalls, marketDataArgs.query);

  const needsCurrentPriceFetch = marketDataArgs !== null && currentPrice === null && !marketDataAttempted;

  const needsForecastRun = forecastArgs !== null
    && !hasPolymarketForecastCoverage(toolCalls, forecastArgs)
    && !hasPolymarketForecastErrorForCoverage(toolCalls, forecastArgs);

  return needsCurrentPriceFetch || needsForecastRun;
}

export function shouldForceCryptoForecastTools(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;

  const marketDataArgs = buildForcedMarketDataArgs(query);
  const socialSentimentArgs = buildForcedSocialSentimentArgs(query);
  const markovArgs = buildForcedCryptoForecastMarkovArgs(query);
  const polymarketArgs = buildForcedPolymarketForecastArgs(query, toolCalls);
  const onchainArgs = buildForcedOnchainArgs(query);
  const fixedIncomeArgs = buildForcedFixedIncomeArgs();
  const forecastArbiterArgs = buildForcedForecastArbiterArgs(query, toolCalls);
  const hasMarketData = marketDataArgs !== null
    && extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query) !== null;
  const hasSocialSentiment = extractSentimentScoreForCryptoQuery(query, toolCalls) !== null;
  const hasPolymarketForecast = polymarketArgs !== null
    && hasPolymarketForecastCoverage(toolCalls, polymarketArgs);
  const hasOnchain = hasUsableOnchainResultForCryptoQuery(query, toolCalls);
  const hasFixedIncome = hasUsableFixedIncomeResult(toolCalls);
  const needsMarketData = marketDataArgs !== null
    && !hasMarketData
    && !hasErrorAttemptForArgs(toolCalls, 'get_market_data', marketDataArgs);
  const needsSocialSentiment = socialSentimentArgs !== null
    && !hasSocialSentiment
    && !hasErrorAttemptForArgs(toolCalls, 'social_sentiment', socialSentimentArgs);
  const needsMarkov = markovArgs !== null
    && !hasCompletedMarkovDistributionForQuery(query, toolCalls)
    && !hasErrorAttemptForArgs(toolCalls, 'markov_distribution', markovArgs);
  const needsPolymarketForecast = polymarketArgs !== null
    && !hasPolymarketForecast
    && !hasErrorAttemptForArgs(toolCalls, 'polymarket_forecast', polymarketArgs);
  const needsPolymarketRerun = polymarketArgs !== null
    && shouldRerunPolymarketForecastWithMarkov(query, toolCalls)
    && !hasErrorAttemptForArgs(toolCalls, 'polymarket_forecast', polymarketArgs);
  const needsOnchain = onchainArgs !== null
    && !hasOnchain
    && !hasErrorAttemptForArgs(toolCalls, 'get_onchain_crypto', onchainArgs);
  const needsFixedIncome = !hasFixedIncome
    && !hasErrorAttemptForArgs(toolCalls, 'get_fixed_income', fixedIncomeArgs);
  const needsForecastArbiter = forecastArbiterArgs !== null
    && !hasForecastArbitratorForQuery(query, toolCalls)
    && !hasErrorAttemptForArgs(toolCalls, 'forecast_arbitrator', forecastArbiterArgs);

  return needsMarketData
    || needsSocialSentiment
    || needsPolymarketForecast
    || needsOnchain
    || needsFixedIncome
    || needsMarkov
    || needsPolymarketRerun
    || needsForecastArbiter;
}

export function shouldForceGoldCombinedForecastTools(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isExplicitGoldCombinedMarkovPolymarketRequest(query)) return false;
  if (!hasCompletedMarkovDistributionForQuery(query, toolCalls)) return false;

  const marketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  const forecastArgs = buildForcedNonCryptoPolymarketForecastArgs(query, toolCalls);
  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  const marketDataAttempted = marketDataArgs !== null && hasMarketDataQuery(toolCalls, marketDataArgs.query);

  const needsCurrentPriceFetch = marketDataArgs !== null && currentPrice === null && !marketDataAttempted;

  const needsForecastRun = forecastArgs !== null
    && !hasPolymarketForecastCoverage(toolCalls, forecastArgs)
    && !hasPolymarketForecastErrorForCoverage(toolCalls, forecastArgs);

  return needsCurrentPriceFetch || needsForecastRun || shouldForceGoldCombinedForecastArbitrator(query, toolCalls);
}

function inferLeverageFromQuery(query: string): number | null {
  const match = query.match(/\b(\d{1,3}(?:\.\d+)?)\s*x\b/i);
  if (!match) return null;
  const leverage = Number.parseFloat(match[1]!);
  return Number.isFinite(leverage) && leverage > 0 ? leverage : null;
}

function isTradeDecisionQuery(query: string): boolean {
  return /\b(direction|entry|enter|stop|stop-loss|target|take profit|leverage|leveraged|\d{1,3}(?:\.\d+)?\s*x|long|short|trade setup|trade plan|position|arbitrator|verdict)\b/i.test(query);
}

export function hasForecastArbitratorForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);
  return toolCalls.some((call) => {
    if (call.tool !== 'forecast_arbitrator') return false;
    if (hasErrorLikeToolResult(call.result)) return false;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon_days', desiredHorizon)) return false;
    return hasNonEmptyParsedToolData(call);
  });
}

export function detectBtcShortHorizonDisagreement(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  const ticker = inferDistributionTicker(query);
  const horizon = inferDistributionHorizon(query);
  if ((ticker !== 'BTC' && ticker !== 'BTC-USD') || horizon === null || horizon > 14) return false;

  const markovReturn = extractMarkovReturnForQuery(query, toolCalls);
  const polymarketReturn = extractPolymarketForecastReturnForQuery(query, toolCalls);
  if (markovReturn === null || polymarketReturn === null) return false;

  return markovReturn > 0.01 && polymarketReturn <= 0;
}

export function buildForcedForecastArbiterArgs(query: string, toolCalls: ToolCallRecord[]): ForcedForecastArbiterArgs | null {
  if (!isCryptoForecastQuery(query)) return null;
  if (!isTradeDecisionQuery(query) && !detectBtcShortHorizonDisagreement(query, toolCalls)) return null;

  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  const horizon = inferDistributionHorizon(query) ?? inferBtcShortHorizonForecastHorizon(query) ?? 1;
  const markov = extractMarkovArbiterEvidence(query, toolCalls);
  const polymarket = extractPolymarketArbiterEvidence(query, toolCalls);
  if (!markov && !polymarket) return null;

  const args: ForcedForecastArbiterArgs = {
    ticker: detected.ticker,
    horizon_days: horizon,
    whale: {
      direction: 'neutral',
      confidence: 0.35,
      summary: hasUsableOnchainResultForCryptoQuery(query, toolCalls)
        ? 'On-chain/whale tool completed; treat as neutral unless the final synthesis has a stronger confirmed whale signal.'
        : 'No confirmed whale/on-chain signal available.',
    },
  };

  if (markov) args.markov = markov;
  if (polymarket) args.polymarket = polymarket;

  const currentPrice = extractCurrentPriceForCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const leverage = inferLeverageFromQuery(query);
  if (leverage !== null) args.leverage = leverage;

  return args;
}

export function shouldForceForecastArbitrator(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (hasForecastArbitratorForQuery(query, toolCalls)) return false;
  const args = buildForcedForecastArbiterArgs(query, toolCalls);
  return args !== null && !hasErrorAttemptForArgs(toolCalls, 'forecast_arbitrator', args);
}

const GOLD_COMBINED_MATERIAL_DISAGREEMENT_THRESHOLD = 0.01;

function detectGoldCombinedForecastDisagreement(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isExplicitGoldCombinedMarkovPolymarketRequest(query)) return false;

  const markovReturn = extractMarkovReturnForQuery(query, toolCalls);
  const polymarketReturn = extractPolymarketForecastReturnForQuery(query, toolCalls);
  if (markovReturn === null || polymarketReturn === null) return false;

  const oppositeDirections =
    (markovReturn >= 0.001 && polymarketReturn <= -0.001)
    || (markovReturn <= -0.001 && polymarketReturn >= 0.001);

  return oppositeDirections
    && Math.abs(markovReturn - polymarketReturn) >= GOLD_COMBINED_MATERIAL_DISAGREEMENT_THRESHOLD;
}

export function buildForcedGoldCombinedForecastArbiterArgs(
  query: string,
  toolCalls: ToolCallRecord[],
): ForcedForecastArbiterArgs | null {
  const ticker = inferDistributionTicker(query);
  if (!ticker) return null;

  const horizon = inferDistributionHorizon(query) ?? 7;
  const markov = extractMarkovArbiterEvidence(query, toolCalls);
  const polymarket = extractPolymarketArbiterEvidence(query, toolCalls);
  if (!markov || !polymarket) return null;
  const hasDirectionalDisagreement = detectGoldCombinedForecastDisagreement(query, toolCalls);
  const hasStructuralBreakDiagnostics = markov.structural_break === true;
  if (!hasDirectionalDisagreement && !hasStructuralBreakDiagnostics) return null;

  const args: ForcedForecastArbiterArgs = {
    ticker,
    horizon_days: horizon,
    markov,
    polymarket,
  };

  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const leverage = inferLeverageFromQuery(query);
  if (leverage !== null) args.leverage = leverage;

  return args;
}

export function shouldForceGoldCombinedForecastArbitrator(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (hasForecastArbitratorForQuery(query, toolCalls)) return false;
  const args = buildForcedGoldCombinedForecastArbiterArgs(query, toolCalls);
  return args !== null && !hasErrorAttemptForArgs(toolCalls, 'forecast_arbitrator', args);
}
