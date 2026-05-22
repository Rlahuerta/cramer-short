import type { ToolCallRecord } from '../scratchpad.js';
import { detectAssetType } from '../../tools/finance/signal-extractor.js';
import {
  inferDistributionTicker,
  inferDistributionHorizon,
  inferBtcShortHorizonForecastHorizon,
} from './distribution.js';
import {
  isCryptoForecastQuery,
  isNonCryptoForecastQuery,
} from './classification.js';
import type { ForcedNonCryptoPolymarketForecastArgs } from './types.js';
import {
  extractCurrentPriceForCryptoQuery,
  extractCurrentPriceForNonCryptoQuery,
  extractSentimentScoreForCryptoQuery,
  extractMarkovReturnForQuery,
  extractMarkovReturnFromToolCalls,
} from './tool-call-extractors.js';

export {
  buildForcedMarketDataArgs,
  buildForcedNonCryptoMarketDataArgs,
  buildForcedSocialSentimentArgs,
} from './forced-tool-basic-args.js';

export function buildForcedPolymarketForecastArgs(query: string, toolCalls: ToolCallRecord[]): {
  ticker: string;
  horizon_days?: number;
  current_price?: number;
  sentiment_score?: number;
  markov_return?: number;
} | null {
  if (!isCryptoForecastQuery(query)) return null;

  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  const args: {
    ticker: string;
    horizon_days?: number;
    current_price?: number;
    sentiment_score?: number;
    markov_return?: number;
  } = {
    ticker: detected.ticker,
  };

  const horizon = inferDistributionHorizon(query);
  if (horizon) args.horizon_days = horizon;

  const currentPrice = extractCurrentPriceForCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const sentimentScore = extractSentimentScoreForCryptoQuery(query, toolCalls);
  if (sentimentScore !== null) args.sentiment_score = sentimentScore;

  const markovReturn = extractMarkovReturnForQuery(query, toolCalls);
  if (markovReturn !== null) args.markov_return = markovReturn;

  return args;
}

export function buildForcedNonCryptoPolymarketForecastArgs(
  query: string,
  toolCalls: ToolCallRecord[],
): ForcedNonCryptoPolymarketForecastArgs | null {
  if (!isNonCryptoForecastQuery(query)) return null;

  const ticker = inferDistributionTicker(query);
  if (!ticker) return null;

  const args: ForcedNonCryptoPolymarketForecastArgs = {
    ticker,
    horizon_days: inferDistributionHorizon(query) ?? 7,
  };

  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const markovReturn = extractMarkovReturnFromToolCalls(toolCalls);
  if (markovReturn !== null) args.markov_return = markovReturn;

  return args;
}

export function buildForcedOnchainArgs(query: string): { ticker: string; metrics: string[] } | null {
  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  return {
    ticker: detected.ticker,
    metrics: ['market', 'sentiment'],
  };
}

export function buildForcedFixedIncomeArgs(): { series: string[] } {
  return {
    series: ['treasury_yields', 'yield_curve'],
  };
}

export function buildForcedCryptoForecastMarkovArgs(query: string): {
  ticker: string;
  horizon: number;
  trajectory: true;
  trajectoryDays: number;
} | null {
  if (!isCryptoForecastQuery(query)) return null;

  const ticker = inferDistributionTicker(query);
  let horizon = inferDistributionHorizon(query);

  if (!horizon && ticker === 'BTC-USD') {
    horizon = inferBtcShortHorizonForecastHorizon(query);
  }

  if (!ticker || !horizon || horizon > 14) return null;

  return {
    ticker,
    horizon,
    trajectory: true,
    trajectoryDays: Math.min(30, horizon),
  };
}
