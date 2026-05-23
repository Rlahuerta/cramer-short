import { detectAssetType } from '../../tools/finance/signal-extractor.js';
import { inferDistributionTicker } from './distribution.js';
import { isNonCryptoForecastQuery } from './classification.js';

export function buildForcedMarketDataArgs(query: string): { query: string } | null {
  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  return {
    query: `Current crypto price snapshot for ${detected.ticker}`,
  };
}

export function buildForcedNonCryptoMarketDataArgs(query: string): { query: string } | null {
  if (!isNonCryptoForecastQuery(query)) return null;

  const ticker = inferDistributionTicker(query);
  if (!ticker) return null;

  return {
    query: `${ticker} current price`,
  };
}

export function buildForcedSocialSentimentArgs(query: string): {
  ticker: string;
  include_fear_greed: true;
  limit: number;
} | null {
  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  return {
    ticker: detected.ticker,
    include_fear_greed: true,
    limit: 25,
  };
}
