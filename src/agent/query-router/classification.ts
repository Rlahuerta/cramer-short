import { routeForecastLabQuery } from '../../experiments/forecast-lab/router.js';
import { detectAssetType } from '../../tools/finance/signal-extractor.js';
import { discoverSkills } from '../../skills/registry.js';
import { resolveAssetIntent } from '../../tools/finance/asset-resolver.js';
import { extractTickers as extractTickersFn } from '../../memory/ticker-extractor.js';

const FORECAST_LAB_PLAN_ONLY_PATTERNS = [
  /\bdo not edit files\b/i,
  /\bdon't edit files\b/i,
  /\bdo not run shell commands\b/i,
  /\bdon't run shell commands\b/i,
  /\bdo not write artifacts\b/i,
  /\bdon't write artifacts\b/i,
  /\bexplain the exact experiment plan\b/i,
  /\bexplain (?:the )?plan\b/i,
  /\bwhat plan would you follow\b/i,
  /\bwhat would you do\b/i,
] as const;

export function isForecastLabImprovementQuery(query: string): boolean {
  return routeForecastLabQuery(query).intent === 'improvement';
}

export function isForecastLabPlanOnlyQuery(query: string): boolean {
  return FORECAST_LAB_PLAN_ONLY_PATTERNS.some((pattern) => pattern.test(query));
}

export function isDistributionQuery(query: string): boolean {
  return /probability distribution|price distribution|full distribution|distribution for .*price/i.test(query);
}

export function isExplicitTerminalDistributionQuery(query: string): boolean {
  const lower = query.toLowerCase();
  return isDistributionQuery(query)
    || lower.includes('markov distribution')
    || lower.includes('markov chain')
    || lower.includes('terminal threshold');
}

/**
 * Detect whether the user explicitly requested a day-by-day trajectory
 * (as opposed to a single-horizon snapshot).
 */
export function inferTrajectoryRequest(query: string): boolean {
  const lower = query.toLowerCase();
  return /trajectory|day[- ]by[- ]day|day by day|price path|daily forecast|daily projection/i.test(lower);
}

export function isCryptoForecastQuery(query: string): boolean {
  if (isForecastLabImprovementQuery(query)) return false;
  if (isExplicitTerminalDistributionQuery(query)) return false;

  if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;

  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return false;

  const lower = query.toLowerCase();
  const hasForecastLanguage = /\bforecast\b|\bpredict(?:ion)?\b|\boutlook\b|\bprice target\b|where .* headed|what will .* trade|how .* move/.test(lower);
  const hasFutureHorizon = /over the next\s+\d+\s*(?:day|days|week|weeks)\b|next\s+\d+\s*(?:day|days|week|weeks)\b|in\s+\d+\s*(?:day|days|week|weeks)\b/.test(lower);

  return hasForecastLanguage || hasFutureHorizon;
}

/**
 * Detect non-crypto asset forecast-like queries (stocks, ETFs, commodities).
 * Returns true when the query asks about future price/forecast/outlook for a
 * non-crypto asset — exactly the cases where Markov abstention should trigger
 * a forced get_market_data + polymarket_forecast fallback.
 */
export function isNonCryptoForecastQuery(query: string): boolean {
  if (isForecastLabImprovementQuery(query)) return false;
  if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;

  // Exclude crypto — that path has its own dedicated forcing
  const detected = detectAssetType(query);
  if (detected.type === 'crypto') return false;
  if (!detected.ticker) return false;

  const lower = query.toLowerCase();
  const isHardDistributionQuery = isDistributionQuery(query)
    || lower.includes('markov distribution')
    || lower.includes('terminal threshold');
  if (isHardDistributionQuery) return false;

  const hasForecastLanguage = /\bforecast\b|\bpredict(?:ion)?\b|\boutlook\b|\bprice target\b|where .* headed|what will .* trade|how .* move|price of|will .* (?:beat|hit|reach|drop|rise|fall|exceed|go above|go below)/.test(lower);
  const hasFutureHorizon = /over the next\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|next\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|in\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|\bnext\s+month\b|\bnext\s+quarter\b|\b(?:by\s+)?end\s+of\s+q[1-4](?:\s+\d{4})?\b|\bthrough\s+q[1-4](?:\s+\d{4})?\b/.test(lower);

  return hasForecastLanguage || hasFutureHorizon;
}

export function isExplicitPolymarketForecastRequest(query: string): boolean {
  const lower = query.toLowerCase();
  const hasPolymarketMention = /\bpolymarket\b/.test(lower) || lower.includes('polymarket_forecast');
  const hasForecastIntent = /\b(?:forecast|prediction|predict|price target)\b/.test(lower);
  return lower.includes('polymarket_forecast')
    || /\bpolymarket forecast\b/.test(lower)
    || /\buse\s+(?:the\s+)?polymarket(?:_forecast| forecast)\b/.test(lower)
    || /\brun\s+(?:the\s+)?polymarket(?:_forecast| forecast)\b/.test(lower)
    || (hasPolymarketMention && hasForecastIntent);
}

export function isExplicitCombinedMarkovPolymarketRequest(query: string): boolean {
  const lower = query.toLowerCase();
  const hasMarkovMention = /\bmarkov(?: chain| distribution)?\b/.test(lower);
  const hasPolymarketMention = /\bpolymarket\b/.test(lower) || lower.includes('polymarket_forecast');
  const hasForecastIntent = /\b(?:forecast|prediction|predict|price target|price outlook|outlook)\b/.test(lower);
  return hasMarkovMention && hasPolymarketMention && hasForecastIntent;
}

export function isExplicitGoldCombinedMarkovPolymarketRequest(query: string): boolean {
  const resolved = resolveAssetIntent(query, extractTickersFn(query)[0] ?? null);
  return resolved.assetClass === 'commodity_gold' && isExplicitCombinedMarkovPolymarketRequest(query);
}

export function detectExplicitSkillRequest(query: string): string | null {
  const match = query.match(/\buse (?:the )?([a-z0-9_-]+) skill\b/i);
  if (!match) return null;

  const requested = match[1].toLowerCase();
  const skill = discoverSkills().find((entry) => entry.name.toLowerCase() === requested);
  return skill?.name ?? null;
}
