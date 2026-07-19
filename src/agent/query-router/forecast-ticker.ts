import { extractExclusiveAssetOverride } from '../../tools/finance/asset-resolver.js';
import { inferDistributionTicker } from './distribution.js';
import { isExplicitGoldCombinedMarkovPolymarketRequest } from './classification.js';
import { referencesMultipleDistinctAssets } from './asset-scope.js';

/**
 * Single source of truth for the ticker a forecast tool should use for a query.
 *
 * Shared by the pre-execution normalizer (agent loop) and the tool-executor
 * choke point, so every call path — the model's own tool calls, forced/
 * pre-executed calls, and parallel duplicates — resolves the same canonical
 * ticker. Returns null when the tool is not a forecast tool, when no asset can
 * be resolved (macro), or when the query is an ambiguous multi-asset request
 * without an explicit "X only" scope (stays conservative — never clobber a
 * legitimate comparison). Gold-combined requests keep their dedicated
 * normalizer and are skipped here.
 *
 * Per-tool convention: markov_distribution uses the markov form (hyphenated
 * crypto `BTC-USD` or ETF proxy `GLD`); polymarket_forecast and
 * forecast_arbitrator use the bare root (`BTC`) or the same proxy.
 */
const FORECAST_TICKER_TOOLS = new Set([
  'markov_distribution', 'polymarket_forecast', 'forecast_arbitrator',
]);

export function resolveForecastToolTicker(query: string, toolName: string): string | null {
  if (!FORECAST_TICKER_TOOLS.has(toolName)) return null;
  if (isExplicitGoldCombinedMarkovPolymarketRequest(query)) return null;

  const markovTicker = inferDistributionTicker(query);
  if (!markovTicker) return null;

  if (!extractExclusiveAssetOverride(query) && referencesMultipleDistinctAssets(query)) return null;

  if (toolName === 'markov_distribution') return markovTicker;
  return markovTicker.endsWith('-USD') ? markovTicker.slice(0, -'-USD'.length) : markovTicker;
}
