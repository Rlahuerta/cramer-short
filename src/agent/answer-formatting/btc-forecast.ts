import type { ToolCallRecord } from '../scratchpad.js';
import {
  hasAbstainingMarkovDistributionForQuery,
  hasCryptoPolymarketForecastCoverage,
  hasForecastArbitratorForQuery,
  hasUsableFixedIncomeResult,
  hasUsableOnchainResultForCryptoQuery,
  isBtcShortHorizonForecastQuery,
  isExplicitPolymarketForecastRequest,
} from '../query-router.js';
import { buildAbstainingMarkovAnswer } from './markov-answers.js';

export function shouldPreserveAbstainingBtcShortHorizonForecast(
  query: string,
  toolCalls: ToolCallRecord[],
): boolean {
  if (isExplicitPolymarketForecastRequest(query)) return false;
  if (hasForecastArbitratorForQuery(query, toolCalls)) return false;
  if (hasCryptoPolymarketForecastCoverage(query, toolCalls)) return false;
  if (hasUsableOnchainResultForCryptoQuery(query, toolCalls)) return false;
  if (hasUsableFixedIncomeResult(toolCalls)) return false;
  return isBtcShortHorizonForecastQuery(query)
    && hasAbstainingMarkovDistributionForQuery(query, toolCalls);
}

export function buildAbstainingBtcShortHorizonForecastAnswer(
  query: string,
  toolCalls: ToolCallRecord[],
): string | null {
  if (!shouldPreserveAbstainingBtcShortHorizonForecast(query, toolCalls)) return null;

  const diagnostics = buildAbstainingMarkovAnswer(toolCalls);
  if (!diagnostics) return null;

  return [
    diagnostics,
    '',
    '## Decision guidance',
    'Treat this horizon as no-trade / no-calibrated-edge unless new horizon-matched terminal threshold markets appear. Any later fallback tools may still be useful for context, but they do not replace the abstained Markov forecast for BTC short horizons.',
  ].join('\n');
}
