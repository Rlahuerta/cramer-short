import type { MarkovDistributionPoint } from '../../tools/finance/markov-distribution.js';
import type { ToolCallRecord } from '../scratchpad.js';
import {
  detectBtcShortHorizonDisagreement,
  extractMarkovPredictionConfidenceForQuery,
  getBtcSelectiveMarkovConfidenceThreshold,
  hasAbstainingMarkovDistributionForQuery,
  hasCryptoPolymarketForecastCoverage,
  hasForecastArbitratorForQuery,
  hasLowConfidenceBtcShortHorizonMarkov,
  hasUsableFixedIncomeResult,
  hasUsableOnchainResultForCryptoQuery,
  inferDistributionHorizon,
  inferDistributionTicker,
  inferMarkovQueryHorizon,
  isBtcShortHorizonForecastQuery,
  isDistributionQuery,
  isExplicitPolymarketForecastRequest,
} from '../query-router.js';

/**
 * Remove <think>...</think> blocks that Ollama thinking models sometimes embed
 * directly in response text rather than separating into reasoning_content.
 * Also handles orphan </think> tags (e.g. the model output was: <think>…</think>\nAnswer).
 */
export function stripThinkingTags(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '') // full <think>…</think> blocks
    .replace(/^[\s\S]*?<\/think>\s*/i, '')      // orphan </think>: strip everything up to and including it
    .trim();
}

/**
 * Build a compact Sources footer from a deduplicated list of URLs.
 * Only appended to answers when the model hasn't already cited inline links.
 * Limits to 10 URLs to keep the footer scannable.
 *
 * Social media post URLs (Reddit, X/Twitter, etc.) are excluded — these are
 * inputs to sentiment analysis, not authoritative research citations.
 */

/** Domains excluded from the Sources footer (social media / UGC). */
const EXCLUDED_SOURCE_DOMAINS = [
  'reddit.com',
  'x.com',
  'twitter.com',
  'threads.net',
  'bsky.app',
  'bluesky.app',
];

export function buildSourcesFooter(urls: string[]): string {
  const filtered = urls.filter(u => {
    try {
      const host = new URL(u).hostname.replace(/^www\./, '');
      return !EXCLUDED_SOURCE_DOMAINS.some(d => host === d || host.endsWith(`.${d}`));
    } catch {
      return false;
    }
  });
  const unique = [...new Set(filtered)].slice(0, 10);
  if (unique.length === 0) return '';
  const lines = unique.map((u, i) => `${i + 1}. ${u}`).join('\n');
  return `\n\n---\n**Sources**\n${lines}`;
}

function formatDiagnosticPrice(value: unknown): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value >= 1000
    ? `$${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
    : `$${value.toFixed(2)}`;
}

function formatDiagnosticNumber(value: unknown, digits = 3): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value.toFixed(digits);
}

function inferRequestedBucketCount(query: string): number | null {
  const match = query.match(/\b(\d+)\s*(?:parts|buckets|bins|segments)\b/i);
  if (!match) return null;
  const parsed = Number.parseInt(match[1]!, 10);
  return Number.isInteger(parsed) && parsed >= 2 ? parsed : null;
}

function interpolateSurvivalProbability(
  distribution: MarkovDistributionPoint[],
  price: number,
): number | null {
  if (distribution.length === 0) return null;

  const sorted = [...distribution].sort((a, b) => a.price - b.price);
  if (price <= sorted[0]!.price) return sorted[0]!.probability;
  if (price >= sorted[sorted.length - 1]!.price) return sorted[sorted.length - 1]!.probability;

  for (let i = 0; i < sorted.length - 1; i += 1) {
    const left = sorted[i]!;
    const right = sorted[i + 1]!;
    if (price < left.price || price > right.price) continue;
    if (right.price === left.price) return left.probability;

    const weight = (price - left.price) / (right.price - left.price);
    return left.probability + ((right.probability - left.probability) * weight);
  }

  return sorted[sorted.length - 1]!.probability;
}

function estimateBucketProbabilityPct(
  distribution: MarkovDistributionPoint[],
  lower: number | null,
  upper: number | null,
): number | null {
  const lowerSurvival = lower === null ? 1 : interpolateSurvivalProbability(distribution, lower);
  const upperSurvival = upper === null ? 0 : interpolateSurvivalProbability(distribution, upper);
  if (lowerSurvival === null || upperSurvival === null) return null;
  return Math.max(0, Math.min(100, (lowerSurvival - upperSurvival) * 100));
}

function formatDensityPrice(value: number): string {
  return `$${value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatDensityRange(lower: number | null, upper: number | null): string {
  if (lower === null && upper === null) return 'N/A';
  if (lower === null && upper !== null) return `< ${formatDensityPrice(upper)}`;
  if (lower !== null && upper === null) return `> ${formatDensityPrice(lower)}`;
  return `${formatDensityPrice(lower!)}–${formatDensityPrice(upper!)}`;
}

function buildDensityThresholds(
  minPrice: number,
  maxPrice: number,
  bucketCount: number,
): number[] {
  const thresholdCount = bucketCount - 1;
  if (thresholdCount <= 0) return [];

  const useLogSpacing = minPrice > 0;
  return Array.from({ length: thresholdCount }, (_, index) => {
    const weight = (index + 1) / bucketCount;
    if (useLogSpacing) {
      const logPrice = Math.log(minPrice) + ((Math.log(maxPrice) - Math.log(minPrice)) * weight);
      return Math.exp(logPrice);
    }
    return minPrice + ((maxPrice - minPrice) * weight);
  });
}

function buildCanonicalDensityTable(query: string, toolCalls: ToolCallRecord[]): string | null {
  const requestedBucketCount = inferRequestedBucketCount(query);
  if (requestedBucketCount === null) return null;

  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i -= 1) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'ok') continue;
    if (!Array.isArray(data['distribution'])) continue;

    const distribution = data['distribution']
      .filter((point): point is MarkovDistributionPoint => (
        !!point
        && typeof point === 'object'
        && isFinitePositiveNumber((point as Record<string, unknown>)['price'])
        && isFiniteNumber((point as Record<string, unknown>)['probability'])
      ))
      .map((point) => point as MarkovDistributionPoint)
      .sort((a, b) => a.price - b.price);

    if (distribution.length < 2) continue;

    const minPrice = distribution[0]!.price;
    const maxPrice = distribution[distribution.length - 1]!.price;
    if (!(minPrice > 0) || !(maxPrice > minPrice)) continue;

    const thresholds = buildDensityThresholds(minPrice, maxPrice, requestedBucketCount);

    const rows: string[] = [];
    for (let bucketIndex = 0; bucketIndex < requestedBucketCount; bucketIndex += 1) {
      const lower = bucketIndex === 0 ? null : thresholds[bucketIndex - 1]!;
      const upper = bucketIndex === requestedBucketCount - 1 ? null : thresholds[bucketIndex]!;
      const probabilityPct = estimateBucketProbabilityPct(distribution, lower, upper);
      if (probabilityPct === null) continue;

      rows.push(
        `| ${bucketIndex + 1} | ${formatDensityRange(lower, upper)} | ${probabilityPct.toFixed(2)}% |`,
      );
    }

    if (rows.length < requestedBucketCount) continue;

    return [
      `## ${requestedBucketCount}-Part Density Probability Table`,
      '',
      'Canonical scenario breakdown (P(bucket) = probability mass in each price range):',
      '',
      '| Bucket | Price Range | P(bucket) |',
      '|--------|-------------|-----------|',
      ...rows,
    ].join('\n');
  }

  return null;
}

export function ensureStructuredDensityTable(
  answer: string,
  query: string,
  toolCalls: ToolCallRecord[],
): string {
  if (inferRequestedBucketCount(query) === null) return answer;
  const canonicalTable = buildCanonicalDensityTable(query, toolCalls);
  if (!canonicalTable) return answer;

  const densitySectionPattern = /##\s+.*density probability table[\s\S]*?(?=\n---\n|\n##\s|\n###\s|$)/i;
  if (densitySectionPattern.test(answer)) {
    return answer.replace(densitySectionPattern, canonicalTable);
  }

  return `${canonicalTable}\n\n---\n\n${answer}`;
}

export function buildAbstainingMarkovAnswer(toolCalls: ToolCallRecord[]): string | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(call.result);
    } catch {
      continue;
    }

    const data = parsed && typeof parsed === 'object' ? (parsed as { data?: unknown }).data : null;
    if (!data || typeof data !== 'object') continue;

    const payload = data as Record<string, unknown>;
    if (payload['_tool'] !== 'markov_distribution' || payload['status'] !== 'abstain') continue;

    const canonical = payload['canonical'] && typeof payload['canonical'] === 'object'
      ? payload['canonical'] as Record<string, unknown>
      : {};
    const diagnostics = canonical['diagnostics'] && typeof canonical['diagnostics'] === 'object'
      ? canonical['diagnostics'] as Record<string, unknown>
      : {};
    const abstainReasons = Array.isArray(payload['abstainReasons'])
      ? payload['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
      : [];

    const ticker = typeof canonical['ticker'] === 'string' ? canonical['ticker'] : 'This asset';
    const horizon = typeof canonical['horizon'] === 'number' ? canonical['horizon'] : null;
    const currentPrice = formatDiagnosticPrice(canonical['currentPrice']);
    const trustedAnchors = typeof diagnostics['trustedAnchors'] === 'number' ? diagnostics['trustedAnchors'] : null;
    const totalAnchors = typeof diagnostics['totalAnchors'] === 'number' ? diagnostics['totalAnchors'] : null;
    const anchorQuality = typeof diagnostics['anchorQuality'] === 'string' ? diagnostics['anchorQuality'] : null;
    const outOfSampleR2 = formatDiagnosticNumber(diagnostics['outOfSampleR2']);
    const structuralBreakDetected = diagnostics['structuralBreakDetected'] === true;
    const structuralBreakDivergence = formatDiagnosticNumber(diagnostics['structuralBreakDivergence']);
    const predictionConfidence = formatDiagnosticNumber(diagnostics['predictionConfidence'], 2);

    const lines = [
      `## ${ticker} ${horizon !== null ? `${horizon}-Day ` : ''}Probability Distribution: Model Abstained`,
      '',
      'A calibrated probability distribution is not available. `markov_distribution` explicitly abstained, so no replacement scenario probabilities are shown.',
      '',
      '## Why it abstained',
      ...(abstainReasons.length > 0
        ? abstainReasons.map((reason) => `- ${reason}`)
        : ['- The tool marked this run as diagnostics-only.']),
      '',
      '## Diagnostics',
      '| Metric | Value |',
      '|--------|-------|',
      ...(currentPrice ? [`| Current price | ${currentPrice} |`] : []),
      `| Trusted anchors | ${trustedAnchors !== null && totalAnchors !== null ? `${trustedAnchors} / ${totalAnchors}` : 'N/A'} |`,
      `| Anchor quality | ${anchorQuality ?? 'N/A'} |`,
      `| Out-of-sample R² | ${outOfSampleR2 ?? 'N/A'} |`,
      `| Structural break | ${structuralBreakDetected ? `Yes${structuralBreakDivergence ? ` (${structuralBreakDivergence})` : ''}` : 'No'} |`,
      `| Prediction confidence | ${predictionConfidence ?? 'N/A'} |`,
      '',
      '## Interpretation',
      'Use later prediction-market searches only as diagnostics about missing or low-quality anchors. They do not justify a replacement calibrated distribution for this horizon.',
      '',
      '## Next best option',
      '- Wait for horizon-matched terminal threshold markets to list, or',
      '- Use `polymarket_forecast` for a point estimate with a confidence interval instead of a full distribution.',
    ];

    return lines.join('\n');
  }

  return null;
}

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

function parseToolCallData(call: ToolCallRecord): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(call.result) as { data?: unknown };
    return parsed?.data && typeof parsed.data === 'object'
      ? parsed.data as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function hasErrorLikeToolResult(result: string): boolean {
  return /^Error:/i.test(result) || /"error"\s*:/i.test(result);
}

function hasNonEmptyParsedToolData(call: ToolCallRecord): boolean {
  const data = parseToolCallData(call);
  return data !== null && Object.keys(data).length > 0;
}

function extractPositiveNumericValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return value;
  }

  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }

  return null;
}

function extractPriceFromPayload(value: unknown): number | null {
  if (!value || typeof value !== 'object') return null;
  const record = value as Record<string, unknown>;

  const directPrice = extractPositiveNumericValue(record['price']);
  if (directPrice !== null) {
    return directPrice;
  }

  const closePrice = extractPositiveNumericValue(record['close']);
  if (closePrice !== null) {
    return closePrice;
  }

  const lastTradePrice = extractPositiveNumericValue(record['lastTradePrice']);
  if (lastTradePrice !== null) {
    return lastTradePrice;
  }

  const snapshot = record['snapshot'];
  if (snapshot && typeof snapshot === 'object') {
    const snapshotRecord = snapshot as Record<string, unknown>;
    const snapshotPrice = extractPositiveNumericValue(snapshotRecord['price'])
      ?? extractPositiveNumericValue(snapshotRecord['close'])
      ?? extractPositiveNumericValue(snapshotRecord['lastTradePrice']);
    if (snapshotPrice !== null) {
      return snapshotPrice;
    }
  }

  return null;
}


function extractCurrentPriceFromAbstainingMarkovQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'abstain') continue;

    const canonical = data['canonical'];
    if (!canonical || typeof canonical !== 'object') continue;

    const currentPrice = extractPositiveNumericValue((canonical as Record<string, unknown>)['currentPrice']);
    if (currentPrice !== null) {
      return currentPrice;
    }
  }

  return null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isFinitePositiveNumber(value: unknown): value is number {
  return isFiniteNumber(value) && value > 0;
}

function numbersApproximatelyMatch(actual: unknown, expected: number): boolean {
  if (!isFiniteNumber(actual)) return false;
  const tolerance = Math.max(1e-6, Math.abs(expected) * 1e-6);
  return Math.abs(actual - expected) <= tolerance;
}






function getForecastHorizonArg(args: Record<string, unknown>): number {
  return isFinitePositiveNumber(args['horizon_days']) ? Math.trunc(args['horizon_days']) : 7;
}

function getPositiveIntegerArg(args: Record<string, unknown>, key: string): number | null {
  return isFinitePositiveNumber(args[key]) ? Math.trunc(args[key]) : null;
}

function matchesTickerAndOptionalHorizon(
  args: Record<string, unknown>,
  ticker: string | null,
  horizonKey: string,
  horizon: number | null,
): boolean {
  if (ticker) {
    const existingTicker = typeof args['ticker'] === 'string' ? args['ticker'].toUpperCase() : null;
    const expectedTicker = ticker.toUpperCase();
    const equivalentCryptoTicker = expectedTicker.endsWith('-USD')
      && existingTicker === expectedTicker.replace(/-USD$/, '');
    if (existingTicker !== expectedTicker && !equivalentCryptoTicker) return false;
  }

  if (horizon !== null && getPositiveIntegerArg(args, horizonKey) !== horizon) {
    return false;
  }

  return true;
}

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


function hasSuccessfulMarkovDistribution(toolCalls: ToolCallRecord[]): boolean {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    try {
      const parsed = JSON.parse(call.result) as { data?: { _tool?: string; status?: string } };
      if (parsed?.data?._tool === 'markov_distribution' && parsed.data.status === 'ok') {
        return true;
      }
    } catch {
      continue;
    }
  }
  return false;
}

export function buildUnavailableDistributionAnswer(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!isDistributionQuery(query)) return null;
  if (hasSuccessfulMarkovDistribution(toolCalls)) return null;

  return [
    '## Probability Distribution Unavailable',
    '',
    'A calibrated price-distribution answer is not available for this query. No successful non-abstaining `markov_distribution` result was produced, so I am not emitting substitute distribution buckets or historical-volatility scenario percentages.',
    '',
    '## Why this answer stops here',
    '- A price-distribution request needs horizon-matched terminal threshold anchors or a validated canonical Markov distribution.',
    '- Neither condition was met in this run, so any bucketed probability table would be speculative rather than grounded.',
    '',
    '## Safe next options',
    '- Wait for terminal threshold markets that match the requested horizon, or',
    '- Use `polymarket_forecast` for a point estimate with a confidence interval instead of a full distribution.',
  ].join('\n');
}

export function buildDistributionWarningPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!isDistributionQuery(query)) return null;
  if (hasSuccessfulMarkovDistribution(toolCalls)) return null;

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(call.result);
    } catch {
      continue;
    }

    const data = parsed && typeof parsed === 'object' ? (parsed as { data?: unknown }).data : null;
    if (!data || typeof data !== 'object') continue;

    const payload = data as Record<string, unknown>;
    if (payload['_tool'] !== 'markov_distribution' || payload['status'] !== 'abstain') continue;

    const abstainReasons = Array.isArray(payload['abstainReasons'])
      ? payload['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
      : [];

    return [
      '## Warning: no calibrated Markov terminal distribution was available',
      '',
      'The Markov distribution workflow abstained for this query. Any fallback analysis below must be treated as non-distribution context unless it comes directly from a validated canonical Markov payload.',
      ...(abstainReasons.length > 0
        ? ['', 'Key abstain reasons:', ...abstainReasons.map((reason) => `- ${reason}`)]
        : []),
      '',
      '---',
      '',
    ].join('\n');
  }

  return [
    '## Warning: no validated Markov distribution was produced',
    '',
    'Cramer-Short did not produce a successful non-abstaining `markov_distribution` result for this distribution query. Any answer below should be read as fallback analysis, not a calibrated probability distribution.',
    '',
    '---',
    '',
  ].join('\n');
}

export function buildForecastDisagreementPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!detectBtcShortHorizonDisagreement(query, toolCalls)) return null;

  return [
    '## Warning: BTC short-horizon signals are mixed',
    '',
    'Markov and Polymarket are pointing in different directions for this BTC short-horizon forecast. Read any directional takeaway below as mixed evidence with moderated confidence, not a high-conviction signal.',
    '',
    '---',
    '',
  ].join('\n');
}

export function buildLowConfidenceBtcShortHorizonForecastPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!hasLowConfidenceBtcShortHorizonMarkov(query, toolCalls)) return null;

  const predictionConfidence = extractMarkovPredictionConfidenceForQuery(query, toolCalls);
  const selectiveBtcThreshold = getBtcSelectiveMarkovConfidenceThreshold();
  const confidenceText = predictionConfidence !== null
    ? predictionConfidence.toFixed(2)
    : 'N/A';

  return [
    '## Warning: BTC short-horizon selective Markov gate did not clear',
    '',
    `The Markov run completed, but prediction confidence ${confidenceText} is below the ${selectiveBtcThreshold.toFixed(2)} selective threshold. Do not treat the Markov direction here as part of the aggregate selective-accuracy slice or as a validated BTC edge. Read any forecast below as fallback context, not a selective Markov signal.`,
    '',
    '---',
    '',
  ].join('\n');
}
