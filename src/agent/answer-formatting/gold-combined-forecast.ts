import type { ToolCallRecord } from '../scratchpad.js';
import {
  buildForcedGoldCombinedForecastArbiterArgs,
  inferDistributionHorizon,
  inferDistributionTicker,
  isExplicitGoldCombinedMarkovPolymarketRequest,
} from '../query-router.js';
import {
  formatDiagnosticNumber,
  formatDiagnosticPrice,
  matchesTickerAndOptionalHorizon,
  parseToolCallData,
} from './tool-call-utils.js';

type ForecastArbitratorResult = {
  currentPrice?: unknown;
  verdict?: unknown;
  preferredDirection?: unknown;
  confidence?: unknown;
  shouldEnterNow?: unknown;
  disagreement?: { summary?: unknown } | null;
  leverageAssessment?: { warning?: unknown } | null;
  policy?: {
    level?: unknown;
    tradeEligible?: unknown;
    reasons?: unknown;
  } | null;
};

function formatReturnPct(value: unknown): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`;
}

function formatProbabilityPct(value: unknown): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return `${(value * 100).toFixed(1)}%`;
}

function normalizeSummary(value: unknown, maxLength = 220): string | null {
  if (typeof value !== 'string') return null;
  const collapsed = value.replace(/\s+/g, ' ').trim();
  if (!collapsed) return null;
  return collapsed.length <= maxLength ? collapsed : `${collapsed.slice(0, maxLength - 1)}…`;
}

function extractDivergence(summary: string | null): string | null {
  if (!summary) return null;
  const match = summary.match(/\bdivergence\s+([0-9]+(?:\.[0-9]+)?)\b/i);
  if (!match) return null;
  const value = Number.parseFloat(match[1]!);
  return Number.isFinite(value) ? value.toFixed(3) : null;
}

function extractLatestForecastArbitratorResult(
  query: string,
  toolCalls: ToolCallRecord[],
): ForecastArbitratorResult | null {
  const ticker = inferDistributionTicker(query);
  const horizon = inferDistributionHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i -= 1) {
    const call = toolCalls[i];
    if (call.tool !== 'forecast_arbitrator') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, ticker, 'horizon_days', horizon)) continue;

    const data = parseToolCallData(call);
    const result = data?.result;
    if (result && typeof result === 'object') {
      return result as ForecastArbitratorResult;
    }
  }

  return null;
}

function formatHorizonLabel(query: string, horizon: number): string {
  if (/\b24\s*hours?\b/i.test(query)) return '24h';
  if (/\b48\s*hours?\b/i.test(query)) return '48h';
  return horizon === 1 ? '1-day' : `${horizon}-day`;
}

export function buildExplicitGoldCombinedForecastAnswer(
  query: string,
  toolCalls: ToolCallRecord[],
): string | null {
  if (!isExplicitGoldCombinedMarkovPolymarketRequest(query)) return null;

  const ticker = inferDistributionTicker(query);
  const horizon = inferDistributionHorizon(query);
  if (ticker !== 'GLD' || horizon === null) return null;

  const arbiterArgs = buildForcedGoldCombinedForecastArbiterArgs(query, toolCalls);
  if (!arbiterArgs?.markov || !arbiterArgs.polymarket) return null;

  const arbiterResult = extractLatestForecastArbitratorResult(query, toolCalls);
  const markov = arbiterArgs.markov;
  const polymarket = arbiterArgs.polymarket;
  const currentPrice = formatDiagnosticPrice(
    typeof arbiterResult?.currentPrice === 'number' ? arbiterResult.currentPrice : arbiterArgs.current_price,
  );
  const markovReturn = formatReturnPct(markov.forecast_return);
  const markovConfidence = formatDiagnosticNumber(markov.confidence, 2);
  const markovPUp = formatProbabilityPct(markov.p_up);
  const markovCiLow = formatDiagnosticPrice(markov.ci_low);
  const markovCiHigh = formatDiagnosticPrice(markov.ci_high);
  const polymarketReturn = formatReturnPct(polymarket.forecast_return);
  const polymarketQuality = typeof polymarket.quality_score === 'number'
    ? `${polymarket.quality_score}/100`
    : null;
  const firstMarket = polymarket.markets?.[0];
  const firstMarketProbability = formatProbabilityPct(firstMarket?.probability);
  const markovSummary = normalizeSummary(markov.summary);
  const disagreementSummary = normalizeSummary(arbiterResult?.disagreement?.summary);
  const leverageWarning = normalizeSummary(arbiterResult?.leverageAssessment?.warning);
  const policyLevel = typeof arbiterResult?.policy?.level === 'string'
    ? arbiterResult.policy.level.replace(/-/g, ' ')
    : null;
  const policyReasons = Array.isArray(arbiterResult?.policy?.reasons)
    ? arbiterResult?.policy?.reasons.filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
    : [];
  const ciWidened = markov.conformal?.mode === 'break'
    || markov.conformal?.applied === true
    || /ci widening applied|ci widened|ci widen/i.test(markovSummary ?? '');
  const structuralBreakDetected = markov.structural_break === true;
  const divergence = extractDivergence(markovSummary);
  const anchorCoverage = markov.trusted_anchors !== undefined && markov.total_anchors !== undefined
    ? `${markov.trusted_anchors}/${markov.total_anchors}`
    : null;

  const signalSummaryParts = [
    currentPrice ? `Current price ${currentPrice}` : null,
    markovReturn ? `Markov expected return ${markovReturn}` : null,
    markovPUp ? `P(up) ${markovPUp}` : null,
    markovConfidence ? `confidence ${markovConfidence}` : null,
    markovCiLow && markovCiHigh ? `95% CI ${markovCiLow} to ${markovCiHigh}` : null,
  ].filter((part): part is string => part !== null);

  const polymarketParts = [
    polymarketReturn ? `forecast return ${polymarketReturn}` : null,
    polymarketQuality ? `quality ${polymarketQuality}` : null,
    firstMarket?.question && firstMarketProbability
      ? `${firstMarket.question} (${firstMarketProbability} YES)`
      : null,
  ].filter((part): part is string => part !== null);

  const lines = [
    `## GOLD (GLD) ${formatHorizonLabel(query, horizon)} combined forecast`,
    '',
    '## Signal summary',
    `- ${signalSummaryParts.join('; ') || 'Markov evidence loaded for GLD.'}.`,
    `- Polymarket: ${polymarketParts.join('; ') || 'horizon-matched market evidence loaded for GLD.'}.`,
  ];

  if (arbiterResult) {
    const verdict = typeof arbiterResult.verdict === 'string' ? arbiterResult.verdict : null;
    const preferredDirection = typeof arbiterResult.preferredDirection === 'string'
      ? arbiterResult.preferredDirection
      : null;
    const confidence = typeof arbiterResult.confidence === 'string' ? arbiterResult.confidence : null;
    const shouldEnterNow = typeof arbiterResult.shouldEnterNow === 'boolean'
      ? arbiterResult.shouldEnterNow
      : null;

    lines.push(
      '',
      '## Final trade view',
      `- Verdict: ${verdict ?? 'N/A'}${preferredDirection ? ` | Preferred direction: ${preferredDirection}` : ''}${confidence ? ` | Confidence: ${confidence}` : ''}.`,
      ...(shouldEnterNow === null ? [] : [`- Enter now: ${shouldEnterNow ? 'yes' : 'no'}.`]),
      ...(disagreementSummary ? [`- Reconciliation: ${disagreementSummary}.`] : []),
      ...(policyLevel ? [`- Policy: ${policyLevel}${policyReasons.length > 0 ? ` — ${policyReasons.join('; ')}` : ''}.`] : []),
      ...(leverageWarning ? [`- Risk note: ${leverageWarning}.`] : []),
    );
  }

  if (structuralBreakDetected) {
    lines.push(
      '',
      '## Structural Break Diagnostic',
      `- Detected: yes${divergence ? ` | Divergence score: ${divergence}` : ''}.`,
      ...(markovSummary ? [`- Trigger summary: ${markovSummary}.`] : []),
      ...(anchorCoverage || markov.anchor_quality
        ? [`- Anchor quality: ${markov.anchor_quality ?? 'N/A'}${anchorCoverage ? ` (${anchorCoverage} trusted anchors)` : ''}.`]
        : []),
      `- CI widening applied: ${ciWidened ? 'yes' : 'no'}.`,
      '- Confidence should be treated as downgraded until cleaner GLD anchor coverage returns.',
    );
  }

  return lines.join('\n');
}
