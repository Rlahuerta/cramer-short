import type { ToolCallRecord } from '../scratchpad.js';
import {
  type ParsedMarkovCanonical,
  type ParsedMarkovActionSignal,
  type ParsedMarkovDiagnostics,
  type ParsedMarkovScenarios,
  type ParsedMarkovForecastHint,
  type ParsedMarkovConformal,
  type ParsedPolymarketForecastPayload,
  type ForcedForecastArbiterArgs,
  narrowObj,
  parseToolCallData,
  extractPriceFromPayload,
  extractPositiveNumericValue,
  isFiniteNumber,
  isFinitePositiveNumber,
  matchesTickerAndOptionalHorizon,
} from './types.js';
import {
  inferDistributionTicker,
  inferDistributionHorizon,
  inferMarkovQueryHorizon,
  getBtcSelectiveMarkovConfidenceThreshold,
  isBtcShortHorizonForecastQuery,
} from './distribution.js';
import {
  buildForcedMarketDataArgs,
  buildForcedNonCryptoMarketDataArgs,
  buildForcedSocialSentimentArgs,
} from './forced-tool-basic-args.js';

export function hasMarketDataQuery(toolCalls: ToolCallRecord[], query: string): boolean {
  return toolCalls.some((call) => call.tool === 'get_market_data' && call.args['query'] === query);
}

export function extractCurrentPriceFromMarketDataQuery(toolCalls: ToolCallRecord[], query: string): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'get_market_data' || call.args['query'] !== query) continue;

    const data = parseToolCallData(call);
    if (!data) continue;

    const direct = extractPriceFromPayload(data);
    if (direct !== null) return direct;

    for (const [key, value] of Object.entries(data)) {
      if (!key.startsWith('get_crypto_price_snapshot_') && !key.startsWith('get_stock_price_')) continue;
      const extracted = extractPriceFromPayload(value);
      if (extracted !== null) return extracted;
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

    const canonical = narrowObj<ParsedMarkovCanonical>(data['canonical']);
    if (!canonical) continue;

    const currentPrice = extractPositiveNumericValue(canonical.currentPrice);
    if (currentPrice !== null) {
      return currentPrice;
    }
  }

  return null;
}

export function extractCurrentPriceFromToolCalls(toolCalls: ToolCallRecord[]): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'get_market_data') continue;

    const data = parseToolCallData(call);
    if (!data) continue;

    const direct = extractPriceFromPayload(data);
    if (direct !== null) return direct;

    for (const [key, value] of Object.entries(data)) {
      if (!key.startsWith('get_crypto_price_snapshot_') && !key.startsWith('get_stock_price_')) continue;
      const extracted = extractPriceFromPayload(value);
      if (extracted !== null) return extracted;
    }
  }

  return null;
}

export function extractCurrentPriceForCryptoQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const marketDataArgs = buildForcedMarketDataArgs(query);
  return marketDataArgs
    ? extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query)
    : null;
}
export function extractCurrentPriceForNonCryptoQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const marketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  if (!marketDataArgs) return null;

  const marketDataPrice = extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query);
  if (marketDataPrice !== null) return marketDataPrice;

  if (!hasMarketDataQuery(toolCalls, marketDataArgs.query)) return null;

  return extractCurrentPriceFromAbstainingMarkovQuery(query, toolCalls);
}

export function extractSentimentScoreFromToolCalls(toolCalls: ToolCallRecord[]): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'social_sentiment') continue;

    const data = parseToolCallData(call);
    const report = data?.['result'];
    if (typeof report !== 'string') continue;

    const match = report.match(/score\s*([+-]?\d+)\/100/i);
    if (!match) continue;

    const parsedScore = parseInt(match[1]!, 10) / 100;
    if (Number.isFinite(parsedScore)) {
      return Math.max(-1, Math.min(1, parsedScore));
    }
  }

  return null;
}
export function extractSentimentScoreForCryptoQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desired = buildForcedSocialSentimentArgs(query);
  if (!desired) return null;

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'social_sentiment') continue;
    if (call.args['ticker'] !== desired.ticker) continue;

    const data = parseToolCallData(call);
    const report = data?.['result'];
    if (typeof report !== 'string') continue;

    const match = report.match(/score\s*([+-]?\d+)\/100/i);
    if (!match) continue;

    const parsedScore = parseInt(match[1]!, 10) / 100;
    if (Number.isFinite(parsedScore)) {
      return Math.max(-1, Math.min(1, parsedScore));
    }
  }

  return null;
}

export function extractMarkovReturnFromToolCalls(toolCalls: ToolCallRecord[]): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'ok') continue;

    const canonical = narrowObj<ParsedMarkovCanonical>(data['canonical']);
    if (!canonical) continue;

    const actionSignal = narrowObj<ParsedMarkovActionSignal>(canonical.actionSignal);
    const diagnostics = narrowObj<ParsedMarkovDiagnostics>(canonical.diagnostics);
    if (!actionSignal || !diagnostics) continue;

    const { expectedReturn } = actionSignal;
    const { markovWeight } = diagnostics;
    if (
      typeof expectedReturn === 'number' && Number.isFinite(expectedReturn)
      && typeof markovWeight === 'number' && Number.isFinite(markovWeight)
    ) {
      return expectedReturn * markovWeight;
    }
  }

  return null;
}

export function extractMarkovReturnForQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);
  const requiresSelectiveBtcGate = isBtcShortHorizonForecastQuery(query);
  const selectiveBtcThreshold = requiresSelectiveBtcGate
    ? getBtcSelectiveMarkovConfidenceThreshold()
    : null;

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution') continue;

    if (data['status'] === 'ok') {
      const canonical = narrowObj<ParsedMarkovCanonical>(data['canonical']);
      if (!canonical) continue;

      const actionSignal = narrowObj<ParsedMarkovActionSignal>(canonical.actionSignal);
      const diagnostics = narrowObj<ParsedMarkovDiagnostics>(canonical.diagnostics);
      if (!actionSignal || !diagnostics) continue;

      const { predictionConfidence } = diagnostics;
      if (
        requiresSelectiveBtcGate
        && isFiniteNumber(predictionConfidence)
        && predictionConfidence < selectiveBtcThreshold!
      ) {
        return null;
      }

      const { expectedReturn } = actionSignal;
      const { markovWeight } = diagnostics;
      if (
        typeof expectedReturn === 'number' && Number.isFinite(expectedReturn)
        && typeof markovWeight === 'number' && Number.isFinite(markovWeight)
      ) {
        return expectedReturn * markovWeight;
      }
    }
  }

  return null;
}

export function extractMarkovPredictionConfidenceForQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution') continue;

    if (data['status'] !== 'ok') continue;

    const canonical = narrowObj<ParsedMarkovCanonical>(data['canonical']);
    if (!canonical) continue;

    const diagnostics = narrowObj<ParsedMarkovDiagnostics>(canonical.diagnostics);
    if (!diagnostics) continue;

    const { predictionConfidence } = diagnostics;
    if (isFiniteNumber(predictionConfidence)) return predictionConfidence;
  }

  return null;
}

export function extractMarkovArbiterEvidence(query: string, toolCalls: ToolCallRecord[]): ForcedForecastArbiterArgs['markov'] | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution') continue;

    if (data['status'] === 'abstain') {
      const evidence: NonNullable<ForcedForecastArbiterArgs['markov']> = {};
      const hint = narrowObj<ParsedMarkovForecastHint>(data['forecastHint']);
      if (hint) {
        if (isFiniteNumber(hint.markovReturn)) evidence.forecast_return = hint.markovReturn;
        if (isFiniteNumber(hint.confidenceScore)) evidence.confidence = hint.confidenceScore;
      }
      const reasons = Array.isArray(data['abstainReasons'])
        ? data['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
        : [];
      const canonical = narrowObj<ParsedMarkovCanonical>(data['canonical']);
      const diagnostics = narrowObj<ParsedMarkovDiagnostics>(canonical?.diagnostics);
      if (diagnostics) {
        if (typeof diagnostics.structuralBreakDetected === 'boolean') evidence.structural_break = diagnostics.structuralBreakDetected;
        if (isFiniteNumber(diagnostics.trustedAnchors)) evidence.trusted_anchors = diagnostics.trustedAnchors;
        if (isFiniteNumber(diagnostics.totalAnchors)) evidence.total_anchors = diagnostics.totalAnchors;
        if (typeof diagnostics.anchorQuality === 'string') evidence.anchor_quality = diagnostics.anchorQuality;
        if (evidence.confidence === undefined && isFiniteNumber(diagnostics.predictionConfidence)) {
          evidence.confidence = diagnostics.predictionConfidence;
        }
      }
      const structuralBreakSummary = diagnostics?.structuralBreakDetected === true
        ? [
            'Structural break detected',
            isFiniteNumber(diagnostics.structuralBreakDivergence)
              ? `divergence ${diagnostics.structuralBreakDivergence.toFixed(3)}`
              : null,
            diagnostics.ciWidened === true ? 'CI widening applied' : null,
          ].filter((part): part is string => part !== null).join(', ')
        : null;
      evidence.summary = [
        reasons.length > 0
          ? `Markov abstained: ${reasons.join('; ')}`
          : 'Markov abstained; treat Markov evidence as diagnostics only.',
        structuralBreakSummary,
      ].filter((part): part is string => Boolean(part)).join(' ');
      return evidence;
    }

    if (data['status'] !== 'ok') continue;

    const canonical = narrowObj<ParsedMarkovCanonical>(data['canonical']);
    if (!canonical) continue;
    const scenarios = narrowObj<ParsedMarkovScenarios>(canonical.scenarios);
    const diagnostics = narrowObj<ParsedMarkovDiagnostics>(canonical.diagnostics);
    const actionSignal = narrowObj<ParsedMarkovActionSignal>(canonical.actionSignal);
    const distribution = data['distribution'];

    const evidence: NonNullable<ForcedForecastArbiterArgs['markov']> = {};
    const weightedReturn = extractMarkovReturnForQuery(query, toolCalls);
    if (weightedReturn !== null) evidence.forecast_return = weightedReturn;

    if (scenarios) {
      if (isFiniteNumber(scenarios.pUp)) evidence.p_up = scenarios.pUp;
      if (isFiniteNumber(scenarios.expectedReturn) && evidence.forecast_return === undefined) {
        evidence.forecast_return = scenarios.expectedReturn;
      }
      if (Array.isArray(scenarios.buckets)) {
        const flat = scenarios.buckets.find((bucket) =>
          bucket && typeof bucket === 'object'
          && typeof bucket.label === 'string'
          && bucket.label.toLowerCase().includes('flat')
        );
        if (flat && isFiniteNumber(flat.probability)) evidence.flat_probability = flat.probability;
      }
    }

    if (diagnostics) {
      if (isFiniteNumber(diagnostics.predictionConfidence)) evidence.confidence = diagnostics.predictionConfidence;
      if (typeof diagnostics.structuralBreakDetected === 'boolean') evidence.structural_break = diagnostics.structuralBreakDetected;
      if (isFiniteNumber(diagnostics.trustedAnchors)) evidence.trusted_anchors = diagnostics.trustedAnchors;
      if (isFiniteNumber(diagnostics.totalAnchors)) evidence.total_anchors = diagnostics.totalAnchors;
      if (typeof diagnostics.anchorQuality === 'string') evidence.anchor_quality = diagnostics.anchorQuality;
      const conformal = narrowObj<ParsedMarkovConformal>(diagnostics.conformal);
      if (conformal) {
        const conformed: NonNullable<NonNullable<ForcedForecastArbiterArgs['markov']>['conformal']> = {};
        if (typeof conformal.applied === 'boolean') conformed.applied = conformal.applied;
        if (isFiniteNumber(conformal.radius)) conformed.radius = conformal.radius;
        if (conformal.coverageEstimate === null || isFiniteNumber(conformal.coverageEstimate)) {
          conformed.coverageEstimate = conformal.coverageEstimate;
        }
        if (conformal.mode === 'normal' || conformal.mode === 'break') {
          conformed.mode = conformal.mode;
        }
        if (Object.keys(conformed).length > 0) evidence.conformal = conformed;
      }
    }

    if (actionSignal && typeof actionSignal.confidence === 'string') {
      evidence.summary = `Markov action signal confidence ${actionSignal.confidence}`;
    }

    if (Array.isArray(distribution)) {
      const prices = distribution
        .map((point) => narrowObj<{ price?: unknown }>(point)?.price ?? null)
        .filter((price): price is number => isFinitePositiveNumber(price));
      if (prices.length > 0) {
        evidence.ci_low = Math.min(...prices);
        evidence.ci_high = Math.max(...prices);
      }
    }

    return Object.keys(evidence).length > 0 ? evidence : null;
  }

  return null;
}

export function hasLowConfidenceBtcShortHorizonMarkov(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isBtcShortHorizonForecastQuery(query)) return false;
  const predictionConfidence = extractMarkovPredictionConfidenceForQuery(query, toolCalls);
  return predictionConfidence !== null && predictionConfidence < getBtcSelectiveMarkovConfidenceThreshold();
}

export function extractPolymarketForecastReturnForQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'polymarket_forecast') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon_days', desiredHorizon)) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(call.result);
    } catch (error) {
      if (!(error instanceof SyntaxError)) {
        throw error;
      }
      continue;
    }

    const data = parsed && typeof parsed === 'object' ? (parsed as { data?: unknown }).data : null;
    const payload = narrowObj<ParsedPolymarketForecastPayload>(data);
    if (!payload) continue;

    const directForecastReturn = payload.forecastReturn;
    if (typeof directForecastReturn === 'number' && Number.isFinite(directForecastReturn)) {
      return directForecastReturn;
    }

    const resultText = payload.result;
    if (typeof resultText !== 'string') continue;
    const match = resultText.match(/(?:forecast return|expected return)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)%/i);
    if (!match) continue;
    const parsedPct = Number.parseFloat(match[1]!);
    if (Number.isFinite(parsedPct)) return parsedPct / 100;
  }

  return null;
}

export function extractPolymarketArbiterEvidence(query: string, toolCalls: ToolCallRecord[]): ForcedForecastArbiterArgs['polymarket'] | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'polymarket_forecast') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon_days', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data) continue;

    const evidence: NonNullable<ForcedForecastArbiterArgs['polymarket']> = {};
    const forecastReturn = extractPolymarketForecastReturnForQuery(query, toolCalls);
    if (forecastReturn !== null) evidence.forecast_return = forecastReturn;

    const resultText = data['result'];
    if (typeof resultText === 'string') {
      const scoreMatch = resultText.match(/Grade:\s*[A-Z][^(]*\((\d+)\/100\)/i);
      if (scoreMatch) {
        const score = Number.parseInt(scoreMatch[1]!, 10);
        if (Number.isFinite(score)) evidence.quality_score = score;
      }

      const markets: Array<{ question: string; probability?: number }> = [];
      const marketPattern = /(Will [^:\n|]+?)[:|]\s*(\d{1,3})%\s+YES/gi;
      let marketMatch: RegExpExecArray | null;
      while ((marketMatch = marketPattern.exec(resultText)) !== null && markets.length < 5) {
        const question = marketMatch[1]?.trim();
        const probability = Number.parseInt(marketMatch[2]!, 10) / 100;
        if (question) markets.push({ question, probability });
      }
      if (markets.length > 0) evidence.markets = markets;
      evidence.summary = resultText.slice(0, 600);
    }

    return Object.keys(evidence).length > 0 ? evidence : null;
  }

  return null;
}
