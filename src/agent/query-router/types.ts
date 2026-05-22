import type { ToolCallRecord } from '../scratchpad.js';

// Shared type definitions for query-router modules

export type ForcedNonCryptoPolymarketForecastArgs = {
  ticker: string;
  horizon_days: number;
  current_price?: number;
  markov_return?: number;
};

export type ForecastCoverageArgs = {
  ticker: string;
  horizon_days?: number;
  current_price?: number;
  sentiment_score?: number;
  markov_return?: number;
};

export type ForcedForecastArbiterArgs = {
  ticker: string;
  horizon_days: number;
  current_price?: number;
  leverage?: number;
  markov?: {
    forecast_return?: number;
    p_up?: number;
    confidence?: number;
    structural_break?: boolean;
    flat_probability?: number;
    ci_low?: number;
    ci_high?: number;
    trusted_anchors?: number;
    total_anchors?: number;
    anchor_quality?: string;
    conformal?: {
      applied?: boolean;
      radius?: number;
      coverageEstimate?: number | null;
      mode?: 'normal' | 'break';
    };
    summary?: string;
  };
  polymarket?: {
    forecast_return?: number;
    quality_score?: number;
    markets?: Array<{ question: string; probability?: number }>;
    summary?: string;
  };
  whale?: {
    direction?: 'long' | 'short' | 'neutral';
    confidence?: number;
    summary?: string;
  };
};

// Typed interfaces for Markov tool result payloads (replaces Record<string,unknown> casts)

export interface ParsedMarkovConformal {
  applied?: boolean;
  radius?: number;
  coverageEstimate?: number | null;
  mode?: 'normal' | 'break';
}

export interface ParsedMarkovDiagnostics {
  predictionConfidence?: number;
  markovWeight?: number;
  structuralBreakDetected?: boolean;
  trustedAnchors?: number;
  totalAnchors?: number;
  anchorQuality?: string;
  conformal?: ParsedMarkovConformal;
  structuralBreakDivergence?: number;
  ciWidened?: boolean;
}

export interface ParsedMarkovActionSignal {
  expectedReturn?: number;
  confidence?: string;
}

export interface ParsedMarkovScenariosBucket {
  label?: string;
  probability?: number;
}

export interface ParsedMarkovScenarios {
  pUp?: number;
  expectedReturn?: number;
  buckets?: ParsedMarkovScenariosBucket[];
}

export interface ParsedMarkovCanonical {
  actionSignal?: ParsedMarkovActionSignal;
  diagnostics?: ParsedMarkovDiagnostics;
  scenarios?: ParsedMarkovScenarios;
  currentPrice?: number;
}

export interface ParsedMarkovForecastHint {
  markovReturn?: number;
  confidenceScore?: number;
}

export interface ParsedPricePayload {
  price?: number;
  close?: number;
  lastTradePrice?: number;
  snapshot?: ParsedPricePayload;
}

export interface ParsedPolymarketForecastPayload {
  forecastReturn?: unknown;
  result?: unknown;
}

// Utility functions shared across modules

/** Narrows `v` to `T` only if it is a non-null, non-array object. */
export function narrowObj<T>(v: unknown): T | null {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? v as T : null;
}

export function parseToolCallData(call: ToolCallRecord): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(call.result) as { data?: unknown };
    return narrowObj<Record<string, unknown>>(parsed?.data);
  } catch (error) {
    if (!(error instanceof SyntaxError)) {
      throw error;
    }
    return null;
  }
}

export function hasErrorLikeToolResult(result: string): boolean {
  return /^Error:/i.test(result) || /"error"\s*:/i.test(result);
}

export function hasNonEmptyParsedToolData(call: ToolCallRecord): boolean {
  const data = parseToolCallData(call);
  return data !== null && Object.keys(data).length > 0;
}

export function extractPositiveNumericValue(value: unknown): number | null {
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

export function extractPriceFromPayload(value: unknown): number | null {
  if (!value || typeof value !== 'object') return null;
  const record = value as ParsedPricePayload;

  const directPrice = extractPositiveNumericValue(record.price);
  if (directPrice !== null) {
    return directPrice;
  }

  const closePrice = extractPositiveNumericValue(record.close);
  if (closePrice !== null) {
    return closePrice;
  }

  const lastTradePrice = extractPositiveNumericValue(record.lastTradePrice);
  if (lastTradePrice !== null) {
    return lastTradePrice;
  }

  const snapshot = record.snapshot;
  if (snapshot && typeof snapshot === 'object') {
    const snapshotRecord = snapshot as ParsedPricePayload;
    const snapshotPrice = extractPositiveNumericValue(snapshotRecord.price)
      ?? extractPositiveNumericValue(snapshotRecord.close)
      ?? extractPositiveNumericValue(snapshotRecord.lastTradePrice);
    if (snapshotPrice !== null) {
      return snapshotPrice;
    }
  }

  return null;
}

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function isFinitePositiveNumber(value: unknown): value is number {
  return isFiniteNumber(value) && value > 0;
}

export function numbersApproximatelyMatch(actual: unknown, expected: number): boolean {
  if (!isFiniteNumber(actual)) return false;
  const tolerance = Math.max(1e-6, Math.abs(expected) * 1e-6);
  return Math.abs(actual - expected) <= tolerance;
}

export function getForecastHorizonArg(args: Record<string, unknown>): number {
  return isFinitePositiveNumber(args['horizon_days']) ? Math.trunc(args['horizon_days']) : 7;
}

export function getPositiveIntegerArg(args: Record<string, unknown>, key: string): number | null {
  return isFinitePositiveNumber(args[key]) ? Math.trunc(args[key]) : null;
}

export function matchesTickerAndOptionalHorizon(
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
