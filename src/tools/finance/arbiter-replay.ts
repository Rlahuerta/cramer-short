import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { cramerShortPath } from '../../utils/paths.js';
import {
  classifyPolymarketQuestion,
  extractPriceLevels,
  type ForecastArbiterInput,
  type ForecastMarketSemantics,
} from './forecast-arbitrator.js';

export interface RawPolymarketReplayMarket {
  marketId: string;
  assetId?: string;
  question: string;
  probability: number;
  volume24h: number;
  createdAt?: string;
  endDate?: string | null;
  bid?: number;
  ask?: number;
  active?: boolean;
  closed?: boolean;
  enableOrderBook?: boolean;
}

export interface RawPolymarketReplayRow {
  capturedAt: string;
  ticker: string;
  horizonDays: number;
  currentPrice: number | null;
  querySet: string[];
  selectedMarketIds: string[];
  candidates: RawPolymarketReplayMarket[];
  warnings?: string[];
}

export interface PolymarketReplaySelectionMarket {
  marketId?: string;
  assetId?: string;
  question: string;
  probability: number;
  volume24h: number;
  endDate?: string | null;
  bid?: number;
  ask?: number;
  relevanceScore?: number;
  signalCategory?: string;
  active?: boolean;
  closed?: boolean;
  enableOrderBook?: boolean;
}

export interface RawWhaleReplayTransaction {
  hash: string;
  timestamp: string;
  symbol?: string;
  valueUsd?: number | null;
  fromOwner?: string;
  toOwner?: string;
}

export interface RawWhaleReplayRow {
  capturedAt: string;
  ticker: string;
  source: string;
  observationWindowStart: string;
  observationWindowEnd: string;
  transactions: RawWhaleReplayTransaction[];
  warnings?: string[];
}

export interface ArbiterReplayPolymarketMarket {
  marketId: string;
  assetId: string;
  question: string;
  probability: number;
  volume24h: number;
  endDate: string;
  semantics: ForecastMarketSemantics;
  extractedPriceLevels: number[];
  relevanceScore?: number;
  bid?: number;
  ask?: number;
}

export interface ArbiterReplayForecastLabel {
  realizedPrice: number;
  realizedReturn: number;
  actualBinary: 0 | 1;
  labeledAt: string;
}

export interface ArbiterReplaySemanticLabel {
  marketId: string;
  semantics: ForecastMarketSemantics;
  outcome: 'yes' | 'no' | 'unsupported';
  labeledAt: string;
}

export interface ArbiterReplayBundle {
  capturedAt: string;
  ticker: string;
  horizonDays: number;
  currentPrice: number | null;
  leverage?: number;
  markov?: ForecastArbiterInput['markov'];
  polymarket?: {
    querySet: string[];
    selectedMarketIds: string[];
    selectedMarkets: ArbiterReplayPolymarketMarket[];
    summary?: string;
    confidence?: number;
    forecastReturn?: number;
    qualityScore?: number;
    qualityGrade?: string;
    semanticParserVersion?: string;
    warnings: string[];
  } | null;
  whale?: (
    NonNullable<ForecastArbiterInput['whale']> & {
      source: string;
      observationWindowStart: string;
      observationWindowEnd: string;
      txCount?: number;
      notionalUsd?: number | null;
      txHashes?: string[];
    }
  ) | null;
  onchainContext?: {
    market?: Record<string, unknown>;
    sentiment?: Record<string, unknown>;
    developer?: Record<string, unknown>;
    community?: Record<string, unknown>;
    global?: Record<string, unknown>;
  };
  warnings: string[];
  labels?: {
    forecast?: ArbiterReplayForecastLabel;
    semantic?: ArbiterReplaySemanticLabel[];
  };
}

export const DEFAULT_ARBITER_REPLAY_BUNDLES_PATH = cramerShortPath('arbiter-replay-bundles.jsonl');
export const DEFAULT_ARBITER_REPLAY_POLYMARKET_RAW_PATH = cramerShortPath('arbiter-replay-polymarket-raw.jsonl');
export const DEFAULT_ARBITER_REPLAY_WHALE_RAW_PATH = cramerShortPath('arbiter-replay-whale-raw.jsonl');
export const DEFAULT_ARBITER_REPLAY_CACHE_DIR = cramerShortPath('cache', 'arbiter-replay');
export const DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH = cramerShortPath('cache', 'arbiter-replay', 'bundles.jsonl');
export const DEFAULT_ARBITER_REPLAY_CACHE_POLYMARKET_RAW_PATH = cramerShortPath('cache', 'arbiter-replay', 'polymarket-raw.jsonl');
export const DEFAULT_ARBITER_REPLAY_CACHE_WHALE_RAW_PATH = cramerShortPath('cache', 'arbiter-replay', 'whale-raw.jsonl');
export const DEFAULT_POLYMARKET_SEMANTIC_PARSER_VERSION = 'forecast-arbitrator:classifyPolymarketQuestion';

const KNOWN_EXCHANGE_MARKERS = [
  'exchange',
  'binance',
  'coinbase',
  'kraken',
  'bitfinex',
  'okx',
  'bybit',
  'gemini',
  'huobi',
  'kucoin',
];

const KNOWN_ACCUMULATION_MARKERS = [
  'accumulation',
  'cold wallet',
  'custody',
  'treasury',
  'vault',
];

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isOptionalFiniteNumber(value: unknown): value is number | null | undefined {
  return value === undefined || value === null || isFiniteNumber(value);
}

function isOptionalNumber(value: unknown): value is number | undefined {
  return value === undefined || isFiniteNumber(value);
}

function isIsoTimestamp(value: unknown): value is string {
  return typeof value === 'string' && Number.isFinite(Date.parse(value));
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === 'string');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isReplaySemantics(value: unknown): value is ForecastMarketSemantics {
  return value === 'terminal'
    || value === 'barrier_touch'
    || value === 'range'
    || value === 'path_dependent'
    || value === 'unknown';
}

function isRawPolymarketReplayMarket(value: unknown): value is RawPolymarketReplayMarket {
  if (!isRecord(value)) return false;
  if (typeof value.marketId !== 'string' || value.marketId.trim().length === 0) return false;
  if (value.assetId !== undefined && typeof value.assetId !== 'string') return false;
  if (typeof value.question !== 'string') return false;
  if (!isFiniteNumber(value.probability) || value.probability < 0 || value.probability > 1) return false;
  if (!isFiniteNumber(value.volume24h) || value.volume24h < 0) return false;
  if (value.createdAt !== undefined && !isIsoTimestamp(value.createdAt)) return false;
  if (
    value.endDate !== undefined
    && value.endDate !== null
    && (typeof value.endDate !== 'string' || !Number.isFinite(Date.parse(value.endDate)))
  ) {
    return false;
  }
  if (!isOptionalFiniteNumber(value.bid) || !isOptionalFiniteNumber(value.ask)) return false;
  return true;
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim().length > 0 ? value : undefined;
}

function asFiniteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function asIsoFromUnknownTimestamp(value: unknown): string | undefined {
  if (typeof value === 'string' && Number.isFinite(Date.parse(value))) {
    return new Date(Date.parse(value)).toISOString();
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    const ms = value < 1e12 ? value * 1000 : value;
    if (Number.isFinite(ms)) return new Date(ms).toISOString();
  }
  return undefined;
}

function ownerLabel(value: unknown): string | undefined {
  if (typeof value === 'string' && value.trim().length > 0) return value;
  if (!isRecord(value)) return undefined;
  return asString(value.owner)
    ?? asString(value.owner_type)
    ?? asString(value.label)
    ?? asString(value.name)
    ?? asString(value.address);
}

function isRawWhaleReplayTransaction(value: unknown): value is RawWhaleReplayTransaction {
  if (!isRecord(value)) return false;
  if (typeof value.hash !== 'string' || value.hash.trim().length === 0) return false;
  if (!isIsoTimestamp(value.timestamp)) return false;
  if (value.symbol !== undefined && typeof value.symbol !== 'string') return false;
  if (!isOptionalFiniteNumber(value.valueUsd)) return false;
  if (value.fromOwner !== undefined && typeof value.fromOwner !== 'string') return false;
  if (value.toOwner !== undefined && typeof value.toOwner !== 'string') return false;
  return true;
}

function isReplayPolymarketMarket(value: unknown): value is ArbiterReplayPolymarketMarket {
  if (!isRecord(value)) return false;
  if (typeof value.marketId !== 'string' || value.marketId.trim().length === 0) return false;
  if (typeof value.assetId !== 'string' || value.assetId.trim().length === 0) return false;
  if (typeof value.question !== 'string') return false;
  if (!isFiniteNumber(value.probability) || value.probability < 0 || value.probability > 1) return false;
  if (!isFiniteNumber(value.volume24h) || value.volume24h < 0) return false;
  if (typeof value.endDate !== 'string' || !Number.isFinite(Date.parse(value.endDate))) return false;
  if (!isReplaySemantics(value.semantics)) return false;
  if (!Array.isArray(value.extractedPriceLevels) || !value.extractedPriceLevels.every((entry) => isFiniteNumber(entry))) {
    return false;
  }
  if (!isOptionalFiniteNumber(value.relevanceScore) || !isOptionalFiniteNumber(value.bid) || !isOptionalFiniteNumber(value.ask)) {
    return false;
  }
  return true;
}

function isReplaySemanticLabel(value: unknown): value is ArbiterReplaySemanticLabel {
  if (!isRecord(value)) return false;
  if (typeof value.marketId !== 'string' || value.marketId.trim().length === 0) return false;
  if (!isReplaySemantics(value.semantics)) return false;
  if (value.outcome !== 'yes' && value.outcome !== 'no' && value.outcome !== 'unsupported') return false;
  if (!isIsoTimestamp(value.labeledAt)) return false;
  return true;
}

function readJsonlRecords<T>(
  filePath: string,
  parser: (line: string) => T | null,
  warningLabel: string,
): T[] {
  if (!existsSync(filePath)) return [];
  const content = readFileSync(filePath, 'utf-8');
  if (!content.trim()) return [];

  const rows: T[] = [];
  for (const [index, line] of content.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    const parsed = parser(line);
    if (!parsed) {
      console.warn(`[${warningLabel}] Skipping malformed line ${index + 1} in ${filePath}`);
      continue;
    }
    rows.push(parsed);
  }
  return rows;
}

function appendJsonlRecords<T>(filePath: string, records: T[]): void {
  if (records.length === 0) return;
  mkdirSync(dirname(filePath), { recursive: true });
  const payload = `${records.map((record) => JSON.stringify(record)).join('\n')}\n`;
  appendFileSync(filePath, payload, 'utf-8');
}

export function parseRawPolymarketReplayLine(rawLine: string): RawPolymarketReplayRow | null {
  const trimmed = rawLine.trim();
  if (!trimmed) return null;

  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (!isIsoTimestamp(parsed.capturedAt)) return null;
    if (typeof parsed.ticker !== 'string' || parsed.ticker.trim().length === 0) return null;
    if (!Number.isInteger(parsed.horizonDays) || (parsed.horizonDays as number) < 1) return null;
    if (!(parsed.currentPrice === null || (isFiniteNumber(parsed.currentPrice) && parsed.currentPrice > 0))) return null;
    if (!isStringArray(parsed.querySet) || !isStringArray(parsed.selectedMarketIds)) return null;
    if (!Array.isArray(parsed.candidates) || !parsed.candidates.every(isRawPolymarketReplayMarket)) return null;
    if (parsed.warnings !== undefined && !isStringArray(parsed.warnings)) return null;
    const capturedAt = parsed.capturedAt as string;
    const ticker = parsed.ticker as string;
    const horizonDays = parsed.horizonDays as number;
    const currentPrice = parsed.currentPrice as number | null;
    const querySet = parsed.querySet as string[];
    const selectedMarketIds = parsed.selectedMarketIds as string[];
    const candidates = parsed.candidates as RawPolymarketReplayMarket[];
    const warnings = parsed.warnings as string[] | undefined;

    return {
      capturedAt,
      ticker,
      horizonDays,
      currentPrice,
      querySet: [...querySet],
      selectedMarketIds: [...selectedMarketIds],
      candidates: candidates.map((candidate) => ({ ...candidate })),
      ...(warnings ? { warnings: [...warnings] } : {}),
    };
  } catch {
    return null;
  }
}

export function readRawPolymarketReplayRows(
  filePath: string = DEFAULT_ARBITER_REPLAY_POLYMARKET_RAW_PATH,
): RawPolymarketReplayRow[] {
  return readJsonlRecords(filePath, parseRawPolymarketReplayLine, 'arbiter-replay-polymarket');
}

export function appendRawPolymarketReplayRows(
  records: RawPolymarketReplayRow[],
  filePath: string = DEFAULT_ARBITER_REPLAY_POLYMARKET_RAW_PATH,
): void {
  appendJsonlRecords(filePath, records);
}

export function appendRawPolymarketReplayRow(
  record: RawPolymarketReplayRow,
  filePath: string = DEFAULT_ARBITER_REPLAY_POLYMARKET_RAW_PATH,
): void {
  appendRawPolymarketReplayRows([record], filePath);
}

export function parseRawWhaleReplayLine(rawLine: string): RawWhaleReplayRow | null {
  const trimmed = rawLine.trim();
  if (!trimmed) return null;

  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (!isIsoTimestamp(parsed.capturedAt)) return null;
    if (typeof parsed.ticker !== 'string' || parsed.ticker.trim().length === 0) return null;
    if (typeof parsed.source !== 'string' || parsed.source.trim().length === 0) return null;
    if (!isIsoTimestamp(parsed.observationWindowStart) || !isIsoTimestamp(parsed.observationWindowEnd)) return null;
    if (!Array.isArray(parsed.transactions) || !parsed.transactions.every(isRawWhaleReplayTransaction)) return null;
    if (parsed.warnings !== undefined && !isStringArray(parsed.warnings)) return null;
    const capturedAt = parsed.capturedAt as string;
    const ticker = parsed.ticker as string;
    const source = parsed.source as string;
    const observationWindowStart = parsed.observationWindowStart as string;
    const observationWindowEnd = parsed.observationWindowEnd as string;
    const transactions = parsed.transactions as RawWhaleReplayTransaction[];
    const warnings = parsed.warnings as string[] | undefined;

    return {
      capturedAt,
      ticker,
      source,
      observationWindowStart,
      observationWindowEnd,
      transactions: transactions.map((transaction) => ({ ...transaction })),
      ...(warnings ? { warnings: [...warnings] } : {}),
    };
  } catch {
    return null;
  }
}

export function readRawWhaleReplayRows(
  filePath: string = DEFAULT_ARBITER_REPLAY_WHALE_RAW_PATH,
): RawWhaleReplayRow[] {
  return readJsonlRecords(filePath, parseRawWhaleReplayLine, 'arbiter-replay-whale');
}

export function appendRawWhaleReplayRows(
  records: RawWhaleReplayRow[],
  filePath: string = DEFAULT_ARBITER_REPLAY_WHALE_RAW_PATH,
): void {
  appendJsonlRecords(filePath, records);
}

export function appendRawWhaleReplayRow(
  record: RawWhaleReplayRow,
  filePath: string = DEFAULT_ARBITER_REPLAY_WHALE_RAW_PATH,
): void {
  appendRawWhaleReplayRows([record], filePath);
}

export function parseArbiterReplayBundleLine(rawLine: string): ArbiterReplayBundle | null {
  const trimmed = rawLine.trim();
  if (!trimmed) return null;

  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (!isIsoTimestamp(parsed.capturedAt)) return null;
    if (typeof parsed.ticker !== 'string' || parsed.ticker.trim().length === 0) return null;
    if (!Number.isInteger(parsed.horizonDays) || (parsed.horizonDays as number) < 1) return null;
    if (!(parsed.currentPrice === null || (isFiniteNumber(parsed.currentPrice) && parsed.currentPrice > 0))) return null;
    if (!isStringArray(parsed.warnings)) return null;
    if (parsed.leverage !== undefined && (!isFiniteNumber(parsed.leverage) || parsed.leverage <= 0)) return null;

    if (parsed.polymarket !== undefined && parsed.polymarket !== null) {
      if (!isRecord(parsed.polymarket)) return null;
      if (!isStringArray(parsed.polymarket.querySet) || !isStringArray(parsed.polymarket.selectedMarketIds)) return null;
      if (!Array.isArray(parsed.polymarket.selectedMarkets) || !parsed.polymarket.selectedMarkets.every(isReplayPolymarketMarket)) {
        return null;
      }
      if (parsed.polymarket.summary !== undefined && typeof parsed.polymarket.summary !== 'string') return null;
      if (!isOptionalNumber(parsed.polymarket.confidence) || !isOptionalNumber(parsed.polymarket.forecastReturn)) {
        return null;
      }
      if (!isOptionalNumber(parsed.polymarket.qualityScore)) return null;
      if (parsed.polymarket.qualityGrade !== undefined && typeof parsed.polymarket.qualityGrade !== 'string') return null;
      if (parsed.polymarket.semanticParserVersion !== undefined && typeof parsed.polymarket.semanticParserVersion !== 'string') {
        return null;
      }
      if (!isStringArray(parsed.polymarket.warnings)) return null;
    }

    if (parsed.whale !== undefined && parsed.whale !== null) {
      if (!isRecord(parsed.whale)) return null;
      if (typeof parsed.whale.source !== 'string' || parsed.whale.source.trim().length === 0) return null;
      if (!isIsoTimestamp(parsed.whale.observationWindowStart) || !isIsoTimestamp(parsed.whale.observationWindowEnd)) return null;
      if (parsed.whale.direction !== undefined && parsed.whale.direction !== 'long' && parsed.whale.direction !== 'short' && parsed.whale.direction !== 'neutral') {
        return null;
      }
      if (!isOptionalNumber(parsed.whale.confidence) || !isOptionalFiniteNumber(parsed.whale.notionalUsd)) return null;
      if (parsed.whale.summary !== undefined && typeof parsed.whale.summary !== 'string') return null;
      if (parsed.whale.txCount !== undefined && (!Number.isInteger(parsed.whale.txCount) || (parsed.whale.txCount as number) < 0)) {
        return null;
      }
      if (parsed.whale.txHashes !== undefined && !isStringArray(parsed.whale.txHashes)) return null;
    }

    if (parsed.labels !== undefined) {
      if (!isRecord(parsed.labels)) return null;
      if (parsed.labels.forecast !== undefined) {
        if (!isRecord(parsed.labels.forecast)) return null;
        if (!isFiniteNumber(parsed.labels.forecast.realizedPrice) || parsed.labels.forecast.realizedPrice <= 0) return null;
        if (!isFiniteNumber(parsed.labels.forecast.realizedReturn)) return null;
        if (parsed.labels.forecast.actualBinary !== 0 && parsed.labels.forecast.actualBinary !== 1) return null;
        if (!isIsoTimestamp(parsed.labels.forecast.labeledAt)) return null;
      }
      if (parsed.labels.semantic !== undefined) {
        if (!Array.isArray(parsed.labels.semantic) || !parsed.labels.semantic.every(isReplaySemanticLabel)) return null;
      }
    }

    const parsedPolymarket = parsed.polymarket as Record<string, unknown> | null | undefined;
    const parsedWhale = parsed.whale as Record<string, unknown> | null | undefined;
    const parsedLabels = parsed.labels as Record<string, unknown> | undefined;
    const parsedForecastLabel = parsedLabels?.forecast as Record<string, unknown> | undefined;
    const parsedSemanticLabels = parsedLabels?.semantic as ArbiterReplaySemanticLabel[] | undefined;

    const polymarket = parsedPolymarket === null
      ? null
      : parsedPolymarket
        ? {
            querySet: [...(parsedPolymarket.querySet as string[])],
            selectedMarketIds: [...(parsedPolymarket.selectedMarketIds as string[])],
            selectedMarkets: (parsedPolymarket.selectedMarkets as ArbiterReplayPolymarketMarket[]).map((market) => ({ ...market })),
            ...(parsedPolymarket.summary !== undefined ? { summary: parsedPolymarket.summary as string } : {}),
            ...(parsedPolymarket.confidence !== undefined ? { confidence: parsedPolymarket.confidence as number } : {}),
            ...(parsedPolymarket.forecastReturn !== undefined ? { forecastReturn: parsedPolymarket.forecastReturn as number } : {}),
            ...(parsedPolymarket.qualityScore !== undefined ? { qualityScore: parsedPolymarket.qualityScore as number } : {}),
            ...(parsedPolymarket.qualityGrade !== undefined ? { qualityGrade: parsedPolymarket.qualityGrade as string } : {}),
            ...(parsedPolymarket.semanticParserVersion !== undefined ? { semanticParserVersion: parsedPolymarket.semanticParserVersion as string } : {}),
            warnings: [...(parsedPolymarket.warnings as string[])],
          }
        : undefined;

    const whale = parsedWhale === null
      ? null
      : parsedWhale
        ? {
            source: parsedWhale.source as string,
            observationWindowStart: parsedWhale.observationWindowStart as string,
            observationWindowEnd: parsedWhale.observationWindowEnd as string,
            ...(parsedWhale.direction !== undefined ? { direction: parsedWhale.direction as 'long' | 'short' | 'neutral' } : {}),
            ...(parsedWhale.confidence !== undefined ? { confidence: parsedWhale.confidence as number } : {}),
            ...(parsedWhale.summary !== undefined ? { summary: parsedWhale.summary as string } : {}),
            ...(parsedWhale.txCount !== undefined ? { txCount: parsedWhale.txCount as number } : {}),
            ...(parsedWhale.notionalUsd !== undefined ? { notionalUsd: parsedWhale.notionalUsd as number | null } : {}),
            ...(parsedWhale.txHashes !== undefined ? { txHashes: [...(parsedWhale.txHashes as string[])] } : {}),
          }
        : undefined;

    const labels = parsedLabels
      ? {
          ...(parsedForecastLabel ? {
            forecast: {
              realizedPrice: parsedForecastLabel.realizedPrice as number,
              realizedReturn: parsedForecastLabel.realizedReturn as number,
              actualBinary: parsedForecastLabel.actualBinary as 0 | 1,
              labeledAt: parsedForecastLabel.labeledAt as string,
            },
          } : {}),
          ...(parsedSemanticLabels ? {
            semantic: parsedSemanticLabels.map((label) => ({
              marketId: label.marketId,
              semantics: label.semantics,
              outcome: label.outcome,
              labeledAt: label.labeledAt,
            })),
          } : {}),
        }
      : undefined;

    const capturedAt = parsed.capturedAt as string;
    const ticker = parsed.ticker as string;
    const horizonDays = parsed.horizonDays as number;
    const currentPrice = parsed.currentPrice as number | null;
    const warnings = parsed.warnings as string[];

    return {
      capturedAt,
      ticker,
      horizonDays,
      currentPrice,
      ...(parsed.leverage !== undefined ? { leverage: parsed.leverage as number } : {}),
      ...(parsed.markov !== undefined ? { markov: parsed.markov as ForecastArbiterInput['markov'] } : {}),
      ...(polymarket !== undefined ? { polymarket } : {}),
      ...(whale !== undefined ? { whale } : {}),
      ...(parsed.onchainContext !== undefined ? { onchainContext: parsed.onchainContext as ArbiterReplayBundle['onchainContext'] } : {}),
      warnings: [...warnings],
      ...(labels ? { labels } : {}),
    };
  } catch {
    return null;
  }
}

export function readArbiterReplayBundles(
  filePath: string = DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
): ArbiterReplayBundle[] {
  return readJsonlRecords(filePath, parseArbiterReplayBundleLine, 'arbiter-replay-bundle');
}

export function appendArbiterReplayBundles(
  records: ArbiterReplayBundle[],
  filePath: string = DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
): void {
  appendJsonlRecords(filePath, records);
}

export function appendArbiterReplayBundle(
  record: ArbiterReplayBundle,
  filePath: string = DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
): void {
  appendArbiterReplayBundles([record], filePath);
}

export interface ArbiterReplayCachePaths {
  bundlePath: string;
  polymarketRawPath: string;
  whaleRawPath: string;
}

export function getArbiterReplayCachePaths(
  overrides: Partial<ArbiterReplayCachePaths> = {},
): ArbiterReplayCachePaths {
  return {
    bundlePath: overrides.bundlePath ?? DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH,
    polymarketRawPath: overrides.polymarketRawPath ?? DEFAULT_ARBITER_REPLAY_CACHE_POLYMARKET_RAW_PATH,
    whaleRawPath: overrides.whaleRawPath ?? DEFAULT_ARBITER_REPLAY_CACHE_WHALE_RAW_PATH,
  };
}

export function appendReplayCachePolymarketCapture(
  capture: {
    rawRow: RawPolymarketReplayRow;
    polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
  },
  paths: Partial<ArbiterReplayCachePaths> = {},
): void {
  const resolved = getArbiterReplayCachePaths(paths);
  appendRawPolymarketReplayRow(capture.rawRow, resolved.polymarketRawPath);
  appendArbiterReplayBundle({
    capturedAt: capture.rawRow.capturedAt,
    ticker: capture.rawRow.ticker,
    horizonDays: capture.rawRow.horizonDays,
    currentPrice: capture.rawRow.currentPrice,
    polymarket: capture.polymarket,
    warnings: [...new Set(capture.polymarket.warnings)],
  }, resolved.bundlePath);
}

export function appendReplayCacheWhaleCapture(
  row: RawWhaleReplayRow,
  paths: Partial<ArbiterReplayCachePaths> = {},
): void {
  const resolved = getArbiterReplayCachePaths(paths);
  appendRawWhaleReplayRow(row, resolved.whaleRawPath);
}

export function appendReplayCacheBundle(
  bundle: ArbiterReplayBundle,
  paths: Partial<ArbiterReplayCachePaths> = {},
): void {
  const resolved = getArbiterReplayCachePaths(paths);
  appendArbiterReplayBundle(bundle, resolved.bundlePath);
}

export function createRawPolymarketReplayRow(params: {
  capturedAt: string;
  ticker: string;
  horizonDays: number;
  currentPrice: number | null;
  querySet: string[];
  selectedMarkets: PolymarketReplaySelectionMarket[];
  warnings?: string[];
}): RawPolymarketReplayRow {
  const selectedMarkets = params.selectedMarkets.filter(
    (market): market is PolymarketReplaySelectionMarket & { marketId: string } =>
      typeof market.marketId === 'string' && market.marketId.trim().length > 0,
  );

  return {
    capturedAt: params.capturedAt,
    ticker: params.ticker,
    horizonDays: params.horizonDays,
    currentPrice: params.currentPrice,
    querySet: [...new Set(params.querySet.map((entry) => entry.trim()).filter(Boolean))],
    selectedMarketIds: selectedMarkets.map((market) => market.marketId),
    candidates: selectedMarkets.map((market) => ({
      marketId: market.marketId,
      ...(market.assetId ? { assetId: market.assetId } : {}),
      question: market.question,
      probability: market.probability,
      volume24h: market.volume24h,
      ...(market.endDate !== undefined ? { endDate: market.endDate } : {}),
      ...(market.bid !== undefined ? { bid: market.bid } : {}),
      ...(market.ask !== undefined ? { ask: market.ask } : {}),
      ...(market.active !== undefined ? { active: market.active } : {}),
      ...(market.closed !== undefined ? { closed: market.closed } : {}),
      ...(market.enableOrderBook !== undefined ? { enableOrderBook: market.enableOrderBook } : {}),
    })),
    ...(params.warnings && params.warnings.length > 0 ? { warnings: [...params.warnings] } : {}),
  };
}

export function freezePolymarketReplayBlock(params: {
  querySet: string[];
  selectedMarkets: PolymarketReplaySelectionMarket[];
  warnings?: string[];
  summary?: string;
  confidence?: number;
  forecastReturn?: number;
  qualityScore?: number;
  qualityGrade?: string;
  semanticParserVersion?: string;
}): NonNullable<ArbiterReplayBundle['polymarket']> {
  const warnings = [...(params.warnings ?? [])];
  const selectedMarkets = params.selectedMarkets.flatMap((market) => {
    if (typeof market.marketId !== 'string' || market.marketId.trim().length === 0) {
      warnings.push(`Skipped Polymarket replay capture row with no marketId for question: ${market.question}`);
      return [];
    }
    if (typeof market.assetId !== 'string' || market.assetId.trim().length === 0) {
      warnings.push(`Missing CLOB token id for Polymarket market ${market.marketId}`);
      return [];
    }

    return [{
      marketId: market.marketId,
      assetId: market.assetId,
      question: market.question,
      probability: market.probability,
      volume24h: market.volume24h,
      endDate: market.endDate ?? '',
      semantics: classifyPolymarketQuestion(market.question),
      extractedPriceLevels: extractPriceLevels(market.question),
      ...(market.relevanceScore !== undefined ? { relevanceScore: market.relevanceScore } : {}),
      ...(market.bid !== undefined ? { bid: market.bid } : {}),
      ...(market.ask !== undefined ? { ask: market.ask } : {}),
    }];
  });

  return {
    querySet: [...new Set(params.querySet.map((entry) => entry.trim()).filter(Boolean))],
    selectedMarketIds: selectedMarkets.map((market) => market.marketId),
    selectedMarkets,
    ...(params.summary !== undefined ? { summary: params.summary } : {}),
    ...(params.confidence !== undefined ? { confidence: params.confidence } : {}),
    ...(params.forecastReturn !== undefined ? { forecastReturn: params.forecastReturn } : {}),
    ...(params.qualityScore !== undefined ? { qualityScore: params.qualityScore } : {}),
    ...(params.qualityGrade !== undefined ? { qualityGrade: params.qualityGrade } : {}),
    semanticParserVersion: params.semanticParserVersion ?? DEFAULT_POLYMARKET_SEMANTIC_PARSER_VERSION,
    warnings,
  };
}

export function createArbiterReplayBundleFromArbitratorInput(params: {
  capturedAt: string;
  input: ForecastArbiterInput;
  warnings?: string[];
}): ArbiterReplayBundle {
  const input = params.input;
  const warnings = [...(params.warnings ?? [])];
  const ticker = input.ticker.trim().toUpperCase();
  const polymarketMarkets = (input.polymarket?.markets ?? []).map((market) => ({
    marketId: market.marketId,
    assetId: market.assetId,
    question: market.question,
    probability: market.probability ?? 0.5,
    volume24h: market.volume24h ?? 0,
    endDate: market.endDate,
    bid: market.bid,
    ask: market.ask,
  }));
  const polymarket = input.polymarket
    ? freezePolymarketReplayBlock({
        querySet: input.polymarket.querySet ?? [],
        selectedMarkets: polymarketMarkets,
        warnings,
        summary: input.polymarket.summary,
        confidence: input.polymarket.confidence,
        forecastReturn: input.polymarket.forecast_return,
        qualityScore: input.polymarket.quality_score,
        qualityGrade: input.polymarket.quality_grade,
      })
    : undefined;

  const whale = input.whale
    ? {
        source: input.whale.source ?? 'tool-input',
        observationWindowStart: input.whale.observationWindowStart ?? params.capturedAt,
        observationWindowEnd: input.whale.observationWindowEnd ?? params.capturedAt,
        ...(input.whale.direction !== undefined ? { direction: input.whale.direction } : {}),
        ...(input.whale.confidence !== undefined ? { confidence: input.whale.confidence } : {}),
        ...(input.whale.summary !== undefined ? { summary: input.whale.summary } : {}),
        ...(input.whale.txCount !== undefined ? { txCount: input.whale.txCount } : {}),
        ...(input.whale.notionalUsd !== undefined ? { notionalUsd: input.whale.notionalUsd } : {}),
        ...(input.whale.txHashes !== undefined ? { txHashes: [...input.whale.txHashes] } : {}),
      }
    : undefined;

  return {
    capturedAt: params.capturedAt,
    ticker,
    horizonDays: input.horizon_days,
    currentPrice: input.current_price ?? null,
    ...(input.leverage !== undefined ? { leverage: input.leverage } : {}),
    ...(input.markov !== undefined ? { markov: input.markov } : {}),
    ...(polymarket !== undefined ? { polymarket } : {}),
    ...(whale !== undefined ? { whale } : {}),
    warnings,
  };
}

function classifyOwnerBucket(owner: string | undefined): 'exchange' | 'accumulation' | 'other' {
  if (!owner) return 'other';
  const lower = owner.toLowerCase();
  if (KNOWN_EXCHANGE_MARKERS.some((marker) => lower.includes(marker))) return 'exchange';
  if (KNOWN_ACCUMULATION_MARKERS.some((marker) => lower.includes(marker))) return 'accumulation';
  return 'other';
}

export function normalizeWhaleReplayRow(
  row: RawWhaleReplayRow,
): NonNullable<ArbiterReplayBundle['whale']> {
  let longUsd = 0;
  let shortUsd = 0;
  let unresolvedUsd = 0;

  for (const tx of row.transactions) {
    const valueUsd = tx.valueUsd ?? 0;
    if (!Number.isFinite(valueUsd) || valueUsd <= 0) continue;

    const fromBucket = classifyOwnerBucket(tx.fromOwner);
    const toBucket = classifyOwnerBucket(tx.toOwner);

    if (fromBucket === 'exchange' && toBucket !== 'exchange') {
      longUsd += valueUsd;
      continue;
    }
    if (toBucket === 'exchange' && fromBucket !== 'exchange') {
      shortUsd += valueUsd;
      continue;
    }
    unresolvedUsd += valueUsd;
  }

  const directionalUsd = longUsd + shortUsd;
  const totalUsd = directionalUsd + unresolvedUsd;
  const imbalance = directionalUsd > 0 ? Math.abs(longUsd - shortUsd) / directionalUsd : 0;
  const confidence = directionalUsd > 0
    ? Math.min(0.9, 0.4 + imbalance * 0.45)
    : 0.35;
  const direction = directionalUsd === 0
    ? 'neutral'
    : longUsd > shortUsd * 1.1
      ? 'long'
      : shortUsd > longUsd * 1.1
        ? 'short'
        : 'neutral';
  const summary = direction === 'long'
    ? `Whale flow tilts bullish: ${Math.round(longUsd).toLocaleString('en-US')} USD left exchanges versus ${Math.round(shortUsd).toLocaleString('en-US')} USD sent to exchanges.`
    : direction === 'short'
      ? `Whale flow tilts bearish: ${Math.round(shortUsd).toLocaleString('en-US')} USD moved onto exchanges versus ${Math.round(longUsd).toLocaleString('en-US')} USD withdrawn.`
      : directionalUsd > 0
        ? `Whale flow is mixed: ${Math.round(longUsd).toLocaleString('en-US')} USD in bullish withdrawals versus ${Math.round(shortUsd).toLocaleString('en-US')} USD in bearish deposits.`
        : 'Whale flow is neutral: no exchange-linked directional transfers were captured.';

  return {
    source: row.source,
    direction,
    confidence,
    summary,
    observationWindowStart: row.observationWindowStart,
    observationWindowEnd: row.observationWindowEnd,
    txCount: row.transactions.length,
    notionalUsd: totalUsd > 0 ? totalUsd : null,
    txHashes: row.transactions.map((tx) => tx.hash),
  };
}

export function createRawWhaleReplayRowFromToolResult(params: {
  capturedAt: string;
  ticker: string;
  whale: Record<string, unknown>;
  warnings?: string[];
}): RawWhaleReplayRow | null {
  const whale = params.whale;
  const source = asString(whale.source) ?? 'unknown';
  const candidates = Array.isArray(whale.transactions)
    ? whale.transactions
    : Array.isArray(whale.recent_large_transactions)
      ? whale.recent_large_transactions
      : [];

  const transactions = candidates.flatMap((candidate) => {
    if (!isRecord(candidate)) return [];
    const hash = asString(candidate.hash)
      ?? asString(candidate.id)
      ?? asString(candidate.transaction_id)
      ?? asString(candidate.txid);
    const timestamp = asIsoFromUnknownTimestamp(candidate.timestamp)
      ?? asIsoFromUnknownTimestamp(candidate.time)
      ?? asIsoFromUnknownTimestamp(candidate.transaction_time);
    if (!hash || !timestamp) return [];

    const valueUsd = asFiniteNumber(candidate.valueUsd)
      ?? asFiniteNumber(candidate.value_usd)
      ?? asFiniteNumber(candidate.usd_value)
      ?? asFiniteNumber(candidate.amount_usd);
    const symbol = asString(candidate.symbol) ?? params.ticker.trim().toUpperCase();
    const fromOwner = ownerLabel(candidate.fromOwner) ?? ownerLabel(candidate.from);
    const toOwner = ownerLabel(candidate.toOwner) ?? ownerLabel(candidate.to);

    return [{
      hash,
      timestamp,
      ...(symbol ? { symbol } : {}),
      ...(valueUsd !== undefined ? { valueUsd } : {}),
      ...(fromOwner ? { fromOwner } : {}),
      ...(toOwner ? { toOwner } : {}),
    }];
  });

  if (transactions.length === 0) return null;
  const timestamps = transactions.map((transaction) => Date.parse(transaction.timestamp)).filter(Number.isFinite);
  const observationWindowStart = new Date(Math.min(...timestamps)).toISOString();
  const observationWindowEnd = new Date(Math.max(...timestamps)).toISOString();

  return {
    capturedAt: params.capturedAt,
    ticker: params.ticker.trim().toUpperCase(),
    source,
    observationWindowStart,
    observationWindowEnd,
    transactions,
    ...(params.warnings && params.warnings.length > 0 ? { warnings: [...params.warnings] } : {}),
  };
}

export function toForecastArbiterInput(bundle: ArbiterReplayBundle): ForecastArbiterInput {
  return {
    ticker: bundle.ticker,
    horizon_days: bundle.horizonDays,
    current_price: bundle.currentPrice ?? undefined,
    leverage: bundle.leverage,
    markov: bundle.markov ? { ...bundle.markov } : undefined,
    polymarket: bundle.polymarket ? {
      forecast_return: bundle.polymarket.forecastReturn,
      confidence: bundle.polymarket.confidence,
      quality_score: bundle.polymarket.qualityScore,
      quality_grade: bundle.polymarket.qualityGrade,
      markets: bundle.polymarket.selectedMarkets.map((market) => ({
        question: market.question,
        probability: market.probability,
        semantics: market.semantics,
        price: market.extractedPriceLevels[0],
      })),
      summary: bundle.polymarket.summary,
    } : undefined,
    whale: bundle.whale ? {
      direction: bundle.whale.direction,
      confidence: bundle.whale.confidence,
      summary: bundle.whale.summary,
    } : undefined,
  };
}
