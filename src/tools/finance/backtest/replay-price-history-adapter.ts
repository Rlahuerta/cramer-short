import { readFileSync } from 'node:fs';
import type { ReplayPriceHistory, ReplayPricePoint } from '../arbiter-replay-labeler.js';
import type { HistoryProvider } from './replay-label-runner.js';

export const DEFAULT_REPLAY_PRICE_FIXTURE_PATH = new URL('../fixtures/backtest-prices.json', import.meta.url);

export interface ReplayFixtureTickerSeries {
  type: 'stock' | 'etf' | 'crypto';
  closes: number[];
  dates: string[];
  count: number;
  synthetic?: boolean;
}

export interface ReplayFixturePriceStore {
  generatedAt?: string;
  startDate?: string;
  endDate?: string;
  syntheticNote?: string;
  tickers: Record<string, ReplayFixtureTickerSeries>;
}

type ReplayHistoryBar = {
  at?: string;
  date?: string;
  time?: string;
  price?: number;
  close?: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function normalizeTimestamp(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : null;
}

function normalizePoint(at: unknown, price: unknown): ReplayPricePoint | null {
  const normalizedAt = normalizeTimestamp(at);
  if (!normalizedAt || typeof price !== 'number' || !Number.isFinite(price) || price <= 0) {
    return null;
  }

  return { at: normalizedAt, price };
}

function sortPoints(points: ReplayPricePoint[]): ReplayPricePoint[] {
  return [...points].sort((a, b) => {
    const timestampDelta = Date.parse(a.at) - Date.parse(b.at);
    return timestampDelta !== 0 ? timestampDelta : a.price - b.price;
  });
}

function hasReplayPoints(value: unknown): value is ReplayPriceHistory {
  return isRecord(value) && Array.isArray(value.points);
}

function hasFixtureSeries(value: unknown): value is ReplayFixtureTickerSeries {
  return isRecord(value) && Array.isArray(value.dates) && Array.isArray(value.closes);
}

export function normalizeReplayPriceHistory(
  source: ReplayPriceHistory | ReplayFixtureTickerSeries | ReplayHistoryBar[],
): ReplayPriceHistory | null {
  if (hasReplayPoints(source)) {
    const points = sortPoints(
      source.points
        .map((point) => normalizePoint(point.at, point.price))
        .filter((point): point is ReplayPricePoint => point !== null),
    );
    return points.length > 0 ? { points } : null;
  }

  if (Array.isArray(source)) {
    const points = sortPoints(
      source
        .map((bar) => normalizePoint(bar.time ?? bar.date ?? bar.at, bar.close ?? bar.price))
        .filter((point): point is ReplayPricePoint => point !== null),
    );
    return points.length > 0 ? { points } : null;
  }

  if (hasFixtureSeries(source)) {
    if (source.dates.length !== source.closes.length || source.dates.length === 0) {
      return null;
    }

    const points = sortPoints(
      source.dates
        .map((date, index) => normalizePoint(date, source.closes[index]))
        .filter((point): point is ReplayPricePoint => point !== null),
    );
    return points.length > 0 ? { points } : null;
  }

  return null;
}

function readReplayFixturePriceStore(fixturePath: string | URL): ReplayFixturePriceStore {
  const parsed = JSON.parse(readFileSync(fixturePath, 'utf-8')) as unknown;
  if (!isRecord(parsed) || !isRecord(parsed.tickers)) {
    throw new Error('replay-price-history-adapter: fixture store must contain a tickers object.');
  }

  return parsed as unknown as ReplayFixturePriceStore;
}

function tickerCandidates(ticker: string): string[] {
  const normalized = ticker.trim().toUpperCase();
  if (normalized.length === 0) return [];

  const candidates = [normalized];
  if (normalized.endsWith('-USD')) {
    candidates.push(normalized.slice(0, -4));
  } else {
    candidates.push(`${normalized}-USD`);
  }

  return [...new Set(candidates)];
}

function resolveFixtureTicker(
  ticker: string,
  fixture: ReplayFixturePriceStore,
): string | null {
  for (const candidate of tickerCandidates(ticker)) {
    if (fixture.tickers[candidate]) {
      return candidate;
    }
  }

  return null;
}

export function createReplayPriceHistoryProvider(params: {
  fixture?: ReplayFixturePriceStore;
  fixturePath?: string | URL;
} = {}): HistoryProvider {
  const fixture = params.fixture ?? readReplayFixturePriceStore(
    params.fixturePath ?? DEFAULT_REPLAY_PRICE_FIXTURE_PATH,
  );

  return (ticker, bundle) => {
    const resolvedTicker = resolveFixtureTicker(ticker || bundle.ticker, fixture);
    if (!resolvedTicker) return null;
    return normalizeReplayPriceHistory(fixture.tickers[resolvedTicker] ?? null);
  };
}
