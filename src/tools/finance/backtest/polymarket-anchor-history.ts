/**
 * Historical Polymarket anchors for an anchored walk-forward replay: resolved price-threshold markets with one
 * closing probability per UTC day, reconstructed from the public CLOB price history
 * (scripts/build-polymarket-btc-anchor-history.ts writes fixtures/polymarket-btc-anchor-history.json).
 */
import { MS_PER_DAY } from '../../../utils/time.js';
import type { WalkForwardConfig } from './walk-forward.js';

export interface AnchorHistoryMarket {
  question: string;
  createdAt: string;
  endDate: string;
  /** Lifetime volume: the history has no daily volume, and trust only needs it to be positive. */
  volume: number;
  /** UTC date (YYYY-MM-DD) -> the market's last YES price at or before 23:59:59Z that day. */
  daily: Record<string, number>;
}

export interface AnchorHistoryFixture {
  source: string;
  builtAt: string;
  markets: AnchorHistoryMarket[];
}

const dayKey = (ms: number) => new Date(ms).toISOString().slice(0, 10);
const dayClose = (date: string) => Date.parse(`${date}T23:59:59Z`);

/**
 * One closing probability per UTC day, from the first traded day to the day before resolution (a close on the
 * resolution day would already know the outcome). Days without a trade carry the last close over.
 */
export function sampleDailyCloses(history: ReadonlyArray<{ t: number; p: number }>, endDate: string): Record<string, number> {
  if (history.length === 0) return {};
  const points = [...history].sort((a, b) => a.t - b.t);
  const lastDay = dayKey(Date.parse(endDate) - MS_PER_DAY);
  const daily: Record<string, number> = {};
  let i = 0;
  let close: number | undefined;
  for (let day = dayKey(points[0].t * 1000); day <= lastDay; day = dayKey(Date.parse(`${day}T00:00:00Z`) + MS_PER_DAY)) {
    while (i < points.length && points[i].t * 1000 <= dayClose(day)) close = points[i++].p;
    if (close !== undefined) daily[day] = close;
  }
  return daily;
}

/** The walk-forward `anchorsAt` hook: the markets open at each origin's daily close, with that day's probability. */
export function anchorsAtFromHistory(
  dates: readonly string[],
  markets: readonly AnchorHistoryMarket[],
): NonNullable<WalkForwardConfig['anchorsAt']> {
  return (t) => {
    const date = dates[t];
    const referenceTimeMs = dayClose(date);
    return {
      referenceTimeMs,
      markets: markets
        .filter(m => m.daily[date] !== undefined && Date.parse(m.createdAt) <= referenceTimeMs && Date.parse(m.endDate) > referenceTimeMs)
        .map(m => ({ question: m.question, probability: m.daily[date], volume: m.volume, createdAt: m.createdAt, endDate: m.endDate })),
    };
  };
}
