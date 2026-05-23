import { api } from '../tools/finance/api.js';
import type { PriceFetcher, PriceSnapshot } from './watchlist-display.js';

export const makePriceFetcher = (): PriceFetcher => async (ticker: string) => {
  try {
    const { data } = await api.get('/prices/snapshot/', { ticker });
    const snap = data.snapshot as Record<string, unknown> | undefined;
    if (!snap) return null;
    const price = Number(snap.price ?? snap.close ?? 0);
    if (!price) return null;
    return {
      ticker,
      price,
      changePercent: Number(snap.change_percent ?? snap.percent_change ?? 0),
      high52Week:    snap.week_52_high !== undefined ? Number(snap.week_52_high) : undefined,
      low52Week:     snap.week_52_low  !== undefined ? Number(snap.week_52_low)  : undefined,
      marketCap:     snap.market_cap   !== undefined ? Number(snap.market_cap)   : undefined,
      name:          typeof snap.name === 'string' ? snap.name : undefined,
    } satisfies PriceSnapshot;
  } catch {
    return null;
  }
};

export const fetchShowData = async (ticker: string): Promise<PriceSnapshot | null> => {
  const fetcher = makePriceFetcher();
  const base = await fetcher(ticker);
  if (!base) return null;

  // Fetch ratios + analyst targets + news + Markov confidence in parallel
  const [ratiosResult, analystResult, newsResult, markovResult] = await Promise.allSettled([
    api.get('/financial-metrics/snapshot/', { ticker }),
    // Yahoo Finance analyst targets via env-conditional endpoint
    (async () => {
      const yahooFd = await import('../tools/finance/yahoo-finance.js');
      const result = await yahooFd.getYahooAnalystTargets.invoke({ ticker });
      return typeof result === 'string' ? JSON.parse(result) : result;
    })(),
    api.get('/news', { ticker, limit: 3 }),
    (async () => {
      const markov = await import('../tools/finance/markov-distribution.js');
      const result = await markov.markovDistributionTool.invoke({ ticker, horizon: 14 });
      return typeof result === 'string' ? JSON.parse(result) : result;
    })(),
  ]);

  const snap: PriceSnapshot = { ...base };

  if (ratiosResult.status === 'fulfilled') {
    const r = (ratiosResult.value.data.snapshot ?? {}) as Record<string, unknown>;
    if (r.price_to_earnings !== undefined) snap.pe = Number(r.price_to_earnings);
    if (r.price_to_book     !== undefined) snap.pb = Number(r.price_to_book);
    if (r.ev_to_ebitda      !== undefined) snap.evEbitda = Number(r.ev_to_ebitda);
    if (r.peg_ratio         !== undefined) snap.peg = Number(r.peg_ratio);
  }

  if (analystResult.status === 'fulfilled') {
    const a = analystResult.value as Record<string, unknown>;
    if (a.recommendationKey) snap.analystRating  = String(a.recommendationKey);
    if (a.targetMeanPrice)   snap.analystAvgTarget = Number(a.targetMeanPrice);
  }

  if (newsResult.status === 'fulfilled') {
    const items = (newsResult.value.data.news as unknown[]) ?? [];
    snap.news = items.slice(0, 3).map((n) => {
      const item = n as Record<string, unknown>;
      return {
        title:  String(item.title ?? ''),
        date:   String(item.date ?? item.published_at ?? ''),
        source: typeof item.source === 'string' ? item.source : undefined,
      };
    });
  }

  if (markovResult.status === 'fulfilled') {
    const m = markovResult.value as { diagnostics?: { predictionConfidence?: number } };
    if (m.diagnostics?.predictionConfidence !== undefined) {
      snap.predictionConfidence = m.diagnostics.predictionConfidence;
    }
  }

  return snap;
};
