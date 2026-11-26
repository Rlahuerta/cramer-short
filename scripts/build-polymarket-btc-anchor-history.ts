/**
 * Build fixtures/polymarket-btc-anchor-history.json: resolved "Bitcoin above/below $X on <date>" Polymarket markets
 * inside the price fixture's window, each with one closing YES probability per UTC day from the public CLOB price
 * history. Public, keyless endpoints only. `asset` is the Gamma tag and the word the questions must contain
 * (bitcoin, ethereum).
 *
 *   bun run scripts/build-polymarket-btc-anchor-history.ts [from=2024-01] [to=2025-12] [out] [asset=bitcoin]
 */
import { join } from 'node:path';
import { sampleDailyCloses, type AnchorHistoryFixture, type AnchorHistoryMarket } from '../src/tools/finance/backtest/polymarket-anchor-history.js';

const GAMMA = 'https://gamma-api.polymarket.com';
const CLOB = 'https://clob.polymarket.com';

interface GammaMarket {
  question?: string;
  createdAt?: string;
  endDate?: string;
  volume?: string | number;
  outcomes?: string;
  clobTokenIds?: string;
}

// Gamma returns at most 100 events per page and stops paging near offset 2,000; the bitcoin tag carries ~100
// short-term markets a day, so a month-wide window lost most of its markets. One day per window stays under the cap.
const PAGE = 100;

export function dayWindows(from: string, to: string): Array<[string, string]> {
  const windows: Array<[string, string]> = [];
  const last = new Date(`${to}-01T00:00:00Z`);
  last.setUTCMonth(last.getUTCMonth() + 1);
  for (let d = new Date(`${from}-01T00:00:00Z`); d < last; d.setUTCDate(d.getUTCDate() + 1)) {
    const next = new Date(d.getTime() + 86_400_000);
    windows.push([d.toISOString().slice(0, 10), next.toISOString().slice(0, 10)]);
  }
  return windows;
}

export async function closedMarkets(tag: string, start: string, end: string, fetchFn: typeof fetch = fetch): Promise<GammaMarket[]> {
  const markets: GammaMarket[] = [];
  for (let offset = 0; ; offset += PAGE) {
    const res = await fetchFn(`${GAMMA}/events?closed=true&tag_slug=${tag}&end_date_min=${start}&end_date_max=${end}&limit=${PAGE}&offset=${offset}`);
    if (!res.ok) throw new Error(`gamma ${res.status} for ${start}..${end} offset ${offset}`);
    const events = await res.json() as Array<{ markets?: GammaMarket[] }>;
    for (const event of events) markets.push(...(event.markets ?? []));
    if (events.length < PAGE) return markets;
  }
}

function yesToken(market: GammaMarket): string | undefined {
  const tokens = JSON.parse(market.clobTokenIds ?? '[]') as string[];
  const outcomes = JSON.parse(market.outcomes ?? '["Yes","No"]') as string[];
  const yes = outcomes.findIndex(o => o.toLowerCase() === 'yes');
  return tokens[yes >= 0 ? yes : 0];
}

async function history(token: string, createdAt: string, endDate: string): Promise<Array<{ t: number; p: number }>> {
  const startTs = Math.floor(Date.parse(createdAt) / 1000);
  const endTs = Math.floor(Date.parse(endDate) / 1000);
  const res = await fetch(`${CLOB}/prices-history?market=${token}&startTs=${startTs}&endTs=${endTs}&fidelity=60`);
  if (!res.ok) return [];
  return ((await res.json()) as { history?: Array<{ t: number; p: number }> }).history ?? [];
}

if (import.meta.main) {
  const [from = '2024-01', to = '2025-12', out = join(import.meta.dir, '..', 'src/tools/finance/fixtures/polymarket-btc-anchor-history.json'), asset = 'bitcoin'] = process.argv.slice(2);
  const assetPattern = new RegExp(asset, 'i');
  const seen = new Set<string>();
  const markets: AnchorHistoryMarket[] = [];
  for (const [start, end] of dayWindows(from, to)) {
    const candidates = (await closedMarkets(asset, start, end)).filter(m =>
      m.question && assetPattern.test(m.question) && /\b(?:above|below)\b/i.test(m.question) && m.createdAt && m.endDate);
    for (const m of candidates) {
      const key = `${m.question}|${m.endDate}`;
      const token = yesToken(m);
      if (seen.has(key) || !token) continue;
      seen.add(key);
      const daily = sampleDailyCloses(await history(token, m.createdAt!, m.endDate!), m.endDate!);
      if (Object.keys(daily).length > 0) {
        markets.push({ question: m.question!, createdAt: m.createdAt!, endDate: m.endDate!, volume: Number(m.volume ?? 0), daily });
      }
    }
    if (end.endsWith('-01')) console.log(`through ${start}: ${markets.length} markets`);
  }
  markets.sort((a, b) => a.endDate.localeCompare(b.endDate));
  const fixture: AnchorHistoryFixture = {
    source: `Polymarket Gamma closed events (tag ${asset}) + CLOB prices-history, YES token, daily close at 23:59:59Z`,
    builtAt: new Date().toISOString(),
    markets,
  };
  await Bun.write(out, JSON.stringify(fixture, null, 1));
  console.log(`${markets.length} markets -> ${out}`);
}
