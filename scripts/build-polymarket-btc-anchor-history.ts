/**
 * Build fixtures/polymarket-btc-anchor-history.json: resolved "Bitcoin above/below $X on <date>" Polymarket markets
 * inside the price fixture's window, each with one closing YES probability per UTC day from the public CLOB price
 * history. Public, keyless endpoints only.
 *
 *   bun run scripts/build-polymarket-btc-anchor-history.ts [from=2024-01] [to=2025-12] [out]
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

function monthWindows(from: string, to: string): Array<[string, string]> {
  const windows: Array<[string, string]> = [];
  for (let d = new Date(`${from}-01T00:00:00Z`); d <= new Date(`${to}-01T00:00:00Z`); d.setUTCMonth(d.getUTCMonth() + 1)) {
    const end = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0));
    windows.push([d.toISOString().slice(0, 10), end.toISOString().slice(0, 10)]);
  }
  return windows;
}

async function closedBitcoinMarkets(start: string, end: string): Promise<GammaMarket[]> {
  const markets: GammaMarket[] = [];
  for (let offset = 0; ; offset += 500) {
    const res = await fetch(`${GAMMA}/events?closed=true&tag_slug=bitcoin&end_date_min=${start}&end_date_max=${end}&limit=500&offset=${offset}`);
    if (!res.ok) throw new Error(`gamma ${res.status} for ${start}..${end} offset ${offset}`);
    const events = await res.json() as Array<{ markets?: GammaMarket[] }>;
    for (const event of events) markets.push(...(event.markets ?? []));
    if (events.length < 500) return markets;
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
  const [from = '2024-01', to = '2025-12', out = join(import.meta.dir, '..', 'src/tools/finance/fixtures/polymarket-btc-anchor-history.json')] = process.argv.slice(2);
  const seen = new Set<string>();
  const markets: AnchorHistoryMarket[] = [];
  for (const [start, end] of monthWindows(from, to)) {
    const candidates = (await closedBitcoinMarkets(start, end)).filter(m =>
      m.question && /bitcoin/i.test(m.question) && /\b(?:above|below)\b/i.test(m.question) && m.createdAt && m.endDate);
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
    console.log(`${start}..${end}: ${candidates.length} candidates, ${markets.length} markets so far`);
  }
  markets.sort((a, b) => a.endDate.localeCompare(b.endDate));
  const fixture: AnchorHistoryFixture = {
    source: 'Polymarket Gamma closed events (tag bitcoin) + CLOB prices-history, YES token, daily close at 23:59:59Z',
    builtAt: new Date().toISOString(),
    markets,
  };
  await Bun.write(out, JSON.stringify(fixture, null, 1));
  console.log(`${markets.length} markets -> ${out}`);
}
