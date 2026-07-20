import { MS_PER_DAY } from '../../utils/time.js';
import { fetchClobPriceHistory, fetchClobSpread } from './polymarket-clob.js';
import type {
  GammaDateFilter,
  NormalizedPolymarketMarket,
  NormalizedPolymarketOutcome,
  PolymarketReadDriver,
  PolymarketSearchInput,
  PolymarketSearchResult,
} from './polymarket-types.js';

const GAMMA_BASE = 'https://gamma-api.polymarket.com';
const TEXT_FILTER_STOP_WORDS = new Set([
  'the', 'and', 'for', 'are', 'not', 'will', 'can', 'has', 'was',
  'how', 'what', 'that', 'this', 'its', 'from', 'with',
]);

const WEAK_QUERY_WORDS = new Set([
  'price', 'prices', 'market', 'markets', 'commodity', 'commodities',
  'forecast', 'forecasts', 'current', 'target', 'targets',
]);

export let RETRY_DELAYS = [1_000, 2_000, 4_000];

export function setRetryDelays(delays: number[]): void {
  RETRY_DELAYS = delays;
}

export interface GammaMarket {
  id: string;
  conditionId?: string;
  question: string;
  outcomes: string | readonly string[];
  outcomePrices: string | readonly string[];
  clobTokenIds?: string | readonly string[];
  endDateIso?: string;
  createdAt?: string;
  volume24hr?: number;
  liquidityNum?: number;
  active: boolean;
  closed: boolean;
  enableOrderBook?: boolean;
  description?: string;
}

export interface GammaEvent {
  id: string;
  title: string;
  endDate?: string;
  markets?: GammaMarket[];
  volume24hr?: number;
}

export async function fetchWithRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  delays: number[] = RETRY_DELAYS,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (err instanceof Error) {
        const match = /\b(\d{3})\b/.exec(err.message);
        const status = match ? Number(match[1]) : Number.NaN;
        if (Number.isFinite(status) && status >= 400 && status < 500 && status !== 429) {
          throw err;
        }
      }
      if (attempt < maxRetries) {
        const delay = delays[attempt] ?? 4_000;
        if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }
  throw lastError;
}

function parseStringArrayField(raw: string | readonly string[] | undefined): string[] {
  const parsed = typeof raw === 'string'
    ? (() => {
        try {
          return JSON.parse(raw) as unknown;
        } catch {
          return [];
        }
      })()
    : raw;

  return Array.isArray(parsed)
    ? parsed
        .filter((item): item is string | number => typeof item === 'string' || typeof item === 'number')
        .map((item) => String(item))
    : [];
}

function computeAgeDays(createdAt: string | undefined): number | undefined {
  if (!createdAt) return undefined;
  const ms = Date.now() - new Date(createdAt).getTime();
  if (Number.isNaN(ms) || ms < 0) return undefined;
  return Math.floor(ms / MS_PER_DAY);
}

function toOutcome(
  label: string,
  rawProbability: string | undefined,
  rawTokenId: string | undefined,
): NormalizedPolymarketOutcome {
  const probability = Math.min(1, Math.max(0, Number.parseFloat(rawProbability ?? '0')));
  return {
    label,
    probability,
    ...(rawTokenId ? { tokenId: rawTokenId } : {}),
  };
}

export function normalizeGammaMarket(market: GammaMarket): NormalizedPolymarketMarket | null {
  const outcomes = parseStringArrayField(market.outcomes);
  const prices = parseStringArrayField(market.outcomePrices);
  if (outcomes.length === 0 || prices.length === 0) return null;

  const tokenIds = parseStringArrayField(market.clobTokenIds);
  const mappedOutcomes = outcomes.reduce<NormalizedPolymarketMarket['outcomes']>((acc, outcome, index) => {
    const normalized = outcome.trim().toLowerCase();
    if (normalized === 'yes') {
      acc.yes = toOutcome(outcome, prices[index], tokenIds[index]);
    } else if (normalized === 'no') {
      acc.no = toOutcome(outcome, prices[index], tokenIds[index]);
    }
    return acc;
  }, {});

  return {
    marketId: market.conditionId ?? market.id,
    question: market.question,
    outcomes: mappedOutcomes,
    ...(tokenIds[0] ? { primaryYesTokenId: tokenIds[0] } : {}),
    endDate: market.endDateIso ?? null,
    volume24h: typeof market.volume24hr === 'number' && Number.isFinite(market.volume24hr) ? market.volume24hr : 0,
    liquidity: typeof market.liquidityNum === 'number' && Number.isFinite(market.liquidityNum) ? market.liquidityNum : 0,
    ageDays: computeAgeDays(market.createdAt),
    active: market.active,
    closed: market.closed,
    ...(market.enableOrderBook !== undefined ? { enableOrderBook: market.enableOrderBook } : {}),
    ...(market.description ? { description: market.description } : {}),
  };
}

export function questionMatchesQuery(text: string, query: string): boolean {
  const words = query
    .toLowerCase()
    .split(/[\s\-_/]+/)
    .map((word) => word.replace(/[^a-z0-9]/g, ''))
    .filter((word) => word.length >= 3 && !TEXT_FILTER_STOP_WORDS.has(word));

  if (words.length === 0) return true;

  const anchorWords = words.filter((word) => !WEAK_QUERY_WORDS.has(word));
  if (words.length > 0 && anchorWords.length === 0) return false;

  const candidateWords = anchorWords.length > 0 ? anchorWords : words;
  const lower = text.toLowerCase();
  return candidateWords.some((word) => lower.includes(word));
}

const TAG_SLUG_PATTERNS: Array<{ patterns: string[]; slugs: string[] }> = [
  { patterns: ['bitcoin', 'btc'], slugs: ['bitcoin', 'crypto-prices', 'crypto'] },
  { patterns: ['ethereum', 'eth'], slugs: ['ethereum', 'crypto-prices', 'crypto'] },
  { patterns: ['solana', 'sol', 'crypto', 'defi', 'nft', 'web3'], slugs: ['crypto-prices', 'crypto'] },
  { patterns: ['fed', 'fomc', 'federal reserve', 'rate cut', 'rate hike', 'interest rate', 'basis point'], slugs: ['fed-rates', 'fed', 'economic-policy'] },
  { patterns: ['recession', 'gdp', 'inflation', 'cpi', 'unemployment', 'economic'], slugs: ['economy', 'business', 'economic-policy'] },
  { patterns: ['tariff', 'trade war', 'trade deal', 'import duty'], slugs: ['tariffs', 'politics', 'world'] },
  { patterns: ['oil', 'opec', 'crude', 'energy', 'wti', 'brent', 'petroleum'], slugs: ['commodities', 'world', 'business'] },
  { patterns: ['gold', 'silver', 'copper', 'platinum', 'palladium', 'precious metal', 'metal'], slugs: ['commodities', 'business'] },
  { patterns: ['wheat', 'corn', 'soybean', 'coffee', 'sugar', 'grain', 'natural gas'], slugs: ['commodities'] },
  { patterns: ['fda', 'drug approval', 'clinical trial', 'pharma', 'pfizer', 'moderna', 'eli lilly'], slugs: ['science', 'health'] },
  { patterns: ['nvidia', 'apple', 'microsoft', 'google', 'amazon', 'meta', 'tesla', 'broadcom', 'qualcomm', 'intel', 'spacex'], slugs: ['big-tech', 'tech', 'business'] },
  { patterns: ['earnings', 'revenue', 'eps', 'quarterly results'], slugs: ['business', 'finance'] },
  { patterns: ['ai regulation', 'artificial intelligence', 'chatgpt', 'openai', 'antitrust'], slugs: ['tech', 'science'] },
  { patterns: ['middle east', 'ukraine', 'russia', 'china', 'taiwan', 'war', 'conflict', 'sanctions', 'geopolitical'], slugs: ['world', 'politics'] },
  { patterns: ['election', 'president', 'senate', 'congress', 'trump', 'white house'], slugs: ['elections', 'us-politics', 'politics'] },
  { patterns: ['ipo', 'initial public offering'], slugs: ['ipos', 'ipo', 'business'] },
];

export function inferTagSlugs(query: string): string[] {
  const lower = query.toLowerCase();
  for (const { patterns, slugs } of TAG_SLUG_PATTERNS) {
    if (patterns.some((pattern) => lower.includes(pattern))) return slugs;
  }
  return [];
}

function buildEventSearchParams(
  limit: number,
  offset: number,
  tagSlug: string | undefined,
  endDateFilter: GammaDateFilter | undefined,
): URLSearchParams {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
    active: 'true',
    closed: 'false',
    order: 'volume24hr',
    ascending: 'false',
  });
  if (tagSlug) params.set('tag_slug', tagSlug);
  if (endDateFilter) {
    params.set('end_date_min', endDateFilter.end_date_min);
    params.set('end_date_max', endDateFilter.end_date_max);
  }
  return params;
}

function collectMatchingMarkets(
  events: GammaEvent[],
  query: string,
  seenQuestions: Set<string>,
  targetCount: number,
): NormalizedPolymarketMarket[] {
  const out: NormalizedPolymarketMarket[] = [];
  for (const event of events) {
    if (!event.markets?.length) continue;
    const titleMatches = questionMatchesQuery(event.title ?? '', query);
    const sorted = [...event.markets]
      .filter((market) => market.active && !market.closed)
      .sort((a, b) => (b.volume24hr ?? 0) - (a.volume24hr ?? 0));
    for (const market of sorted) {
      if (!titleMatches && !questionMatchesQuery(market.question, query)) continue;
      const normalized = normalizeGammaMarket(market);
      if (!normalized || seenQuestions.has(normalized.question)) continue;
      seenQuestions.add(normalized.question);
      out.push(normalized);
      if (out.length >= targetCount) return out;
    }
  }
  return out;
}

export async function searchGammaEventsPaginated(
  input: PolymarketSearchInput,
  fetchFn: typeof fetch = fetch,
): Promise<PolymarketSearchResult> {
  const limit = Math.max(1, input.limit);
  const pageSize = Math.max(limit, input.pageSize ?? 50);
  const maxPages = Math.max(1, input.maxPages ?? 4);
  const tagSlugs = input.tagSlugs?.length ? input.tagSlugs : [undefined];
  const seenQuestions = new Set<string>();
  const markets: NormalizedPolymarketMarket[] = [];
  let pagesRead = 0;

  for (const tagSlug of tagSlugs) {
    for (let pageIndex = 0; pageIndex < maxPages && markets.length < limit; pageIndex += 1) {
      const offset = pageIndex * pageSize;
      const params = buildEventSearchParams(pageSize, offset, tagSlug, input.endDateFilter);
      const events = await fetchWithRetry(async () => {
        const response = await fetchFn(`${GAMMA_BASE}/events?${params}`);
        if (!response.ok) {
          throw new Error(`Gamma API ${response.status}`);
        }
        return response.json() as Promise<GammaEvent[]>;
      });
      pagesRead += 1;
      markets.push(...collectMatchingMarkets(events, input.query, seenQuestions, limit - markets.length));
      if (events.length < pageSize) break;
    }
    if (markets.length >= limit) break;
  }

  return {
    markets,
    warnings: [],
    provenance: {
      source: 'gamma-http',
      pagesRead,
      tagSlugsTried: tagSlugs.filter((slug): slug is string => typeof slug === 'string'),
      usedEndDateFilter: input.endDateFilter !== undefined,
    },
  };
}

export function createHttpReadDriver(fetchFn: typeof fetch = fetch): PolymarketReadDriver {
  return {
    async searchEvents(input: PolymarketSearchInput) {
      return searchGammaEventsPaginated(input, fetchFn);
    },
    async fetchSpread(tokenId: string) {
      return fetchClobSpread(tokenId);
    },
    async fetchPriceHistory(tokenId: string, interval: '1h' | '6h' | '1d') {
      return fetchClobPriceHistory(tokenId, interval);
    },
  };
}
