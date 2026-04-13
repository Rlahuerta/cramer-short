/**
 * Robinhood API HTTP client — wraps the unofficial Robinhood Trade private REST API.
 *
 * IMPORTANT: This is a reverse-engineered private API with no official documentation
 * or ToS protection. Endpoints may break without notice. Use as a fallback only.
 *
 * Public endpoints (quotes, fundamentals) do not require authentication.
 * Base URL: https://api.robinhood.com
 *
 * Reference: github.com/sanko/Robinhood (reverse-engineered API docs)
 */

const RH_BASE_URL = 'https://api.robinhood.com';

export interface RobinhoodQuote {
  symbol: string;
  bidPrice: string | null;
  askPrice: string | null;
  bidSize: number | null;
  askSize: number | null;
  lastTradePrice: string | null;
  lastTradeSize: number | null;
  lastTradeCondition: string | null;
  lastUpdatedAt: string;
  previousClose: string | null;
  adjustedPreviousClose: string | null;
  tradingHalted: boolean;
  marketState: string | null;
  volume: number | null;
}

export interface RobinhoodFundamentals {
  symbol: string;
  open: string | null;
  high: string | null;
  low: string | null;
  volume: number | null;
  averageVolume: number | null;
  high52Week: string | null;
  low52Week: string | null;
  marketCap: string | null;
  adjustedMarketCap: number | null;
  priceEarningsRatio: string | null;
  earningsPerShare: string | null;
  // Dividend
  dividendsYield: number | null;
  dividendsPerShare: number | null;
  dividendDate: string | null;
  // Shares
  sharesOutstanding: number | null;
  // Description
  description: string | null;
}

async function rhFetch<T>(path: string): Promise<T> {
  const url = `${RH_BASE_URL}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        'User-Agent': 'Robinhood/8.15.0 (Android 11)',
      },
    });
    if (!res.ok) {
      throw new Error(`[Robinhood API] ${res.status} ${res.statusText}`);
    }
    return res.json() as Promise<T>;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('[Robinhood API] request timed out after 15s');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

/** Fetch a real-time quote for a ticker. Returns null on failure. */
export async function getQuote(ticker: string): Promise<RobinhoodQuote | null> {
  try {
    const data = await rhFetch<RobinhoodQuote | { detail: string }>(
      `/quotes/${ticker.toUpperCase()}/`,
    );
    // Robinhood returns { detail: "Not found" } for unknown tickers
    if ('detail' in data) return null;
    return data;
  } catch {
    return null;
  }
}

/** Fetch fundamental metrics for a ticker. Returns null on failure. */
export async function getFundamentals(ticker: string): Promise<RobinhoodFundamentals | null> {
  try {
    const data = await rhFetch<RobinhoodFundamentals | { detail: string }>(
      `/fundamentals/${ticker.toUpperCase()}/`,
    );
    if ('detail' in data) return null;
    return data;
  } catch {
    return null;
  }
}
