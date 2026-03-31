/**
 * Core implementation of auto-store async functions.
 *
 * Deliberately has NO import of MemoryManager and NO import of auto-store.ts,
 * so that mock.module('../memory/auto-store.js', ...) in controller tests never
 * contaminates these implementations.
 *
 * Uses duck-typed interfaces for the manager/store objects.
 */

import type { ToolCallRecord } from '../agent/scratchpad.js';

// ---------------------------------------------------------------------------
// Ticker extraction (mirrors extractTickers in auto-store.ts — pure function)
// ---------------------------------------------------------------------------

const TICKER_RE = /\b([A-Z]{1,5}(?:\.[A-Z]{2,3})?)\b/g;

const SKIP_TOKENS = new Set([
  'A', 'AN', 'I', 'BE', 'DO', 'GO', 'MY', 'AT', 'BY', 'ME', 'NO', 'OR',
  'SO', 'UP', 'US', 'WE', 'HE', 'IT', 'IN', 'IS', 'IF', 'OF', 'ON', 'TO',
  'AS', 'AND', 'ARE', 'BUT', 'CAN', 'FOR', 'HAS', 'HIM', 'HIS', 'ITS',
  'NOT', 'OUR', 'SHE', 'THE', 'WAS', 'YOU', 'ALL', 'ANY', 'EACH', 'ELSE',
  'FROM', 'HAVE', 'THAT', 'THEM', 'THEN', 'THEY', 'THIS', 'WERE', 'WHAT',
  'WITH', 'YOUR', 'ALSO', 'BEEN', 'BOTH', 'EACH', 'EVEN', 'JUST', 'LESS',
  'LIKE', 'LONG', 'MANY', 'MORE', 'MOST', 'MUCH', 'NEED', 'ONLY', 'OVER',
  'SAME', 'SUCH', 'THAN', 'VERY', 'WELL', 'WILL', 'WHEN', 'YEAR',
  'DCF', 'EPS', 'ETF', 'FMP', 'FTS', 'GDP', 'INC', 'IPO', 'LLC',
  'LTM', 'NAV', 'NAN', 'OTC', 'P/E', 'PE', 'PEG', 'SEC', 'TTM',
  'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY',
  'CEO', 'CFO', 'COO', 'CTO', 'CFR', 'CWD', 'ENV', 'FAQ',
  'API', 'FMT', 'HTML', 'HTTP', 'JSON', 'URL', 'UTC',
  'BUY', 'CALL', 'COME', 'FIND', 'GET', 'GIVE', 'GOOD', 'HELP',
  'HIGH', 'HOLD', 'INTO', 'KEEP', 'KNOW', 'LOOK', 'MAKE', 'NEXT',
  'READ', 'SELL', 'SHOW', 'TAKE', 'TELL', 'THEIR', 'USE', 'WANT',
  'YEAR', 'YOY', 'QOQ', 'FY', 'Q1', 'Q2', 'Q3', 'Q4',
  'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'OUTER', 'NULL',
]);

function extractTickers(text: string): string[] {
  const matches = new Set<string>();
  TICKER_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = TICKER_RE.exec(text)) !== null) {
    const t = m[1]!.toUpperCase();
    if (t.length >= 2 && !SKIP_TOKENS.has(t)) matches.add(t);
  }
  return Array.from(matches);
}

// ---------------------------------------------------------------------------
// Minimal interface for the financial store (duck-typed, no MemoryManager dep)
// ---------------------------------------------------------------------------

interface FinancialStorelike {
  storeInsight(opts: {
    ticker: string;
    content: string;
    tags: string[];
    routing?: string;
    source: string;
  }): Promise<void>;
  recallByTicker(ticker: string): Array<{ source?: string; tags?: string[]; updatedAt?: number }>;
}

interface ManagerLike {
  getFinancialStore(): FinancialStorelike | null;
}

// ---------------------------------------------------------------------------
// Routing inference (duplicated here to avoid any auto-store.js dependency)
// ---------------------------------------------------------------------------

type RoutingResult = 'fmp-ok' | 'fmp-premium' | 'yahoo-ok' | 'web-fallback';

function inferRouting(ticker: string, toolCalls: ToolCallRecord[]): RoutingResult | null {
  const upper = ticker.toUpperCase();
  const relevant = toolCalls.filter((tc) => {
    const haystack = (JSON.stringify(tc.args) + tc.result).toUpperCase();
    return haystack.includes(upper);
  });
  if (relevant.length === 0) return null;

  const premiumPattern = /premium|subscription required|not available.*free|upgrade.*plan/i;
  const dataPattern = /price|revenue|earnings|market.*cap|\$[\d,]+|p\/e|roe|eps|income|balance/i;

  for (const tc of relevant) {
    if ((tc.tool === 'get_financials' || tc.tool === 'get_market_data') && premiumPattern.test(tc.result)) {
      return 'fmp-premium';
    }
  }
  for (const tc of relevant) {
    if (
      (tc.tool === 'get_financials' || tc.tool === 'get_market_data') &&
      dataPattern.test(tc.result) &&
      !premiumPattern.test(tc.result)
    ) {
      return 'fmp-ok';
    }
  }
  for (const tc of relevant) {
    if ((tc.tool === 'web_search' || tc.tool === 'browser') && dataPattern.test(tc.result)) {
      return 'web-fallback';
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Public injectable implementations
// ---------------------------------------------------------------------------

export interface WatchlistEntryLike {
  ticker: string;
  costBasis?: number;
  shares?: number;
}

export async function seedWatchlistEntriesCore(
  entries: WatchlistEntryLike[],
  getManager: () => Promise<ManagerLike>,
): Promise<void> {
  if (entries.length === 0) return;
  try {
    const manager = await getManager();
    const store = manager.getFinancialStore();
    if (!store) return;

    for (const entry of entries) {
      const existing = store.recallByTicker(entry.ticker);
      const hasWatchlistRecord = existing.some(
        (e) => (e.source === 'watchlist') || (e.tags ?? []).includes('source:watchlist'),
      );
      if (hasWatchlistRecord) continue;

      const posDetails: string[] = [];
      if (entry.costBasis !== undefined) posDetails.push(`cost basis $${entry.costBasis}`);
      if (entry.shares !== undefined) posDetails.push(`${entry.shares} shares`);
      const posStr = posDetails.length > 0 ? ` (${posDetails.join(', ')})` : '';

      await store.storeInsight({
        ticker: entry.ticker,
        content: `User is tracking ${entry.ticker} in their investment watchlist${posStr}.`,
        tags: ['source:watchlist', `ticker:${entry.ticker}`],
        source: 'watchlist',
      });
    }
  } catch {
    // Memory persistence is non-critical — never throw to the caller.
  }
}

export async function autoStoreFromRunCore(
  query: string,
  answer: string,
  toolCalls: Array<{ tool: string; args: Record<string, unknown>; result: string }>,
  getManager: () => Promise<ManagerLike>,
): Promise<void> {
  const FINANCIAL_TOOLS = new Set([
    'get_market_data', 'get_financials', 'read_filings',
    'financial_search', 'web_search', 'browser',
  ]);
  const usedFinancialTool = toolCalls.some((tc) => FINANCIAL_TOOLS.has(tc.tool));
  if (!usedFinancialTool) return;

  if (toolCalls.some((tc) => tc.tool === 'store_financial_insight')) return;

  const tickers = extractTickers(query);
  if (tickers.length === 0) return;

  try {
    const manager = await getManager();
    const store = manager.getFinancialStore();
    if (!store) return;

    const recentCutoff = Date.now() - 24 * 60 * 60 * 1000;

    for (const ticker of tickers.slice(0, 6)) {
      const existing = store.recallByTicker(ticker);
      const hasRecent = existing.some((e) => (e.updatedAt ?? 0) > recentCutoff);
      if (hasRecent) continue;

      const routing = inferRouting(ticker, toolCalls as ToolCallRecord[]);
      const tags = [`ticker:${ticker}`, 'source:auto-run'];
      if (routing) tags.push(`routing:${routing}`);

      const answerExcerpt = answer.trim().slice(0, 200);
      const content = [
        `Query: "${query.slice(0, 150)}"`,
        routing ? `Data source: ${routing}` : null,
        answerExcerpt ? `Summary: ${answerExcerpt}${answer.length > 200 ? '…' : ''}` : null,
      ]
        .filter(Boolean)
        .join(' — ');

      await store.storeInsight({
        ticker,
        content,
        tags,
        routing: routing ?? undefined,
        source: 'auto-run',
      });
    }
  } catch {
    // Non-critical.
  }
}
