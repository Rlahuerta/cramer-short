import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { api } from './api.js';
import { formatToolResult } from '../types.js';
import { hasEnv } from '../../utils/env.js';
import { alignPricesByDate, buildPortfolioRiskReport } from '../../utils/finance/portfolio-stats.js';
import { MS_PER_DAY } from '../../utils/time.js';
import { fetchBinanceCloses } from './binance.js';
import { normalizeHistoricalPriceTicker } from './markov-distribution/live-policies.js';
import { fetchYahooChartCloses, type DatedClose } from './yahoo-client.js';

export const PORTFOLIO_RISK_DESCRIPTION = `
Computes portfolio risk metrics — VaR, CVaR (Expected Shortfall), Sharpe ratio,
annualised volatility, max drawdown, and correlation matrix — for a set of tickers.

When to use this tool:
- User asks about portfolio risk, risk metrics, or position sizing
- User wants to know VaR (Value at Risk) or drawdown for their watchlist
- User asks "how correlated are my holdings?" or "what is my Sharpe ratio?"
- User wants to assess concentration risk across positions
- A portfolio_risk skill step calls this tool

When NOT to use this tool:
- For a single stock's current price or fundamentals (use get_financials instead)
- For screening stocks by financial ratios (use stock_screener)

Inputs:
- tickers: list of ticker symbols to analyse
- watchlist_entries: watchlist entries supplied by the caller; pass these when the user asks about their watchlist
- lookback_days: historical window for price data (default 252 ≈ 1 trading year)
- confidence_level: VaR / CVaR confidence (default 0.95)
- risk_free_rate: annual risk-free rate used for Sharpe (default 0.05)

Output:
- Per-ticker: volatility, Sharpe, VaR, CVaR, maxDrawdown
- Correlation matrix across all tickers
- Equal-weighted portfolio-level aggregates
`.trim();

const PortfolioRiskInputSchema = z.object({
  tickers: z
    .array(z.string().max(128))
    .optional()
    .describe('Ticker symbols to analyse. Pass explicitly unless watchlist_entries is supplied.'),
  watchlist_entries: z
    .array(z.object({
      ticker: z.string().max(128),
      costBasis: z.number().optional(),
      shares: z.number().optional(),
      addedAt: z.string().max(64).optional(),
    }))
    .optional()
    .describe('Watchlist entries supplied by the CLI/controller layer. Used when tickers is omitted.'),
  lookback_days: z
    .number()
    .int()
    .min(20)
    .max(1260)
    .default(252)
    .describe('Number of trading days of history to pull (default 252 ≈ 1 year).'),
  confidence_level: z
    .number()
    .min(0.5)
    .max(0.999)
    .default(0.95)
    .describe('Confidence level for VaR / CVaR (default 0.95).'),
  risk_free_rate: z
    .number()
    .min(0)
    .max(1)
    .default(0.05)
    .describe('Annual risk-free rate used for Sharpe ratio (default 0.05 = 5%).'),
});

export interface PortfolioRiskWatchlistEntry {
  ticker: string;
  costBasis?: number;
  shares?: number;
  addedAt?: string;
}

export interface PortfolioRiskToolOptions {
  watchlistEntries?: PortfolioRiskWatchlistEntry[];
}

/** Return a YYYY-MM-DD date that is `days` calendar days in the past. */
function dateNDaysAgo(days: number): string {
  const d = new Date();
  // Buffer for weekends + holidays so we get enough trading days.
  d.setDate(d.getDate() - Math.ceil(days * 1.45));
  return d.toISOString().slice(0, 10);
}

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

/**
 * Dated closes for the last `days` calendar days when Financial Datasets refuses
 * price history: Binance (crypto), then Yahoo.
 */
async function fetchDatedFallbackCloses(ticker: string, days: number): Promise<DatedClose[]> {
  const symbol = normalizeHistoricalPriceTicker(ticker);
  const endMs = Date.now();
  const startMs = endMs - days * MS_PER_DAY;
  const binance = await fetchBinanceCloses(symbol, { startMs, endMs });
  if (binance.length >= 20) return binance;
  return fetchYahooChartCloses(symbol, {
    period1: Math.floor(startMs / 1000),
    period2: Math.floor(endMs / 1000),
  });
}

function normalizeTickers(input: { tickers?: string[]; watchlist_entries?: PortfolioRiskWatchlistEntry[] }): string[] {
  const rawTickers = input.tickers?.length
    ? input.tickers
    : input.watchlist_entries?.map((entry) => entry.ticker) ?? [];

  return [...new Set(
    rawTickers
      .map((ticker) => ticker.trim().toUpperCase())
      .filter((ticker) => ticker.length > 0),
  )];
}

/**
 * Create the portfolio-risk LangChain tool with optional watchlist injection.
 */
export function createPortfolioRiskTool(options: PortfolioRiskToolOptions = {}): DynamicStructuredTool {
  /** Creates the portfolio risk analysis tool. */
  return new DynamicStructuredTool({
    name: 'portfolio_risk',
    description: PORTFOLIO_RISK_DESCRIPTION,
    schema: PortfolioRiskInputSchema,
    func: async (input) => {
      if (!hasEnv('FINANCIAL_DATASETS_API_KEY')) {
        return formatToolResult({
          error: 'FINANCIAL_DATASETS_API_KEY is not set. Portfolio risk analysis requires historical price data.',
        });
      }

      const effectiveInput = !input.tickers?.length && !input.watchlist_entries?.length && options.watchlistEntries?.length
        ? { ...input, watchlist_entries: options.watchlistEntries }
        : input;
      const tickers = normalizeTickers(effectiveInput);
      if (tickers.length === 0) {
        return formatToolResult({
          error: 'No tickers provided. Pass tickers explicitly, or pass watchlist_entries from the controller layer.',
        });
      }

      const startDate = dateNDaysAgo(effectiveInput.lookback_days);
      const endDate = todayStr();
      const sourceUrls: string[] = [];

      // Fetch close-price history for each ticker in parallel
      const pricesByTicker: Record<string, DatedClose[]> = {};
      const errors: string[] = [];

      await Promise.all(
        tickers.map(async (ticker) => {
          try {
            const { data, url } = await api.get('/prices/', {
              ticker,
              interval: 'day',
              start_date: startDate,
              end_date: endDate,
            });
            sourceUrls.push(url);
            const prices: DatedClose[] = ((data as { prices?: unknown[] }).prices ?? [])
              .map((p) => p as { time?: unknown; close?: unknown })
              .filter((p): p is { time: string; close: number } =>
                typeof p.time === 'string' && typeof p.close === 'number' && isFinite(p.close))
              .map((p) => ({ date: p.time.slice(0, 10), close: p.close }));
            if (prices.length < 20) {
              errors.push(`${ticker}: insufficient price history (${prices.length} days)`);
              return;
            }
            pricesByTicker[ticker] = prices;
          } catch (err) {
            // Free Financial Datasets plans refuse price history.
            const fallback = await fetchDatedFallbackCloses(ticker, Math.ceil(effectiveInput.lookback_days * 1.45))
              .catch(() => [] as DatedClose[]);
            if (fallback.length >= 20) {
              pricesByTicker[ticker] = fallback;
              return;
            }
            errors.push(
              `${ticker}: price fetch failed — ${err instanceof Error ? err.message : String(err)}`,
            );
          }
        }),
      );

      const validTickers = Object.keys(pricesByTicker);
      if (validTickers.length === 0) {
        return formatToolResult({ error: 'Could not fetch price data for any ticker.', errors });
      }

      // Pair returns by date: crypto closes come every calendar day, stock closes only on trading days.
      const { dates, prices } = alignPricesByDate(pricesByTicker);
      if (dates.length < 20) {
        return formatToolResult({
          error: `Only ${dates.length} dates are shared by every ticker; at least 20 are needed.`,
          errors,
        });
      }
      // Crypto-only portfolios trade every calendar day; once a stock is in, the shared dates are trading days.
      const isWeekend = (date: string) => [0, 6].includes(new Date(`${date}T00:00:00Z`).getUTCDay());
      const annualizationFactor = dates.some(isWeekend) ? 365 : 252;

      // Equal weights
      const w = 1 / validTickers.length;
      const weights: Record<string, number> = Object.fromEntries(
        validTickers.map((t) => [t, w]),
      );

      const report = buildPortfolioRiskReport(
        prices,
        weights,
        effectiveInput.confidence_level,
        effectiveInput.risk_free_rate,
        annualizationFactor,
      );

      const result = errors.length > 0 ? { ...report, warnings: errors } : report;
      return formatToolResult(result, sourceUrls);
    },
  });
}

/**
 * Default portfolio-risk tool using equal weights and optional watchlist tickers.
 */
export const portfolioRiskTool = createPortfolioRiskTool();
