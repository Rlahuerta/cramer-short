import { seedWatchlistEntries } from '../../memory/auto-store.js';
import type { HistoryItem } from '../types.js';
import { WatchlistController, parseWatchlistSubcommand, type WatchlistEntry } from '../watchlist-controller.js';
import type { PriceSnapshot } from '../watchlist-display.js';

export type WatchlistSlashCommandResult =
  | { handled: true }
  | { handled: false; query?: string };

interface WatchlistSlashCommandOptions {
  readonly cwd: () => string;
  readonly history: () => HistoryItem[];
  readonly flushItemToScrollback: (item: HistoryItem) => void;
  readonly setError: (message: string | null) => void;
  readonly refreshError: () => void;
  readonly requestRender: () => void;
  readonly renderSelectionOverlay: () => void;
  readonly currentModel: () => string;
  readonly setStatus: (message: string) => void;
  readonly isWatchlistVisible: () => boolean;
  readonly setWatchlistVisible: (visible: boolean) => void;
  readonly setWatchlistEntries: (entries: WatchlistEntry[]) => void;
  readonly setWatchlistPrices: (prices: Map<string, PriceSnapshot> | null) => void;
  readonly setWatchlistMode: (mode: 'list' | 'show' | 'snapshot') => void;
  readonly setWatchlistShowTicker: (ticker: string | null) => void;
  readonly refreshWatchlistPrices: () => Promise<void>;
  readonly getWatchlistRefreshIntervalMs: () => number;
  readonly setWatchlistRefresh: (intervalMs: number) => void;
  readonly setTimeoutFn?: typeof setTimeout;
}

function showStatusBriefly(message: string, options: WatchlistSlashCommandOptions): void {
  options.setStatus(message);
  options.requestRender();
  const timeout = options.setTimeoutFn ?? setTimeout;
  timeout(() => { options.setStatus(options.currentModel()); options.requestRender(); }, 3000);
}

export async function handleWatchlistSlashCommand(
  query: string,
  options: WatchlistSlashCommandOptions,
): Promise<WatchlistSlashCommandResult> {
  if (!query.startsWith('/watchlist')) {
    return { handled: false };
  }

  const watchlistCtrl = new WatchlistController(options.cwd());
  const sub = parseWatchlistSubcommand(query.slice('/watchlist'.length).trim());

  // Flush current completed exchange to scrollback before any overlay hides chatLog.
  // This preserves the conversation history so it isn't visually erased when the
  // watchlist panel renders over the chat area.
  if (sub.cmd === 'list' || sub.cmd === 'show' || sub.cmd === 'snapshot') {
    const prevItem = options.history().at(-1);
    if (prevItem && (prevItem.status === 'complete' || prevItem.status === 'interrupted')) {
      options.flushItemToScrollback(prevItem);
    }
  }

  if (sub.cmd === 'add') {
    watchlistCtrl.add(sub.ticker, sub.costBasis, sub.shares);
    const detail = [
      sub.costBasis !== undefined ? `@ $${sub.costBasis}` : '',
      sub.shares !== undefined ? `× ${sub.shares} shares` : '',
    ].filter(Boolean).join(' ');
    options.setError(null);
    showStatusBriefly(`✓ Added ${sub.ticker}${detail ? ' ' + detail : ''} to watchlist`, options);
    // Seed the new ticker into financial memory so recall_financial_context
    // returns a hit even before any LLM analysis runs.
    void seedWatchlistEntries([{ ticker: sub.ticker, costBasis: sub.costBasis, shares: sub.shares }]);
    return { handled: true };
  }

  if (sub.cmd === 'remove') {
    watchlistCtrl.remove(sub.ticker);
    options.setError(null);
    showStatusBriefly(`✓ Removed ${sub.ticker} from watchlist`, options);
    return { handled: true };
  }

  if (sub.cmd === 'refresh') {
    if (!options.isWatchlistVisible()) {
      options.setError('Open a watchlist view first with /watchlist list, show, or snapshot.');
      options.refreshError();
      options.renderSelectionOverlay();
      options.requestRender();
      return { handled: true };
    }
    void options.refreshWatchlistPrices();
    return { handled: true };
  }

  if (sub.cmd === 'list') {
    options.setWatchlistEntries(watchlistCtrl.list());
    options.setWatchlistMode('list');
    options.setWatchlistShowTicker(null);
    options.setWatchlistPrices(null); // will show loading state
    options.setWatchlistVisible(true);
    options.renderSelectionOverlay();
    options.requestRender();
    // Fetch prices in background; re-render when done
    void options.refreshWatchlistPrices();
    // Start auto-refresh if enabled
    if (options.getWatchlistRefreshIntervalMs() > 0) {
      options.setWatchlistRefresh(options.getWatchlistRefreshIntervalMs());
    }
    return { handled: true };
  }

  if (sub.cmd === 'show') {
    const ticker = sub.ticker;
    const allEntries = watchlistCtrl.list();
    options.setWatchlistEntries(allEntries);
    options.setWatchlistMode('show');
    options.setWatchlistShowTicker(ticker);
    options.setWatchlistPrices(null);
    options.setWatchlistVisible(true);
    options.renderSelectionOverlay();
    options.requestRender();
    // Fetch rich data for this ticker
    void options.refreshWatchlistPrices();
    if (options.getWatchlistRefreshIntervalMs() > 0) {
      options.setWatchlistRefresh(options.getWatchlistRefreshIntervalMs());
    }
    return { handled: true };
  }

  if (sub.cmd === 'snapshot') {
    options.setWatchlistEntries(watchlistCtrl.list());
    options.setWatchlistMode('snapshot');
    options.setWatchlistShowTicker(null);
    options.setWatchlistPrices(null);
    options.setWatchlistVisible(true);
    options.renderSelectionOverlay();
    options.requestRender();
    void options.refreshWatchlistPrices();
    if (options.getWatchlistRefreshIntervalMs() > 0) {
      options.setWatchlistRefresh(options.getWatchlistRefreshIntervalMs());
    }
    return { handled: true };
  }

  // Bare /watchlist — run briefing skill with injected context.
  if (watchlistCtrl.isEmpty()) {
    options.setError('Watchlist is empty. Use /watchlist add TICKER [cost] [shares] to add positions.');
    options.refreshError();
    options.renderSelectionOverlay();
    options.requestRender();
    return { handled: true };
  }

  const entries = watchlistCtrl.list();
  const context = entries
    .map((e) => {
      const parts = [e.ticker];
      if (e.shares !== undefined && e.costBasis !== undefined)
        parts.push(`(${e.shares} shares @ $${e.costBasis})`);
      else if (e.costBasis !== undefined)
        parts.push(`(@ $${e.costBasis})`);
      return parts.join(' ');
    })
    .join(', ');
  // Fall through to agent submission with injected watchlist context.
  return { handled: false, query: `Run watchlist briefing for: ${context}` };
}
