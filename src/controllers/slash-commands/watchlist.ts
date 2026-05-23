import { seedWatchlistEntries } from '../../memory/auto-store.js';
import { WatchlistController, parseWatchlistSubcommand } from '../watchlist-controller.js';
import type { TuiStateController } from '../tui-state-controller.js';

export type WatchlistSlashCommandResult =
  | { handled: true }
  | { handled: false; query?: string };

function showStatusBriefly(message: string, tuiState: TuiStateController): void {
  tuiState.setStatus(message);
  tuiState.requestRender();
  const timeout = tuiState.timers?.setTimeout ?? setTimeout;
  timeout(() => { tuiState.setStatus(tuiState.currentModel()); tuiState.requestRender(); }, 3000);
}

export async function handleWatchlistSlashCommand(
  query: string,
  tuiState: TuiStateController,
): Promise<WatchlistSlashCommandResult> {
  if (!query.startsWith('/watchlist')) {
    return { handled: false };
  }

  const watchlistCtrl = new WatchlistController(tuiState.cwd());
  const sub = parseWatchlistSubcommand(query.slice('/watchlist'.length).trim());

  // Flush current completed exchange to scrollback before any overlay hides chatLog.
  // This preserves the conversation history so it isn't visually erased when the
  // watchlist panel renders over the chat area.
  if (sub.cmd === 'list' || sub.cmd === 'show' || sub.cmd === 'snapshot') {
    const prevItem = tuiState.history().at(-1);
    if (prevItem && (prevItem.status === 'complete' || prevItem.status === 'interrupted')) {
      tuiState.flushItemToScrollback(prevItem);
    }
  }

  if (sub.cmd === 'add') {
    watchlistCtrl.add(sub.ticker, sub.costBasis, sub.shares);
    const detail = [
      sub.costBasis !== undefined ? `@ $${sub.costBasis}` : '',
      sub.shares !== undefined ? `× ${sub.shares} shares` : '',
    ].filter(Boolean).join(' ');
    tuiState.setError(null);
    showStatusBriefly(`✓ Added ${sub.ticker}${detail ? ' ' + detail : ''} to watchlist`, tuiState);
    // Seed the new ticker into financial memory so recall_financial_context
    // returns a hit even before any LLM analysis runs.
    void seedWatchlistEntries([{ ticker: sub.ticker, costBasis: sub.costBasis, shares: sub.shares }]);
    return { handled: true };
  }

  if (sub.cmd === 'remove') {
    watchlistCtrl.remove(sub.ticker);
    tuiState.setError(null);
    showStatusBriefly(`✓ Removed ${sub.ticker} from watchlist`, tuiState);
    return { handled: true };
  }

  if (sub.cmd === 'refresh') {
    if (!tuiState.watchlist.isVisible()) {
      tuiState.setError('Open a watchlist view first with /watchlist list, show, or snapshot.');
      tuiState.refreshError();
      tuiState.renderSelectionOverlay();
      tuiState.requestRender();
      return { handled: true };
    }
    void tuiState.watchlist.refreshPrices();
    return { handled: true };
  }

  if (sub.cmd === 'list') {
    tuiState.watchlist.setEntries(watchlistCtrl.list());
    tuiState.watchlist.setMode('list');
    tuiState.watchlist.setShowTicker(null);
    tuiState.watchlist.setPrices(null); // will show loading state
    tuiState.watchlist.setVisible(true);
    tuiState.renderSelectionOverlay();
    tuiState.requestRender();
    // Fetch prices in background; re-render when done
    void tuiState.watchlist.refreshPrices();
    // Start auto-refresh if enabled
    if (tuiState.watchlist.getRefreshIntervalMs() > 0) {
      tuiState.watchlist.setRefresh(tuiState.watchlist.getRefreshIntervalMs());
    }
    return { handled: true };
  }

  if (sub.cmd === 'show') {
    const ticker = sub.ticker;
    const allEntries = watchlistCtrl.list();
    tuiState.watchlist.setEntries(allEntries);
    tuiState.watchlist.setMode('show');
    tuiState.watchlist.setShowTicker(ticker);
    tuiState.watchlist.setPrices(null);
    tuiState.watchlist.setVisible(true);
    tuiState.renderSelectionOverlay();
    tuiState.requestRender();
    // Fetch rich data for this ticker
    void tuiState.watchlist.refreshPrices();
    if (tuiState.watchlist.getRefreshIntervalMs() > 0) {
      tuiState.watchlist.setRefresh(tuiState.watchlist.getRefreshIntervalMs());
    }
    return { handled: true };
  }

  if (sub.cmd === 'snapshot') {
    tuiState.watchlist.setEntries(watchlistCtrl.list());
    tuiState.watchlist.setMode('snapshot');
    tuiState.watchlist.setShowTicker(null);
    tuiState.watchlist.setPrices(null);
    tuiState.watchlist.setVisible(true);
    tuiState.renderSelectionOverlay();
    tuiState.requestRender();
    void tuiState.watchlist.refreshPrices();
    if (tuiState.watchlist.getRefreshIntervalMs() > 0) {
      tuiState.watchlist.setRefresh(tuiState.watchlist.getRefreshIntervalMs());
    }
    return { handled: true };
  }

  // Bare /watchlist — run briefing skill with injected context.
  if (watchlistCtrl.isEmpty()) {
    tuiState.setError('Watchlist is empty. Use /watchlist add TICKER [cost] [shares] to add positions.');
    tuiState.refreshError();
    tuiState.renderSelectionOverlay();
    tuiState.requestRender();
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
