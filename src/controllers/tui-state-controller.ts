import type { HistoryItem } from './types.js';
import type { WatchlistEntry } from './watchlist-controller.js';
import type { PriceSnapshot } from './watchlist-display.js';

export type WatchlistViewMode = 'list' | 'show' | 'snapshot';

export interface TuiStateController {
  readonly cwd: () => string;
  readonly history: () => HistoryItem[];
  readonly flushedItems: WeakSet<HistoryItem>;
  readonly flushItemToScrollback: (item: HistoryItem) => void;
  readonly currentModel: () => string;
  readonly setStatus: (message: string) => void;
  readonly setError: (message: string | null) => void;
  readonly refreshError: () => void;
  readonly requestRender: () => void;
  readonly renderSelectionOverlay: () => void;
  readonly watchlist: {
    readonly isVisible: () => boolean;
    readonly setVisible: (visible: boolean) => void;
    readonly setEntries: (entries: WatchlistEntry[]) => void;
    readonly setPrices: (prices: Map<string, PriceSnapshot> | null) => void;
    readonly setMode: (mode: WatchlistViewMode) => void;
    readonly setShowTicker: (ticker: string | null) => void;
    readonly refreshPrices: () => Promise<void>;
    readonly getRefreshIntervalMs: () => number;
    readonly setRefresh: (intervalMs: number) => void;
  };
  readonly dream: {
    readonly isAgentProcessing: () => boolean;
    readonly isRunning: () => boolean;
    readonly setRunning: (running: boolean) => void;
  };
  readonly timers?: {
    readonly setTimeout?: typeof setTimeout;
    readonly setInterval?: typeof setInterval;
    readonly clearInterval?: typeof clearInterval;
  };
}
