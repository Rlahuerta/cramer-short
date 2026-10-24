/**
 * Regression tests for the /watchlist slash command's scrollback bookkeeping.
 *
 * BUG: handleWatchlistSlashCommand flushed the previous completed exchange to
 * scrollback but never recorded it in `tuiState.flushedItems` (dream.ts does).
 * On the next submit, cli.ts saw the item as un-flushed and wrote it to
 * scrollback a second time.
 */
import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { TuiStateController } from '../tui-state-controller.js';
import type { HistoryItem } from '../types.js';
import { handleWatchlistSlashCommand } from './watchlist.js';

let tmpDir: string;

beforeEach(() => {
  tmpDir = join(tmpdir(), `dexter-wl-slash-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(tmpDir, { recursive: true });
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

function makeTuiState(history: HistoryItem[]) {
  const flushes: HistoryItem[] = [];
  const flushedItems = new WeakSet<HistoryItem>();

  const state = {
    cwd: () => tmpDir,
    history: () => history,
    flushedItems,
    flushItemToScrollback: (item: HistoryItem) => {
      flushes.push(item);
    },
    currentModel: () => 'test-model',
    setStatus: () => {},
    setError: () => {},
    refreshError: () => {},
    requestRender: () => {},
    renderSelectionOverlay: () => {},
    watchlist: {
      isVisible: () => true,
      setVisible: () => {},
      setEntries: () => {},
      setPrices: () => {},
      setMode: () => {},
      setShowTicker: () => {},
      refreshPrices: async () => {},
      getRefreshIntervalMs: () => 0,
      setRefresh: () => {},
    },
    dream: {
      isAgentProcessing: () => false,
      isRunning: () => false,
      setRunning: () => {},
    },
  };

  return { state: state as unknown as TuiStateController, flushes, flushedItems };
}

function completedItem(id: string): HistoryItem {
  return { id, query: 'what is AAPL?', events: [], answer: 'done', status: 'complete' };
}

describe('handleWatchlistSlashCommand scrollback flush bookkeeping', () => {
  it('records the flushed exchange so the next submit does not flush it again', async () => {
    const item = completedItem('q1');
    const { state, flushes, flushedItems } = makeTuiState([item]);

    await handleWatchlistSlashCommand('/watchlist list', state);
    expect(flushes).toEqual([item]);

    // Mirror cli.ts: on the next submit, flush only when not already recorded.
    const prev = state.history().at(-1)!;
    if (prev.status === 'complete' && !flushedItems.has(prev)) {
      flushes.push(prev);
    }

    expect(flushes).toHaveLength(1);
  });

  it('does not flush an exchange that is already in flushedItems', async () => {
    const item = completedItem('q2');
    const { state, flushes, flushedItems } = makeTuiState([item]);
    flushedItems.add(item);

    await handleWatchlistSlashCommand('/watchlist snapshot', state);

    expect(flushes).toHaveLength(0);
  });
});
