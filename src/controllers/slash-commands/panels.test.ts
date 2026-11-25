import { describe, expect, it } from 'bun:test';
import type { Container } from '@mariozechner/pi-tui';
import type { WatchlistEntry } from '../watchlist-controller.js';
import type { PriceSnapshot } from '../watchlist-display.js';
import { buildSnapshotPanel, buildWatchlistPanel } from './panels.js';

function texts(node: Container): string[] {
  const out: string[] = [];
  for (const child of (node as unknown as { children: unknown[] }).children) {
    const text = (child as { text?: string }).text;
    if (text !== undefined) out.push(text.replace(/\x1b\[[0-9;]*m/g, ''));
    if ((child as { children?: unknown[] }).children) out.push(...texts(child as Container));
  }
  return out;
}

describe('watchlist Markov confidence', () => {
  // The score does not rank forecast accuracy (live-config benchmark and 8 held-out tickers, 2026-11-25), so the
  // panels show the number without a quality mark or a high/medium/low breakdown.
  const entries: WatchlistEntry[] = [{ ticker: 'SPY', costBasis: 100, shares: 1, addedAt: '2026-01-01' }];
  const prices = new Map<string, PriceSnapshot>([
    ['SPY', { ticker: 'SPY', price: 110, changePercent: 1, predictionConfidence: 0.23 }],
  ]);

  it('shows the score as a number in the watchlist row', () => {
    const row = texts(buildWatchlistPanel(entries, prices)).find(t => t.includes('SPY'));
    expect(row).toContain('0.23');
    expect(row).not.toMatch(/[✓⚠✗]/);
  });

  it('shows the score as a number in the snapshot, without a quality breakdown', () => {
    const lines = texts(buildSnapshotPanel(entries, prices));
    expect(lines.find(t => t.includes('SPY') && t.includes('%'))).toContain('0.23');
    expect(lines.join('\n')).not.toMatch(/[✓⚠✗]|Confidence Distribution|^\s*(High|Medium|Low) \(/m);
  });
});
