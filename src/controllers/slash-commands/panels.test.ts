import { describe, expect, it } from 'bun:test';
import type { Container } from '@mariozechner/pi-tui';
import { RECOMMENDED_CONFIDENCE_THRESHOLD } from '../../tools/finance/markov-distribution/core.js';
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

describe('watchlist Markov confidence marks', () => {
  // Between the markov_distribution recommended threshold (0.22) and the 0.25 the panel used to hard-code.
  const confidence = 0.23;
  const entries: WatchlistEntry[] = [{ ticker: 'SPY', costBasis: 100, shares: 1, addedAt: '2026-01-01' }];
  const prices = new Map<string, PriceSnapshot>([
    ['SPY', { ticker: 'SPY', price: 110, changePercent: 1, predictionConfidence: confidence }],
  ]);

  it('marks confidence at or above the recommended threshold as usable', () => {
    expect(confidence).toBeGreaterThanOrEqual(RECOMMENDED_CONFIDENCE_THRESHOLD);
    const row = texts(buildWatchlistPanel(entries, prices)).find(t => t.includes('SPY'));
    expect(row).toContain('✓');
    expect(row).not.toContain('⚠');
  });

  it('counts it as medium, not low, in the snapshot distribution', () => {
    const lines = texts(buildSnapshotPanel(entries, prices));
    expect(lines.find(t => t.startsWith('  Medium'))).toMatch(/\(1\)$/);
    expect(lines.find(t => t.startsWith('  Low'))).toMatch(/\(0\)$/);
  });
});
