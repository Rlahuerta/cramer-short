import { describe, expect, it } from 'bun:test';
import { compareBenchmarkRuns } from './live-config-benchmark.js';
import { restrictToOrigins, runAnchoredReplay } from './anchored-replay-benchmark.js';
import type { AnchorHistoryMarket } from './polymarket-anchor-history.js';

describe('runAnchoredReplay', () => {
  // 300 daily prices from 2025-01-01, and a weekly market struck at the price on its creation day.
  const prices = Array.from({ length: 300 }, (_, i) => 100 * Math.exp(0.0006 * i + 0.03 * Math.sin(i / 7)));
  const dates = prices.map((_, i) => new Date(Date.UTC(2025, 0, 1 + i)).toISOString().slice(0, 10));
  const markets: AnchorHistoryMarket[] = [];
  for (let created = 0; created + 7 < prices.length; created += 7) {
    const strike = Math.round(prices[created]);
    const daily: Record<string, number> = {};
    for (let d = created; d < created + 7; d++) daily[dates[d]] = 0.6;
    markets.push({
      question: `Will SYN be above $${strike} on ${dates[created + 7]}?`,
      createdAt: `${dates[created]}T00:00:00Z`,
      endDate: `${dates[created + 7]}T16:00:00Z`,
      volume: 50_000,
      daily,
    });
  }

  it('runs each cell with and without the replayed markets on the same origins', async () => {
    const runs = await runAnchoredReplay({
      prices, dates, markets, stride: 10,
      cells: [{ ticker: 'SYN', horizon: 3, maxHistoryPrices: 120, config: {} }],
    });

    const [plainCell] = runs.plain.cells;
    const [anchoredCell] = runs.anchored.cells;
    expect(anchoredCell.steps.map(s => s.t)).toEqual(plainCell.steps.map(s => s.t));
    const anchoredOrigins = runs.anchoredOrigins['SYN|3'];
    expect(anchoredOrigins.length).toBeGreaterThan(0);

    const restricted = restrictToOrigins(runs, runs.anchoredOrigins);
    expect(compareBenchmarkRuns(restricted.plain, restricted.anchored).pairedSteps).toBe(anchoredOrigins.length);
  });
});
