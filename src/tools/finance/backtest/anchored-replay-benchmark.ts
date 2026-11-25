/**
 * Anchored replay benchmark: the live BTC configuration run over the price fixture twice, without markets and with
 * the resolved Polymarket price-threshold markets that were open at each origin (fixtures/polymarket-btc-anchor-history.json,
 * built by scripts/build-polymarket-btc-anchor-history.ts). Paired step by step, it measures what the Polymarket
 * anchors add to Markov, which the plain walk-forward benchmark cannot see.
 *
 *   bun run src/tools/finance/backtest/anchored-replay-benchmark.ts [stride=1] [out.json]
 */
import { join } from 'node:path';
import {
  btcLiveCell,
  compareBenchmarkRuns,
  type BenchmarkCell,
  type BenchmarkCellSpec,
  type BenchmarkRun,
} from './live-config-benchmark.js';
import { anchorsAtFromHistory, type AnchorHistoryFixture, type AnchorHistoryMarket } from './polymarket-anchor-history.js';
import { walkForward, type WalkForwardResult } from './walk-forward.js';

export interface AnchoredReplayRuns {
  plain: BenchmarkRun;
  anchored: BenchmarkRun;
  /** Per cell ("ticker|horizon"), the origins where the anchored forecast had at least one trusted anchor. */
  anchoredOrigins: Record<string, number[]>;
}

const RANDOM_STATE = 42;
const cellKey = (cell: { ticker: string; horizon: number }) => `${cell.ticker}|${cell.horizon}`;

function toCell(spec: BenchmarkCellSpec, stride: number, result: WalkForwardResult): BenchmarkCell {
  return {
    ticker: spec.ticker,
    horizon: spec.horizon,
    stride,
    errors: result.errors.length,
    steps: result.steps.map(s => ({ t: s.t, predictedProb: s.predictedProb, actualBinary: s.actualBinary, recommendation: s.recommendation })),
  };
}

export async function runAnchoredReplay(input: {
  prices: number[];
  dates: string[];
  markets: AnchorHistoryMarket[];
  cells: BenchmarkCellSpec[];
  stride?: number;
}): Promise<AnchoredReplayRuns> {
  const { prices, dates, markets, cells, stride = 1 } = input;
  const anchorsAt = anchorsAtFromHistory(dates, markets);
  const generatedAt = new Date().toISOString();
  const plain: BenchmarkCell[] = [];
  const anchored: BenchmarkCell[] = [];
  const anchoredOrigins: Record<string, number[]> = {};
  for (const spec of cells) {
    const base = {
      ...spec.config,
      ticker: spec.ticker,
      prices,
      horizon: spec.horizon,
      warmup: spec.maxHistoryPrices,
      stride,
      maxHistoryPrices: spec.maxHistoryPrices,
      randomState: RANDOM_STATE,
    };
    const plainResult = await walkForward(base);
    const anchoredResult = await walkForward({ ...base, anchorsAt });
    plain.push(toCell(spec, stride, plainResult));
    anchored.push(toCell(spec, stride, anchoredResult));
    anchoredOrigins[cellKey(spec)] = anchoredResult.steps.filter(s => (s.trustedAnchors ?? 0) > 0).map(s => s.t);
  }
  return {
    plain: { label: 'markov-only', generatedAt, cells: plain },
    anchored: { label: 'anchored', generatedAt, cells: anchored },
    anchoredOrigins,
  };
}

/** Both runs restricted to the given origins per cell, for a paired comparison where anchors were trusted. */
export function restrictToOrigins(
  runs: Pick<AnchoredReplayRuns, 'plain' | 'anchored'>,
  origins: Record<string, number[]>,
): { plain: BenchmarkRun; anchored: BenchmarkRun } {
  const keep = (run: BenchmarkRun): BenchmarkRun => ({
    ...run,
    cells: run.cells.map(cell => {
      const allowed = new Set(origins[cellKey(cell)] ?? []);
      return { ...cell, steps: cell.steps.filter(s => allowed.has(s.t)) };
    }),
  });
  return { plain: keep(runs.plain), anchored: keep(runs.anchored) };
}

if (import.meta.main) {
  const [strideArg = '1', out] = process.argv.slice(2);
  const fixtures = join(import.meta.dir, '..', 'fixtures');
  const prices = await Bun.file(join(fixtures, 'backtest-prices.json')).json() as {
    tickers: Record<string, { closes: number[]; dates: string[] }>;
  };
  const history = await Bun.file(join(fixtures, 'polymarket-btc-anchor-history.json')).json() as AnchorHistoryFixture;
  const btc = prices.tickers['BTC-USD'];
  const runs = await runAnchoredReplay({
    prices: btc.closes,
    dates: btc.dates,
    markets: history.markets,
    cells: [1, 2, 3, 7].map(btcLiveCell),
    stride: Number(strideArg),
  });
  const restricted = restrictToOrigins(runs, runs.anchoredOrigins);
  const summary = {
    markets: history.markets.length,
    cells: runs.plain.cells.map(cell => ({
      cell: cellKey(cell),
      steps: cell.steps.length,
      anchoredSteps: runs.anchoredOrigins[cellKey(cell)].length,
    })),
    allSteps: compareBenchmarkRuns(runs.plain, runs.anchored),
    anchoredStepsOnly: compareBenchmarkRuns(restricted.plain, restricted.anchored),
  };
  console.log(JSON.stringify(summary, null, 2));
  if (out) await Bun.write(out, JSON.stringify({ summary, runs }, null, 2));
}
