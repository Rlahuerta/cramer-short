/**
 * Live-configuration walk-forward benchmark and its paired decision rule (Step 0.3 of
 * docs/reviews/price-prediction-improvements-plan-2026-11-22.md).
 *
 * A forecast change is judged by pairing each step of a candidate run with the same step of a baseline run
 * and bootstrapping the pooled Brier change, not by per-cell thresholds, which sit below the metrics' noise.
 *
 *   bun run src/tools/finance/backtest/live-config-benchmark.ts run <label> <out.json>
 *   bun run src/tools/finance/backtest/live-config-benchmark.ts compare <base.json> <candidate.json>
 */
import { join } from 'node:path';
import { getBtcShortHorizonLivePolicy } from '../markov-distribution/live-policies.js';
import { mulberry32, type BacktestRecommendation } from './metrics.js';
import { walkForward, type WalkForwardConfig } from './walk-forward.js';
import { xiphosBtcWalkForwardConfig } from './xiphos-btc-preset.js';

export interface BenchmarkStep {
  t: number;
  predictedProb: number;
  actualBinary: number;
  recommendation: BacktestRecommendation;
}

export interface BenchmarkCell {
  ticker: string;
  horizon: number;
  stride: number;
  steps: BenchmarkStep[];
  /** Origins where the forecast failed; a candidate that fails elsewhere will not pair with the baseline. */
  errors?: number;
}

export interface BenchmarkCellSpec {
  ticker: string;
  horizon: number;
  maxHistoryPrices: number;
  config: Partial<WalkForwardConfig>;
}

const STRIDE = 5;
const RANDOM_STATE = 42;
// The live tool fetches calendar days (fetchHistoricalPrices): 120 by default and 252 under the BTC/GLD
// short-horizon policies. In the fixture's trading-day prices that is about 83 for equities and 174 for GLD;
// BTC trades every day, so 252.
const EQUITY_HISTORY_PRICES = 83;
const GLD_HISTORY_PRICES = 174;
const BTC_HISTORY_PRICES = 252;

/** The benchmark cells, each configured the way the live tool forecasts that ticker and horizon. */
export function liveBenchmarkCells(): BenchmarkCellSpec[] {
  const cells: BenchmarkCellSpec[] = [];
  for (const horizon of [1, 7, 14]) {
    for (const ticker of ['SPY', 'QQQ', 'NVDA', 'MSFT']) {
      cells.push({ ticker, horizon, maxHistoryPrices: EQUITY_HISTORY_PRICES, config: {} });
    }
    // The gold live policy's break threshold is applied inside computeMarkovDistribution.
    cells.push({ ticker: 'GLD', horizon, maxHistoryPrices: GLD_HISTORY_PRICES, config: {} });
    cells.push(btcLiveCell(horizon));
  }
  return cells;
}

/**
 * A BTC cell configured like the live tool: the xiphos profile with GARCH off (as `xiphosBtcLiveParams`), then the
 * horizon's live break threshold and rerun.
 */
export function btcLiveCell(horizon: number): BenchmarkCellSpec {
  const policy = getBtcShortHorizonLivePolicy('BTC-USD', horizon);
  return {
    ticker: 'BTC-USD',
    horizon,
    maxHistoryPrices: BTC_HISTORY_PRICES,
    config: {
      ...xiphosBtcWalkForwardConfig(),
      enableGarchVol: false,
      btcBreakDivergenceThreshold: policy?.breakDivergenceThreshold,
      postBreakShortWindow: policy?.rerunOnBreak,
      postBreakWindowSize: policy?.rerunWindowDays,
    },
  };
}

/** Run the benchmark cells over the price fixture. Each origin starts with a full live-length window. */
export async function runLiveConfigBenchmark(
  label: string,
  prices?: Record<string, number[]>,
  cells: BenchmarkCellSpec[] = liveBenchmarkCells(),
): Promise<BenchmarkRun> {
  const closes = prices ?? await loadFixtureCloses();
  const runCells: BenchmarkCell[] = [];
  for (const spec of cells) {
    const tickerPrices = closes[spec.ticker];
    if (!tickerPrices) throw new Error(`no prices for ${spec.ticker}`);
    const result = await walkForward({
      ...spec.config,
      ticker: spec.ticker,
      prices: tickerPrices,
      horizon: spec.horizon,
      warmup: spec.maxHistoryPrices,
      stride: STRIDE,
      maxHistoryPrices: spec.maxHistoryPrices,
      randomState: RANDOM_STATE,
    });
    runCells.push({
      ticker: spec.ticker,
      horizon: spec.horizon,
      stride: STRIDE,
      errors: result.errors.length,
      steps: result.steps.map((s) => ({
        t: s.t,
        predictedProb: s.predictedProb,
        actualBinary: s.actualBinary,
        recommendation: s.recommendation,
      })),
    });
  }
  return { label, generatedAt: new Date().toISOString(), cells: runCells };
}

async function loadFixtureCloses(): Promise<Record<string, number[]>> {
  const fixture = await Bun.file(join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json')).json() as {
    tickers: Record<string, { closes: number[] }>;
  };
  return Object.fromEntries(Object.entries(fixture.tickers).map(([ticker, data]) => [ticker, data.closes]));
}

export interface BenchmarkRun {
  label: string;
  generatedAt: string;
  cells: BenchmarkCell[];
}

export type BenchmarkVerdict = 'improvement' | 'no-harm' | 'harm';

export interface PairedComparison {
  pairedSteps: number;
  /** Mean of candidate minus baseline per-step Brier score; negative is better. */
  meanBrierDelta: number;
  /** 95% interval of `meanBrierDelta` from a per-cell moving-block bootstrap. */
  ci95: [number, number];
  /** Candidate minus baseline share of HOLD recommendations. */
  abstainRateDelta: number;
  verdict: BenchmarkVerdict;
}

/** Keep a change only if the pooled Brier change could be at most this much worse (95% upper bound). */
const NO_HARM_BRIER_UPPER = 0.001;
/** A change may not buy accuracy by abstaining this much more often. */
const MAX_ABSTAIN_RATE_INCREASE = 0.02;

const cellKey = (cell: { ticker: string; horizon: number }) => `${cell.ticker}|${cell.horizon}`;

export function compareBenchmarkRuns(
  base: BenchmarkRun,
  candidate: BenchmarkRun,
  { nResamples = 2000, seed = 7 }: { nResamples?: number; seed?: number } = {},
): PairedComparison {
  const candidateCells = new Map(candidate.cells.map((cell) => [cellKey(cell), cell]));
  const baseKeys = new Set(base.cells.map(cellKey));
  for (const key of candidateCells.keys()) {
    if (!baseKeys.has(key)) throw new Error(`baseline run has no cell ${key}`);
  }

  const cells: { deltas: number[]; block: number }[] = [];
  let baseHolds = 0;
  let candidateHolds = 0;
  for (const baseCell of base.cells) {
    const candidateCell = candidateCells.get(cellKey(baseCell));
    if (!candidateCell) throw new Error(`candidate run has no cell ${cellKey(baseCell)}`);
    if (candidateCell.steps.length !== baseCell.steps.length) {
      throw new Error(`cell ${cellKey(baseCell)}: ${baseCell.steps.length} vs ${candidateCell.steps.length} steps`);
    }
    const deltas = baseCell.steps.map((baseStep, i) => {
      const candidateStep = candidateCell.steps[i];
      if (candidateStep.t !== baseStep.t) {
        throw new Error(`cell ${cellKey(baseCell)}: step t=${baseStep.t} vs t=${candidateStep.t}`);
      }
      if (baseStep.recommendation === 'HOLD') baseHolds++;
      if (candidateStep.recommendation === 'HOLD') candidateHolds++;
      return (candidateStep.predictedProb - candidateStep.actualBinary) ** 2
        - (baseStep.predictedProb - baseStep.actualBinary) ** 2;
    });
    // Steps closer together than the horizon share outcome days, so resample blocks spanning the horizon.
    cells.push({ deltas, block: Math.max(1, Math.ceil(baseCell.horizon / baseCell.stride)) });
  }

  const total = cells.reduce((n, cell) => n + cell.deltas.length, 0);
  if (total === 0) throw new Error('no paired steps to compare');
  const meanBrierDelta = cells.reduce((s, cell) => s + cell.deltas.reduce((a, d) => a + d, 0), 0) / total;

  const rand = mulberry32(seed);
  const means: number[] = [];
  for (let r = 0; r < nResamples; r++) {
    let sum = 0;
    for (const { deltas, block } of cells) {
      // Circular moving-block bootstrap within the cell, keeping its size.
      for (let taken = 0; taken < deltas.length; ) {
        const start = Math.floor(rand() * deltas.length);
        for (let k = 0; k < block && taken < deltas.length; k++, taken++) sum += deltas[(start + k) % deltas.length];
      }
    }
    means.push(sum / total);
  }
  means.sort((a, b) => a - b);
  const ci95: [number, number] = [
    means[Math.floor(0.025 * (nResamples - 1))],
    means[Math.ceil(0.975 * (nResamples - 1))],
  ];

  const abstainRateDelta = (candidateHolds - baseHolds) / total;
  const verdict: BenchmarkVerdict = abstainRateDelta > MAX_ABSTAIN_RATE_INCREASE
    ? 'harm'
    : ci95[1] < 0
      ? 'improvement'
      : ci95[1] <= NO_HARM_BRIER_UPPER
        ? 'no-harm'
        : 'harm';

  return { pairedSteps: total, meanBrierDelta, ci95, abstainRateDelta, verdict };
}

if (import.meta.main) {
  const [command, ...args] = process.argv.slice(2);
  if (command === 'run' && args.length === 2) {
    const [label, out] = args;
    const run = await runLiveConfigBenchmark(label);
    await Bun.write(out, JSON.stringify(run, null, 2));
    const steps = run.cells.reduce((n, cell) => n + cell.steps.length, 0);
    const errors = run.cells.reduce((n, cell) => n + (cell.errors ?? 0), 0);
    console.log(`${label}: ${run.cells.length} cells, ${steps} steps, ${errors} failed origins -> ${out}`);
  } else if (command === 'compare' && args.length === 2) {
    const [baseFile, candidateFile] = args;
    const result = compareBenchmarkRuns(await Bun.file(baseFile).json(), await Bun.file(candidateFile).json());
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.error('Usage: live-config-benchmark.ts run <label> <out.json> | compare <base.json> <candidate.json>');
    process.exit(1);
  }
}
