/**
 * Live-configuration walk-forward benchmark and its paired decision rule (Step 0.3 of
 * docs/reviews/price-prediction-improvements-plan-2026-11-22.md).
 *
 * A forecast change is judged by pairing each step of a candidate run with the same step of a baseline run
 * and bootstrapping the pooled Brier change, not by per-cell thresholds, which sit below the metrics' noise.
 */
import { mulberry32 } from './metrics.js';

export interface BenchmarkStep {
  t: number;
  predictedProb: number;
  actualBinary: number;
  recommendation: 'BUY' | 'SELL' | 'HOLD';
}

export interface BenchmarkCell {
  ticker: string;
  horizon: number;
  stride: number;
  steps: BenchmarkStep[];
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
