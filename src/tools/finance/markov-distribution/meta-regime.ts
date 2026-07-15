/** Mirrors `research/models/meta_regime.py`. */

export type Environment = 'low_uncertainty' | 'high_uncertainty';

export interface MetaRegimeFitResult {
  environment_sequence: Environment[];
  current_environment: Environment | undefined;
  threshold_used: number;
}

export class MetaRegimeDetector {
  readonly vol_window: number;
  readonly high_vol_threshold: number;
  private _threshold: number | undefined;
  private _fitted = false;

  constructor(vol_window = 20, high_vol_threshold = 0.75) {
    this.vol_window = vol_window;
    this.high_vol_threshold = high_vol_threshold;
  }

  get fitted(): boolean {
    return this._fitted;
  }

  get threshold(): number | undefined {
    return this._threshold;
  }

  fit(returns: number[]): MetaRegimeFitResult {
    const rollingVol = Array(returns.length).fill(Number.NaN) as number[];

    for (let i = this.vol_window - 1; i < returns.length; i++) {
      const window = returns.slice(i - this.vol_window + 1, i + 1);
      rollingVol[i] = sampleStd(window);
    }

    const valid = rollingVol.filter(v => !Number.isNaN(v));
    this._threshold = valid.length === 0 ? 0.02 : percentile(valid, this.high_vol_threshold * 100);

    const environment_sequence = rollingVol.map(vol => (
      Number.isNaN(vol) || vol <= this._threshold! ? 'low_uncertainty' : 'high_uncertainty'
    ));

    this._fitted = true;
    return {
      environment_sequence,
      current_environment: environment_sequence.at(-1),
      threshold_used: this._threshold,
    };
  }
}

function sampleStd(values: number[]): number {
  if (values.length <= 1) return Number.NaN;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function percentile(values: number[], percentileRank: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];

  const index = (percentileRank / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;

  return sorted[lower] + (sorted[upper] - sorted[lower]) * weight;
}
