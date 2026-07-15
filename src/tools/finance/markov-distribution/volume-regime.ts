/** Mirrors `research/models/markov/volume_regime.py`. */

export type VolumeEnvironment = 'low_volume' | 'normal_volume' | 'high_volume';
export type VolumeRegime = 'low' | 'normal' | 'high';

export interface VolumeRegimeDetectionResult {
  environment_sequence: VolumeEnvironment[];
  current_environment: VolumeEnvironment;
}

function median(values: number[]): number {
  if (values.some(Number.isNaN)) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function classifyVolumeRegime(
  volumes: number[],
  lookback = 20,
  thresholdMultiplier = 1.5,
): VolumeRegime[] {
  if (thresholdMultiplier <= 0) {
    throw new RangeError(`threshold_multiplier must be positive, got ${thresholdMultiplier}`);
  }
  if (lookback < 2) {
    throw new RangeError(`lookback must be >= 2, got ${lookback}`);
  }

  const n = volumes.length;
  if (n === 0) return [];

  const result: VolumeRegime[] = Array(n).fill('normal');
  if (n <= lookback) return result;

  for (let i = lookback; i < n; i++) {
    const rollingMedian = median(volumes.slice(i - lookback, i));
    if (rollingMedian <= 0 || Number.isNaN(rollingMedian)) continue;

    const value = volumes[i];
    if (Number.isNaN(value)) continue;
    if (value > thresholdMultiplier * rollingMedian) {
      result[i] = 'high';
    } else if (value < rollingMedian / thresholdMultiplier) {
      result[i] = 'low';
    }
  }

  return result;
}

export class VolumeRegimeDetector {
  private isFitted = false;

  constructor(
    public readonly lookback = 20,
    public readonly thresholdMultiplier = 1.5,
  ) {}

  get fitted(): boolean {
    return this.isFitted;
  }

  fit(volumes: number[]): VolumeRegimeDetectionResult {
    const labels = classifyVolumeRegime(volumes, this.lookback, this.thresholdMultiplier);
    const envMap: Record<VolumeRegime, VolumeEnvironment> = {
      high: 'high_volume',
      normal: 'normal_volume',
      low: 'low_volume',
    };
    const environmentSequence = labels.map(label => envMap[label]);

    this.isFitted = true;
    return {
      environment_sequence: environmentSequence,
      current_environment: environmentSequence.at(-1) ?? 'normal_volume',
    };
  }
}
