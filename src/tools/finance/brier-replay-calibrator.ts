export interface BrierReplayCalibratorOptions {
  learningRate?: number;
  midConfidenceWeight?: number;
  midConfidenceMin?: number;
  midConfidenceMax?: number;
  maxSlope?: number;
  maxBias?: number;
}

export interface BrierReplayCalibratorState {
  bias: number;
  slope: number;
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function logit(probability: number): number {
  const p = clamp(probability, 1e-6, 1 - 1e-6);
  return Math.log(p / (1 - p));
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

export function predictWithBrierReplayState(
  rawProbability: number,
  state: BrierReplayCalibratorState,
): number {
  return sigmoid(state.slope * logit(rawProbability) + state.bias);
}

export class BrierReplayCalibrator {
  private readonly learningRate: number;
  private readonly midConfidenceWeight: number;
  private readonly midConfidenceMin: number;
  private readonly midConfidenceMax: number;
  private readonly maxSlope: number;
  private readonly maxBias: number;
  private bias = 0;
  private slope = 1;

  constructor(options: BrierReplayCalibratorOptions = {}) {
    this.learningRate = options.learningRate ?? 0.1;
    this.midConfidenceWeight = options.midConfidenceWeight ?? 2;
    this.midConfidenceMin = options.midConfidenceMin ?? 0.4;
    this.midConfidenceMax = options.midConfidenceMax ?? 0.6;
    this.maxSlope = Math.max(1, options.maxSlope ?? 3);
    this.maxBias = Math.max(0.25, options.maxBias ?? 1.5);
  }

  predict(rawProbability: number): number {
    return predictWithBrierReplayState(rawProbability, this.state());
  }

  record(rawProbability: number, actualBinary: number): BrierReplayCalibratorState {
    const prediction = this.predict(rawProbability);
    const z = logit(rawProbability);
    const midWeight = rawProbability >= this.midConfidenceMin && rawProbability <= this.midConfidenceMax
      ? this.midConfidenceWeight
      : 1;
    const gradient = 2 * (prediction - actualBinary) * prediction * (1 - prediction) * midWeight;
    this.bias = clamp(this.bias - this.learningRate * gradient, -this.maxBias, this.maxBias);
    this.slope = clamp(this.slope - this.learningRate * gradient * z, 1 / this.maxSlope, this.maxSlope);
    return this.state();
  }

  state(): BrierReplayCalibratorState {
    return { bias: this.bias, slope: this.slope };
  }
}
