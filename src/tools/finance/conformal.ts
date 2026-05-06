import {
  createForecastLabAssetScopedRuntimeDefaults,
  type ForecastLabRuntimeAssetScope,
} from './forecast-lab-runtime-defaults.js';

/**
 * Online Conformal PID Wrapper.
 *
 * Implements the PID-controller variant of online conformal prediction from
 * Angelopoulos, Candès & Tibshirani (2023), "Conformal PID Control for Time
 * Series Prediction", arXiv:2307.16895.
 *
 * The wrapper adapts a symmetric radius `q` so long-run miscoverage approaches
 * a target alpha. The integral decay γ < 1 prevents windup (§3 of the paper).
 */

export interface ConformalPIDOptions {
  /** Target miscoverage rate. 0.1 ⇒ 90% coverage. Default 0.1. */
  alpha?: number;
  /** Initial radius for the symmetric interval. Default 1.0. */
  initialRadius?: number;
  /** Base learning rate applied to the PID update. Default 0.05. */
  learningRate?: number;
  /** Proportional gain. Default 1.0. */
  kp?: number;
  /** Integral gain. Default 0.1. */
  ki?: number;
  /** Derivative gain. Default 0.1. */
  kd?: number;
  /** Decay applied to the integral term each step. Default 1.0 (no decay). */
  integralDecay?: number;
}

export interface ConformalInterval {
  low: number;
  high: number;
}

export type AdaptiveConformalMode = 'normal' | 'break';

export interface AdaptiveConformalRecordDiagnostics {
  structuralBreak?: boolean;
  realizedVol?: number;
}

export interface AdaptiveConformalPIDOptions extends ConformalPIDOptions {
  /**
   * Explicit on/off switch for structural-break-aware updates. Default false.
   */
  enabled?: boolean;
  /** Multiplier used for break-mode learning-rate and interval inflation. */
  breakLearningRateMultiplier?: number;
  /** Number of subsequent steps to keep break mode active after a trigger. */
  cooloffWindow?: number;
}

export interface AdaptiveConformalMetadata {
  applied: true;
  radius: number;
  coverageEstimate: number | null;
  mode: AdaptiveConformalMode;
}

export interface ScoreAggregatedConformalOptions {
  /** Target miscoverage rate. Reuses the conformal α. Default 0.1. */
  alpha?: number;
  /** Minimum calibration sample count before score aggregation activates. Default 20. */
  minSamples?: number;
  /** Maximum number of aggregated scores retained for calibration. Default 120. */
  calibrationWindow?: number;
}

export interface ScoreAggregatedConformalInterval extends ConformalInterval {
  applied: boolean;
  radius: number;
  multiplier: number | null;
}

export const FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS: {
  readonly pidLearningRate: number;
  readonly integralDecay: number;
  readonly adaptiveBreakEnabled: boolean;
  readonly adaptiveBreakLearningRateMultiplier: number;
  readonly adaptiveBreakCooloffWindow: number;
  readonly scoreAggregationMinSamples: number;
  readonly scoreAggregationCalibrationWindow: number;
} = {
  pidLearningRate: 0.05,
  integralDecay: 1.0,
  adaptiveBreakEnabled: false,
  adaptiveBreakLearningRateMultiplier: 1.5,
  adaptiveBreakCooloffWindow: 0,
  scoreAggregationMinSamples: 16,
  scoreAggregationCalibrationWindow: 96,
};

const forecastLabConformalRuntimeDefaults = createForecastLabAssetScopedRuntimeDefaults(
  FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS,
);

export function resolveForecastLabConformalParameterDefaults(
  assetScope?: ForecastLabRuntimeAssetScope,
): typeof FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS {
  return forecastLabConformalRuntimeDefaults.resolve(assetScope);
}

export function getForecastLabConformalRuntimeDefaults(
  assetScope: ForecastLabRuntimeAssetScope,
): Partial<typeof FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS> | undefined {
  return forecastLabConformalRuntimeDefaults.get(assetScope);
}

export function setForecastLabConformalRuntimeDefaults(
  assetScope: ForecastLabRuntimeAssetScope,
  overrides?: Partial<typeof FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS>,
): void {
  forecastLabConformalRuntimeDefaults.set(assetScope, overrides);
}

export class ConformalPID {
  readonly alpha: number;
  readonly targetCoverage: number;
  protected readonly lr: number;
  protected readonly kp: number;
  protected readonly ki: number;
  protected readonly kd: number;
  protected readonly gamma: number;

  protected q: number;
  protected integral: number;
  protected prevBias: number;
  protected samples: number;
  protected hits: number;

  constructor(opts: ConformalPIDOptions = {}) {
    const defaults = resolveForecastLabConformalParameterDefaults();
    this.alpha = opts.alpha ?? 0.1;
    this.targetCoverage = 1 - this.alpha;
    this.q = Math.max(0, opts.initialRadius ?? 1.0);
    this.lr = opts.learningRate ?? defaults.pidLearningRate;
    this.kp = opts.kp ?? 1.0;
    this.ki = opts.ki ?? 0.1;
    this.kd = opts.kd ?? 0.1;
    this.gamma = opts.integralDecay ?? defaults.integralDecay;
    this.integral = 0;
    this.prevBias = 0;
    this.samples = 0;
    this.hits = 0;
  }

  /** Record a (forecastCenter, actual) pair and update the PID state. */
  record(forecastCenter: number, actual: number): void {
    this.step(forecastCenter, actual, this.lr);
  }

  protected step(
    forecastCenter: number,
    actual: number,
    learningRate: number,
  ): { residual: number; covered: boolean } | undefined {
    if (!Number.isFinite(forecastCenter) || !Number.isFinite(actual)) return;
    const residual = Math.abs(actual - forecastCenter);
    const covered = residual <= this.q ? 1 : 0;
    const err = 1 - covered; // 1 if missed, 0 if hit
    const bias = err - this.alpha;

    this.integral = this.gamma * this.integral + bias;
    const derivative = bias - this.prevBias;
    this.prevBias = bias;

    const update = learningRate * (this.kp * bias + this.ki * this.integral + this.kd * derivative);
    this.q = Math.max(0, this.q + update);

    this.samples += 1;
    this.hits += covered;
    return { residual, covered: covered === 1 };
  }

  /** Current symmetric radius. */
  currentRadius(): number {
    return this.q;
  }

  /** Reset controller state when the wrapped forecaster effectively restarts. */
  reset(opts: { radius?: number } = {}): void {
    if (opts.radius !== undefined) {
      this.q = Math.max(0, opts.radius);
    }
    this.integral = 0;
    this.prevBias = 0;
    this.samples = 0;
    this.hits = 0;
  }

  /** Wrap a forecast center in the current symmetric interval. */
  wrap(forecastCenter: number): ConformalInterval {
    return { low: forecastCenter - this.q, high: forecastCenter + this.q };
  }

  /** Number of samples observed so far. */
  sampleCount(): number {
    return this.samples;
  }

  /** Running empirical coverage. Undefined until the first record() call. */
  empiricalCoverage(): number | undefined {
    if (this.samples === 0) return undefined;
    return this.hits / this.samples;
  }
}

export class AdaptiveConformalPID extends ConformalPID {
  private readonly enabled: boolean;
  private readonly breakLearningRateMultiplier: number;
  private readonly cooloffWindow: number;
  private cooloffRemaining = 0;
  private residualEma?: number;
  private volatilityEma?: number;
  private mode: AdaptiveConformalMode = 'normal';
  private lastAppliedRadius: number;

  constructor(opts: AdaptiveConformalPIDOptions = {}) {
    super(opts);
    const defaults = resolveForecastLabConformalParameterDefaults();
    this.enabled = opts.enabled ?? defaults.adaptiveBreakEnabled;
    this.breakLearningRateMultiplier = Math.max(
      1,
      opts.breakLearningRateMultiplier ?? defaults.adaptiveBreakLearningRateMultiplier,
    );
    this.cooloffWindow = Math.max(
      0,
      Math.round(opts.cooloffWindow ?? defaults.adaptiveBreakCooloffWindow),
    );
    this.lastAppliedRadius = this.q;
  }

  override wrap(
    forecastCenter: number,
    diagnostics?: AdaptiveConformalRecordDiagnostics,
  ): ConformalInterval {
    const mode = this.resolveMode(diagnostics);
    const radius = this.appliedRadius(mode);
    this.lastAppliedRadius = radius;
    return { low: forecastCenter - radius, high: forecastCenter + radius };
  }

  override record(
    forecastCenter: number,
    actual: number,
    diagnostics?: AdaptiveConformalRecordDiagnostics,
  ): void {
    const mode = this.resolveMode(diagnostics, { consumeCooloff: true });
    const learningRate = mode === 'break'
      ? this.lr * this.breakLearningRateMultiplier
      : this.lr;
    const stepped = this.step(forecastCenter, actual, learningRate);
    if (!stepped) return;
    this.updateResidualEma(stepped.residual);
    this.updateVolatilityEma(diagnostics?.realizedVol);
  }

  currentMode(): AdaptiveConformalMode {
    return this.mode;
  }

  diagnostics(): AdaptiveConformalMetadata {
    return {
      applied: true,
      radius: this.lastAppliedRadius,
      coverageEstimate: this.empiricalCoverage() ?? null,
      mode: this.mode,
    };
  }

  override reset(opts: { radius?: number } = {}): void {
    super.reset(opts);
    this.cooloffRemaining = 0;
    this.residualEma = undefined;
    this.volatilityEma = undefined;
    this.mode = 'normal';
    this.lastAppliedRadius = this.q;
  }

  private resolveMode(
    diagnostics?: AdaptiveConformalRecordDiagnostics,
    options?: { consumeCooloff?: boolean },
  ): AdaptiveConformalMode {
    if (!this.enabled) {
      if (options?.consumeCooloff === true) this.cooloffRemaining = 0;
      this.mode = 'normal';
      return this.mode;
    }

    const triggered = diagnostics?.structuralBreak === true
      || this.isVolatilityShock(diagnostics?.realizedVol);
    const inCooloff = this.cooloffRemaining > 0;
    const mode = triggered || inCooloff ? 'break' : 'normal';

    if (options?.consumeCooloff === true) {
      if (triggered) {
        this.cooloffRemaining = this.cooloffWindow;
      } else if (inCooloff) {
        this.cooloffRemaining -= 1;
      }
    }

    this.mode = mode;
    return this.mode;
  }

  private appliedRadius(mode: AdaptiveConformalMode): number {
    if (mode !== 'break') return this.q;
    return this.q * Math.max(1, Math.sqrt(this.breakLearningRateMultiplier));
  }

  private isVolatilityShock(realizedVol?: number): boolean {
    if (!Number.isFinite(realizedVol) || realizedVol === undefined || realizedVol <= 0) return false;
    const baseline = this.volatilityEma ?? this.residualEma;
    if (baseline === undefined) return false;
    return realizedVol >= baseline * this.breakLearningRateMultiplier;
  }

  private updateResidualEma(residual: number): void {
    if (!Number.isFinite(residual) || residual < 0) return;
    this.residualEma = this.residualEma === undefined
      ? residual
      : (this.residualEma * 0.95) + (residual * 0.05);
  }

  private updateVolatilityEma(realizedVol?: number): void {
    if (!Number.isFinite(realizedVol) || realizedVol === undefined || realizedVol <= 0) return;
    this.volatilityEma = this.volatilityEma === undefined
      ? realizedVol
      : (this.volatilityEma * 0.9) + (realizedVol * 0.1);
  }
}

export class ScoreAggregatedConformal {
  private readonly alpha: number;
  private readonly minSamples: number;
  private readonly calibrationWindow: number;
  private readonly scores: number[] = [];

  constructor(opts: ScoreAggregatedConformalOptions = {}) {
    const defaults = resolveForecastLabConformalParameterDefaults();
    this.alpha = opts.alpha ?? 0.1;
    this.minSamples = Math.max(1, Math.round(opts.minSamples ?? defaults.scoreAggregationMinSamples));
    this.calibrationWindow = Math.max(
      this.minSamples,
      Math.round(opts.calibrationWindow ?? defaults.scoreAggregationCalibrationWindow),
    );
  }

  wrap(
    forecastCenter: number,
    sourceRadii: readonly number[],
  ): ScoreAggregatedConformalInterval {
    const radii = normalizeSourceRadii(sourceRadii);
    const baseRadius = radii.length > 0 ? Math.min(...radii) : 0;
    const multiplier = this.scoreMultiplier();
    const radius = baseRadius * (multiplier ?? 1);
    return {
      low: forecastCenter - radius,
      high: forecastCenter + radius,
      applied: multiplier !== undefined,
      radius,
      multiplier: multiplier ?? null,
    };
  }

  record(
    forecastCenter: number,
    actual: number,
    sourceRadii: readonly number[],
  ): void {
    if (!Number.isFinite(forecastCenter) || !Number.isFinite(actual)) return;
    const radii = normalizeSourceRadii(sourceRadii);
    if (radii.length === 0) return;

    const residual = Math.abs(actual - forecastCenter);
    const aggregatedScore = radii.reduce(
      (maxScore, radius) => Math.max(maxScore, residual / radius),
      0,
    );
    if (!Number.isFinite(aggregatedScore)) return;

    this.scores.push(aggregatedScore);
    if (this.scores.length > this.calibrationWindow) {
      this.scores.splice(0, this.scores.length - this.calibrationWindow);
    }
  }

  sampleCount(): number {
    return this.scores.length;
  }

  reset(): void {
    this.scores.length = 0;
  }

  private scoreMultiplier(): number | undefined {
    if (this.scores.length < this.minSamples) return undefined;
    return Math.max(1, upperQuantile(this.scores, 1 - this.alpha));
  }
}

function normalizeSourceRadii(sourceRadii: readonly number[]): number[] {
  return sourceRadii.filter((radius) => Number.isFinite(radius) && radius > 0);
}

function upperQuantile(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const clampedQuantile = Math.min(1, Math.max(0, quantile));
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(sorted.length * clampedQuantile) - 1),
  );
  return sorted[index];
}
