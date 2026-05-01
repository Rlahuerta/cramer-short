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
    this.alpha = opts.alpha ?? 0.1;
    this.targetCoverage = 1 - this.alpha;
    this.q = Math.max(0, opts.initialRadius ?? 1.0);
    this.lr = opts.learningRate ?? 0.05;
    this.kp = opts.kp ?? 1.0;
    this.ki = opts.ki ?? 0.1;
    this.kd = opts.kd ?? 0.1;
    this.gamma = opts.integralDecay ?? 1.0;
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
    this.enabled = opts.enabled ?? false;
    this.breakLearningRateMultiplier = Math.max(1, opts.breakLearningRateMultiplier ?? 1.5);
    this.cooloffWindow = Math.max(0, Math.round(opts.cooloffWindow ?? 0));
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
