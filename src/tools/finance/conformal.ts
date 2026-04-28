/**
 * Online Conformal PID Wrapper
 *
 * Implements the PID-controller variant of online conformal prediction from
 *
 *   Angelopoulos, Candès & Tibshirani (2023)
 *   "Conformal PID Control for Time Series Prediction"
 *   arXiv:2307.16895
 *
 * The wrapper sits *outside* any forecasting model. Given a stream of
 * (forecastCenter, actual) pairs, it adapts a single radius `q` such that the
 * long-run miscoverage of the symmetric interval [center − q, center + q]
 * approaches a target α. No assumptions on the underlying model — finite
 * sample coverage holds under arbitrary distribution shift in the spirit of
 * the original adaptive-conformal-inference (ACI) family.
 *
 * Why "PID":
 *
 *   bias_t = err_t − α                            (proportional term)
 *   I_t    = γ · I_{t−1} + bias_t                 (decaying integral)
 *   D_t    = bias_t − bias_{t−1}                  (derivative)
 *   q_{t+1} = max(0, q_t + lr · (Kp·b + Ki·I + Kd·D))
 *
 * The integral decay γ < 1 prevents windup; it is the only deviation from a
 * textbook PID and matches the practical recipe in §3 of the paper.
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

export class ConformalPID {
  readonly alpha: number;
  readonly targetCoverage: number;
  private readonly lr: number;
  private readonly kp: number;
  private readonly ki: number;
  private readonly kd: number;
  private readonly gamma: number;

  private q: number;
  private integral: number;
  private prevBias: number;
  private samples: number;
  private hits: number;

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
    if (!Number.isFinite(forecastCenter) || !Number.isFinite(actual)) return;
    const residual = Math.abs(actual - forecastCenter);
    const covered = residual <= this.q ? 1 : 0;
    const err = 1 - covered; // 1 if missed, 0 if hit
    const bias = err - this.alpha;

    this.integral = this.gamma * this.integral + bias;
    const derivative = bias - this.prevBias;
    this.prevBias = bias;

    const update = this.lr * (this.kp * bias + this.ki * this.integral + this.kd * derivative);
    this.q = Math.max(0, this.q + update);

    this.samples += 1;
    this.hits += covered;
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
