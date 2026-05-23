/**
 * Logit Jump-Diffusion for prediction-market prices.
 *
 * Reference: *Toward Black–Scholes for Prediction Markets* (arXiv:2510.15205,
 * Oct 2025). Models the price p_t ∈ (0, 1) of a binary prediction-market
 * contract by working in log-odds space x = logit(p) where the support is
 * unbounded:
 *
 *   dx = μ(p) dt + σ dW + J · dN
 *
 *   μ(p) = -½ σ² (1 - 2p) - λ · ⟨Δp⟩ / (p(1-p))   ← Itô-Jensen drift
 *
 *   J ~ N(μ_J, σ_J²)   in log-odds space
 *   N is a Poisson process with intensity λ.
 *
 * The drift μ(p) is the **martingale-constrained** value: setting it to this
 * Itô-Jensen correction makes E[p_t | p_{t-1}] ≈ p_{t-1} per step, i.e., p
 * is a (discrete-time) martingale up to O(dt²). Without the correction the
 * naive simulation would drift toward 0.5 because of the concavity of the
 * sigmoid.
 *
 * Polymarket-informed intensity: callers can pass `polymarketJumpProb`
 * (the probability the underlying event resolves within `days`) and the
 * function decomposes it into a per-day λ = polymarketJumpProb / days.
 *
 * Mirrors `research/models/logit_jump_diffusion.py`.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LogitJumpDiffusionParams {
  /** Current Polymarket price in (0, 1). Boundary inputs are clipped. */
  initialPrice: number;
  /** Horizon in days (≥ 1). */
  days: number;
  /** Per-day diffusion volatility in *log-odds* units. */
  sigmaPerDay: number;

  /**
   * Per-day Poisson jump intensity λ.
   *   ── If `polymarketJumpProb` is set, that takes precedence and λ is set
   *      to `polymarketJumpProb / days`.
   *   ── Otherwise this raw intensity is used directly.
   */
  jumpIntensityPerDay?: number;
  /** Polymarket-implied total probability of one jump event over the horizon. */
  polymarketJumpProb?: number;

  /** Mean log-odds shift when a jump fires. Sign encodes direction. */
  jumpLogitMean?: number;
  /** Std-dev of the jump's log-odds shift. */
  jumpLogitStd?: number;

  /** Number of MC paths. */
  nPaths: number;
  /** Optional RNG `() => [0,1)`. Defaults to Math.random for production use. */
  rng?: () => number;
  /** Whether to retain the full (nPaths × days) trajectory matrix. */
  storePaths?: boolean;
}

export interface LogitJumpDiffusionResult {
  /** Terminal price for each path, length = nPaths. */
  terminal: number[];
  /** Per-day price trajectory if `storePaths` was true. */
  paths?: number[][];
  /** Total number of jump events fired across all paths and timesteps. */
  totalJumps: number;
  /** Per-day intensity actually used. */
  effectiveLambda: number;
}

// ---------------------------------------------------------------------------
// Logit / inverse-logit (numerically safe)
// ---------------------------------------------------------------------------

const Z_MAX = 30; // exp(30)/(1+exp(30)) ≈ 1 to ~13 digits — good enough.

export function logit(p: number): number {
  const eps = 1e-12;
  const clipped = Math.min(1 - eps, Math.max(eps, p));
  return Math.log(clipped / (1 - clipped));
}

export function invLogit(z: number): number {
  if (z >= Z_MAX) return 1 - 1e-12;
  if (z <= -Z_MAX) return 1e-12;
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

// ---------------------------------------------------------------------------
// Martingale-constrained drift
// ---------------------------------------------------------------------------

/**
 * Itô-Jensen drift correction in log-odds space such that p = sigmoid(x)
 * is a martingale to leading order in dt.
 *
 *   μ_x = -½ σ² (1 - 2p)  +  jump-compensator
 *
 * Jump compensator: -λ · (E_J[sigmoid(x+J) − p]) / (p(1-p)). For a Gaussian
 * jump in logit space we approximate E_J[sigmoid(x+J)] by a Hermite quadrature
 * (3-point) — exact integration is unnecessary; we just need the martingale
 * property to hold to ~O(σ⁴, λ²) which Hermite-3 covers.
 */
export function itoMartingaleDrift(
  p: number,
  sigma: number,
  lambda: number,
  jumpMean: number,
  jumpStd: number,
): number {
  const safeP = Math.min(1 - 1e-12, Math.max(1e-12, p));
  const diffusionDrift = -0.5 * sigma * sigma * (1 - 2 * safeP);
  if (lambda <= 0 || (jumpMean === 0 && jumpStd === 0)) {
    return diffusionDrift;
  }
  // Gauss–Hermite 3-point nodes & weights for ∫ e^{-x²}f(x) dx.
  const nodes = [-Math.sqrt(1.5), 0, Math.sqrt(1.5)];
  const weights = [1 / 6, 2 / 3, 1 / 6];
  const x = logit(safeP);
  let expected = 0;
  for (let i = 0; i < 3; i++) {
    const j = jumpMean + Math.SQRT2 * jumpStd * nodes[i];
    expected += weights[i] * invLogit(x + j);
  }
  const meanDp = expected - safeP;
  const denom = safeP * (1 - safeP);
  if (denom < 1e-12) return diffusionDrift;
  return diffusionDrift - (lambda * meanDp) / denom;
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------

function boxMuller(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function simulateLogitJumpDiffusion(params: LogitJumpDiffusionParams): LogitJumpDiffusionResult {
  const {
    initialPrice,
    days,
    sigmaPerDay,
    jumpLogitMean = 0,
    jumpLogitStd = 0,
    nPaths,
    rng = Math.random,
    storePaths = false,
  } = params;

  if (!(days >= 1)) throw new Error('simulateLogitJumpDiffusion: days must be ≥ 1');
  if (!(nPaths >= 1)) throw new Error('simulateLogitJumpDiffusion: nPaths must be ≥ 1');

  const lambda = params.polymarketJumpProb !== undefined
    ? Math.max(0, Math.min(1, params.polymarketJumpProb)) / days
    : Math.max(0, params.jumpIntensityPerDay ?? 0);

  const initialLogit = logit(initialPrice);
  const terminal: number[] = new Array(nPaths);
  const paths: number[][] | undefined = storePaths ? new Array(nPaths) : undefined;
  let totalJumps = 0;

  for (let s = 0; s < nPaths; s++) {
    let x = initialLogit;
    let p = invLogit(x);
    const trajectory: number[] | undefined = storePaths ? new Array(days) : undefined;
    for (let d = 0; d < days; d++) {
      const drift = itoMartingaleDrift(p, sigmaPerDay, lambda, jumpLogitMean, jumpLogitStd);
      const z = boxMuller(rng);
      x += drift + sigmaPerDay * z;
      if (lambda > 0 && rng() < lambda) {
        const jz = boxMuller(rng);
        x += jumpLogitMean + jumpLogitStd * jz;
        totalJumps++;
      }
      p = invLogit(x);
      if (trajectory) trajectory[d] = p;
    }
    terminal[s] = p;
    if (paths && trajectory) paths[s] = trajectory;
  }

  return { terminal, paths, totalJumps, effectiveLambda: lambda };
}
