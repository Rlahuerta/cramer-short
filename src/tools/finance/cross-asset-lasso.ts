/**
 * R4 Phase D — Cross-asset LASSO pooling for forecast drift bias.
 *
 * Pure coordinate-descent Lasso (Friedman, Hastie & Tibshirani 2010).  No
 * external dependencies.  Used to find sparse, robust signal from a panel of
 * peer-asset daily returns onto the *target* asset's H-day forward return.
 *
 * The fitted intercept-corrected linear predictor at the most recent observation
 * supplies an additive bias to the Markov-chain drift in the trajectory MC.
 *
 * Algorithm:
 *   minimise  (1 / 2N) · Σ_i (y_i − α − x_i·β)² + λ · Σ_j |β_j|
 *
 * with feature standardisation, intercept fit by sample mean, soft-thresholding
 * coordinate updates, and a single λ supplied by the caller (default 0.01 is
 * a safe starting point for daily-return scale).
 *
 * Source: docs/forecast-improvement-ideas-round4-2026-04-28.md Idea 2.
 */

export interface LassoFit {
  /** Intercept α. */
  intercept: number;
  /** Standardised slope coefficients β (length = n_features). */
  coef: number[];
  /** Per-feature mean used for standardisation (so caller can apply at predict-time). */
  featureMean: number[];
  /** Per-feature std used for standardisation (zeros are mapped to 1 to avoid /0). */
  featureStd: number[];
  /** Lambda used. */
  lambda: number;
  /** Iterations actually run. */
  iterations: number;
}

export interface LassoOptions {
  /** L1 regularisation strength.  Default 0.01. */
  lambda?: number;
  /** Max coordinate-descent sweeps.  Default 200. */
  maxIterations?: number;
  /** Convergence tolerance on max |β_new − β_old|.  Default 1e-6. */
  tolerance?: number;
}

function softThreshold(z: number, lam: number): number {
  if (z > lam) return z - lam;
  if (z < -lam) return z + lam;
  return 0;
}

/**
 * Fit Lasso on (X, y).  X is a [n_samples][n_features] matrix.
 *
 * Internally standardises features (zero mean, unit std) and centres y (so
 * the intercept is recovered as ȳ).  Returns the fit in *standardised*
 * coordinate space; use {@link predictLasso} to get a prediction in original
 * units.
 */
export function fitLasso(X: number[][], y: number[], opts: LassoOptions = {}): LassoFit {
  const lambda = opts.lambda ?? 0.01;
  const maxIter = opts.maxIterations ?? 200;
  const tol = opts.tolerance ?? 1e-6;
  const n = y.length;
  if (n === 0 || X.length !== n) {
    throw new Error(`fitLasso: X rows (${X.length}) must equal y length (${n})`);
  }
  const p = X[0]?.length ?? 0;
  if (p === 0) {
    return { intercept: y.reduce((s, v) => s + v, 0) / n, coef: [], featureMean: [], featureStd: [], lambda, iterations: 0 };
  }

  // Column means / stds
  const mean = new Array(p).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) mean[j] += X[i][j];
  for (let j = 0; j < p; j++) mean[j] /= n;
  const std = new Array(p).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) std[j] += (X[i][j] - mean[j]) ** 2;
  for (let j = 0; j < p; j++) std[j] = Math.sqrt(std[j] / n) || 1;

  // Standardise X and centre y.
  const Z: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(p);
    for (let j = 0; j < p; j++) row[j] = (X[i][j] - mean[j]) / std[j];
    Z[i] = row;
  }
  const yBar = y.reduce((s, v) => s + v, 0) / n;
  const yc = y.map(v => v - yBar);

  const beta = new Array(p).fill(0);
  // Pre-compute residual vector r = yc − Z β  (initially yc since β=0).
  const r = yc.slice();

  let iter = 0;
  for (; iter < maxIter; iter++) {
    let maxDelta = 0;
    for (let j = 0; j < p; j++) {
      // Partial residual including current j contribution: r_j = r + Z[:, j] · β_j
      let zr = 0;
      let zz = 0;
      for (let i = 0; i < n; i++) {
        const z = Z[i][j];
        zr += z * (r[i] + z * beta[j]);
        zz += z * z;
      }
      const denom = zz / n;
      const newBeta = denom > 0 ? softThreshold(zr / n, lambda) / denom : 0;
      const delta = newBeta - beta[j];
      if (delta !== 0) {
        for (let i = 0; i < n; i++) r[i] -= delta * Z[i][j];
        if (Math.abs(delta) > maxDelta) maxDelta = Math.abs(delta);
        beta[j] = newBeta;
      }
    }
    if (maxDelta < tol) {
      iter++;
      break;
    }
  }

  return {
    intercept: yBar,
    coef: beta,
    featureMean: mean,
    featureStd: std,
    lambda,
    iterations: iter,
  };
}

/** Predict y for a single observation x in original feature units. */
export function predictLasso(fit: LassoFit, x: readonly number[]): number {
  let acc = fit.intercept;
  for (let j = 0; j < fit.coef.length; j++) {
    if (fit.coef[j] === 0) continue;
    acc += fit.coef[j] * (x[j] - fit.featureMean[j]) / fit.featureStd[j];
  }
  return acc;
}

/**
 * Cross-asset bias estimator.
 *
 * Builds (X, y) from `targetReturns` and `peerReturns`:
 *   y[i] = sum_{k=0..horizon-1} targetReturns[i + lag + k]   (forward H-day return)
 *   x[i] = [ peer1_returns[i], peer2_returns[i], … ]         (today's peer 1-day returns)
 *
 * Fits Lasso, then returns `predict(latest peer returns) / horizon` as a
 * per-day bias to be *added* to the Markov drift inside computeTrajectory.
 *
 * Returns `null` when there is insufficient overlapping data.
 */
export function estimateCrossAssetBias(
  targetReturns: readonly number[],
  peerReturns: Record<string, readonly number[]>,
  horizon: number,
  opts: LassoOptions & { lag?: number; minSamples?: number } = {},
): { perDayBias: number; fit: LassoFit; tickers: string[] } | null {
  const lag = opts.lag ?? 0;
  const minSamples = opts.minSamples ?? 60;
  const tickers = Object.keys(peerReturns).sort();
  if (tickers.length === 0) return null;

  // Align: usable index range is [0, T - horizon - lag).
  const T = targetReturns.length;
  const peerLengths = tickers.map(t => peerReturns[t].length);
  const minPeerLen = Math.min(...peerLengths);
  const usable = Math.min(T, minPeerLen) - horizon - lag;
  if (usable < minSamples) return null;

  const n = usable;
  const X: number[][] = new Array(n);
  const y: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(tickers.length);
    for (let k = 0; k < tickers.length; k++) row[k] = peerReturns[tickers[k]][i];
    X[i] = row;
    let fwd = 0;
    for (let h = 0; h < horizon; h++) fwd += targetReturns[i + lag + h];
    y[i] = fwd;
  }

  const fit = fitLasso(X, y, opts);

  // Latest peer observations (most recent index that exists in all peers).
  const latestIdx = Math.min(...peerLengths) - 1;
  const latest = tickers.map(t => peerReturns[t][latestIdx]);
  const yhat = predictLasso(fit, latest);
  const perDayBias = yhat / Math.max(1, horizon);
  return { perDayBias, fit, tickers };
}
