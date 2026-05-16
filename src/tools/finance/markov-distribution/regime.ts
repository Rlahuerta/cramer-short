/**
 * Mirrors `research/models/markov.py` (classify_regime, classify_regime_series).
 */

import {
  REGIME_STATES,
  resolveForecastLabMarkovParameterDefaults,
  type RegimeState,
} from './core.js';

// ---------------------------------------------------------------------------
// 2. classifyRegimeState
// ---------------------------------------------------------------------------

/**
 * Classify a trading day into a directional regime state.
 *
 * Collapsed from the original 5-state model: high_vol_bull and high_vol_bear
 * are merged into bull and bear respectively, since with 120-day windows the
 * high_vol variants had only 2-4 observations and produced noisy transitions.
 *
 * @param dailyReturn - Return for the day (e.g. 0.012 = +1.2%)
 * @param _dailyVolatility - Ignored (kept for backward compatibility)
 * @param returnThreshold - Adaptive threshold for bull/bear classification (default 0.01)
 * @param _volThreshold - Ignored (kept for backward compatibility)
 */
export function classifyRegimeState(
  dailyReturn: number,
  _dailyVolatility: number,
  returnThreshold = 0.01,
  _volThreshold = 0.02,
): RegimeState {
  if (dailyReturn > returnThreshold)  return 'bull';
  if (dailyReturn < -returnThreshold) return 'bear';
  return 'sideways';
}

/**
 * Compute per-asset adaptive thresholds from the return series.
 * Uses half-median of absolute returns for regime classification and
 * 2× median for high-volatility detection. This ensures ~30-40% of days
 * are bull, ~30-40% bear, ~20-30% sideways regardless of asset volatility.
 */
export function computeAdaptiveThresholds(
  returns: number[],
  returnThresholdMultiplier = 0.5,
): {
  returnThreshold: number;
  volThreshold: number;
} {
  if (returns.length === 0) return { returnThreshold: 0.01, volThreshold: 0.02 };
  const absReturns = returns.map(r => Math.abs(r)).sort((a, b) => a - b);
  const medianAbsReturn = absReturns[Math.floor(absReturns.length / 2)];
  return {
    returnThreshold: Math.max(0.001, returnThresholdMultiplier * medianAbsReturn),
    volThreshold: Math.max(0.005, 2.0 * medianAbsReturn),
  };
}

// ---------------------------------------------------------------------------
// 2b. Momentum signal
// ---------------------------------------------------------------------------

export interface MomentumSignal {
  /** Annualized return over lookback window */
  velocity: number;
  /** Change in velocity (recent half vs older half): >0 = accelerating */
  acceleration: number;
  /** R² of OLS regression on log(prices) — measures trend linearity (0–1) */
  trendStrength: number;
  /** Daily drift adjustment to add to regime-weighted mu (clamped ±0.003) */
  adjustment: number;
}

/**
 * Compute a momentum signal from recent prices.
 * Returns a small drift adjustment that tilts the Markov distribution in the
 * direction of the recent trend, weighted by trend linearity (R²).
 *
 * @param prices  Historical prices (at least lookback+1 entries)
 * @param lookback  Number of days to look back (default 20)
 */
export function computeMomentumSignal(
  prices: number[],
  lookback = resolveForecastLabMarkovParameterDefaults().momentumLookback,
): MomentumSignal {
  const nil: MomentumSignal = { velocity: 0, acceleration: 0, trendStrength: 0, adjustment: 0 };
  if (prices.length < lookback + 1) return nil;

  const window = prices.slice(-lookback - 1);

  // Velocity: annualized return over lookback
  const totalReturn = window[window.length - 1] / window[0] - 1;
  const velocity = Math.sign(totalReturn) * (Math.pow(1 + Math.abs(totalReturn), 252 / lookback) - 1);

  // Acceleration: velocity of recent half minus velocity of older half
  const half = Math.floor(lookback / 2);
  const recentRet = window[window.length - 1] / window[window.length - 1 - half] - 1;
  const olderRet  = window[window.length - 1 - half] / window[0] - 1;
  const recentVel = Math.sign(recentRet) * (Math.pow(1 + Math.abs(recentRet), 252 / half) - 1);
  const olderVel  = Math.sign(olderRet) * (Math.pow(1 + Math.abs(olderRet), 252 / half) - 1);
  const acceleration = recentVel - olderVel;

  // Trend strength: R² of OLS on log(prices)
  const logPrices = window.map(p => Math.log(p));
  const n = logPrices.length;
  const xMean = (n - 1) / 2;
  const yMean = logPrices.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = i - xMean;
    const dy = logPrices[i] - yMean;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  const trendStrength = syy > 0 ? (sxy * sxy) / (sxx * syy) : 0;

  // Adjustment: daily drift tilt = velocity/252 × trendStrength × scaling factor
  // Clamped to ±0.003 per day (~±0.75 annualized) to prevent extreme adjustments
  const markovDefaults = resolveForecastLabMarkovParameterDefaults();
  const rawAdj = (velocity / 252) * trendStrength * markovDefaults.momentumAdjustmentScale;
  const adjustment = Math.max(
    -markovDefaults.momentumAdjustmentClamp,
    Math.min(markovDefaults.momentumAdjustmentClamp, rawAdj),
  );

  return { velocity, acceleration, trendStrength, adjustment };
}

// ---------------------------------------------------------------------------
// 5b. Ensemble signal (Idea D)
// ---------------------------------------------------------------------------

export interface EnsembleSignal {
  /** Mean-reversion z-score: negative = overbought, positive = oversold */
  meanReversionZ: number;
  /** Momentum crossover: SMA5/SMA20 ratio minus 1 */
  momentumCrossover: number;
  /** Volatility compression ratio: vol_5d / vol_20d. <1 = compressed, >1 = expanding */
  volCompression: number;
  /** Combined drift adjustment (daily), clamped to ±0.004 */
  adjustment: number;
  /** Number of signals agreeing on direction (0-3, higher = more consensus) */
  consensus: number;
}

/**
 * Compute ensemble signal from 3 orthogonal indicators:
 * 1. Mean-reversion z-score — contrarian (overbought/oversold)
 * 2. Momentum crossover — trend-following (SMA5 vs SMA20)
 * 3. Volatility compression — breakout anticipation
 *
 * Only applies adjustment when ≥2 of 3 signals agree on direction.
 */
export function computeEnsembleSignal(prices: number[]): EnsembleSignal {
  const nil: EnsembleSignal = { meanReversionZ: 0, momentumCrossover: 0, volCompression: 1, adjustment: 0, consensus: 0 };
  if (prices.length < 25) return nil;

  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }

  // --- Mean-reversion z-score: (price - SMA20) / σ20 ---
  const last20 = prices.slice(-20);
  const sma20 = last20.reduce((s, v) => s + v, 0) / 20;
  const std20 = Math.sqrt(last20.reduce((s, v) => s + (v - sma20) ** 2, 0) / 20);
  const meanReversionZ = std20 > 0 ? (prices[prices.length - 1] - sma20) / std20 : 0;
  // Negative z → price below mean → oversold → bullish signal
  const mrSignal = -meanReversionZ;

  // --- Momentum crossover: SMA5 / SMA20 - 1 ---
  const last5 = prices.slice(-5);
  const sma5 = last5.reduce((s, v) => s + v, 0) / 5;
  const momentumCrossover = sma20 > 0 ? sma5 / sma20 - 1 : 0;
  // Positive crossover → SMA5 above SMA20 → bullish
  const momSignal = momentumCrossover;

  // --- Volatility compression: vol5 / vol20 ---
  const ret5 = returns.slice(-5);
  const ret20 = returns.slice(-20);
  const mean5 = ret5.reduce((s, v) => s + v, 0) / ret5.length;
  const mean20 = ret20.reduce((s, v) => s + v, 0) / ret20.length;
  const vol5 = Math.sqrt(ret5.reduce((s, v) => s + (v - mean5) ** 2, 0) / ret5.length);
  const vol20 = Math.sqrt(ret20.reduce((s, v) => s + (v - mean20) ** 2, 0) / ret20.length);
  const volCompression = vol20 > 0 ? vol5 / vol20 : 1;
  // Low vol compression (<0.7) + trending = breakout imminent → amplify directional signal
  const volAmplifier = volCompression < 0.7 ? 1.5 : volCompression > 1.5 ? 0.5 : 1.0;

  // --- Consensus: count how many signals agree on direction ---
  const bullSignals = [
    mrSignal > 0.3 ? 1 : mrSignal < -0.3 ? -1 : 0,
    momSignal > 0.005 ? 1 : momSignal < -0.005 ? -1 : 0,
    // Vol compression direction: use recent 5d return direction
    ret5.reduce((s, v) => s + v, 0) > 0 ? 1 : -1,
  ] as const;

  const bullCount = bullSignals.filter(s => s > 0).length;
  const bearCount = bullSignals.filter(s => s < 0).length;
  const consensus = Math.max(bullCount, bearCount);

  // --- Combined adjustment: weighted average, only when ≥2 agree ---
  let adjustment = 0;
  if (consensus >= 2) {
    const direction = bullCount >= bearCount ? 1 : -1;
    // Weight: MR 0.4, Momentum 0.4, Vol 0.2
    const rawAdj = direction * (
      0.4 * Math.min(Math.abs(mrSignal), 2) * 0.001 +
      0.4 * Math.min(Math.abs(momSignal), 0.05) * 0.04 +
      0.2 * 0.001
    ) * volAmplifier;
    adjustment = Math.max(-0.004, Math.min(0.004, rawAdj));
  }

  return { meanReversionZ, momentumCrossover, volCompression, adjustment, consensus };
}
// ---------------------------------------------------------------------------
// Regime-conditional up-rate (Idea T, Round 4)
// ---------------------------------------------------------------------------

/**
 * Compute empirical P(up | regime) for each regime state from historical data.
 *
 * For each day where regime=R, look forward `horizon` days and check if the
 * cumulative return was positive. This gives the ACTUAL frequency of "up"
 * outcomes following each regime — much more accurate than the lossy
 * regime → mean return → Student-t survival mapping.
 *
 * Returns a map from regime state to P(up|regime). When combined with the
 * n-step transition probabilities, this yields:
 *   P(up) = Σ P(regime_i at horizon) × P(up | regime_i)
 */
export function computeRegimeUpRates(
  regimeSeq: RegimeState[],
  logReturns: number[],
  horizon: number,
  decayRate?: number,
): Record<RegimeState, number> {
  const counts: Record<RegimeState, { up: number; total: number }> = {
    bull: { up: 0, total: 0 },
    bear: { up: 0, total: 0 },
    sideways: { up: 0, total: 0 },
  };

  // regimeSeq[i] corresponds to logReturns[i]. Look forward `horizon` days.
  // Use log returns so cumulative = Σ log(1+r) is the true cumulative log return
  // (sign of the sum equals sign of the cumulative simple return). Summing simple
  // returns is only an approximation and biases the up-rate downward at multi-day
  // horizons because compound = exp(Σ log(1+r)) − 1 ≠ Σ r when |r| is non-trivial.
  const maxStart = Math.min(regimeSeq.length, logReturns.length) - horizon;
  for (let i = 0; i < maxStart; i++) {
    const regime = regimeSeq[i];
    // Cumulative log return over next `horizon` days (starts at i+1 — future returns only)
    let cumLogReturn = 0;
    for (let j = i + 1; j <= i + horizon; j++) {
      cumLogReturn += logReturns[j];
    }

    // Bounded exponential weighting: recent observations get more weight
    const weight = decayRate !== undefined
      ? Math.pow(decayRate, maxStart - 1 - i)
      : 1;

    counts[regime].total += weight;
    if (cumLogReturn > 0) counts[regime].up += weight;
  }

  const result = {} as Record<RegimeState, number>;
  for (const state of REGIME_STATES) {
    result[state] = counts[state].total > 0
      ? counts[state].up / counts[state].total
      : 0.5; // no data → uninformative
  }
  return result;
}
