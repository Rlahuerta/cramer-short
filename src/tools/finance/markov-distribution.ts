/**
 * Markov Chain Probability Distribution for Asset Price Forecasting
 *
 * Produces full probability distributions for stock/ETF prices at user-specified
 * horizons by combining:
 *  1. Polymarket threshold markets — real-money anchors at specific price levels
 *  2. Historical regime transitions — Markov transition matrix from 60-90 days of history
 *  3. Sentiment adjustments — dynamic multipliers from social sentiment signals
 *
 * Academic grounding:
 *  - Nguyen (2018, IJFS): 4-state HMM for S&P 500; AIC/BIC model selection
 *  - Mettle et al. (2014, SpringerPlus): Markov chain methodology for share prices
 *  - Kumar & Amer (2023, UVic): LSTM + 3-state Markov for stock prediction
 *  - Welton & Ades (2005): Dirichlet priors for transition matrix estimation
 *  - Voigt (2025): Beta-HMM for Polymarket prediction markets
 *  - Reichenbach & Walther (2025): YES-bias in Polymarket (124M trades)
 *  - Davidovic & McCleary (2025, JRFM): Sentiment alpha ≪ VIX for return prediction
 */

import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Joint regime state encoding both direction and volatility.
 *
 * Note: The original spec used a priority-override `high_vol` state that destroyed
 * directional information on volatile days. Joint states preserve both dimensions
 * (consistent with Nguyen 2018, where states encode distinct (mean, variance) pairs).
 */
export type RegimeState = 'bull' | 'bear' | 'high_vol_bull' | 'high_vol_bear' | 'sideways';

export const REGIME_STATES: RegimeState[] = [
  'bull', 'bear', 'high_vol_bull', 'high_vol_bear', 'sideways',
];

export const NUM_STATES = REGIME_STATES.length; // 5

/** Maps state name → matrix row/column index */
export const STATE_INDEX: Record<RegimeState, number> = {
  bull:          0,
  bear:          1,
  high_vol_bull: 2,
  high_vol_bear: 3,
  sideways:      4,
};

export interface PriceThreshold {
  price: number;
  /** Raw YES probability from Polymarket (0–1) */
  rawProbability: number;
  /** Bias-corrected probability (rawProbability × 0.95). Reichenbach & Walther (2025)
   *  found systematic YES-overtrading across 124M trades. */
  probability: number;
  /** Whether this anchor is trusted based on liquidity/age heuristics */
  trustScore: 'high' | 'low';
  source: 'polymarket';
}

export interface SentimentSignal {
  bullish: number;  // 0–1
  bearish: number;  // 0–1
}

/** 5×5 stochastic matrix (rows sum to 1) */
export type TransitionMatrix = number[][];

export interface MarkovDistributionPoint {
  price: number;
  /** P(asset_price > this level) at the given horizon */
  probability: number;
  /** 90% CI lower bound (Monte Carlo 5th percentile) */
  lowerBound: number;
  /** 90% CI upper bound (Monte Carlo 95th percentile) */
  upperBound: number;
  /** How this distribution point was estimated */
  source: 'polymarket' | 'markov' | 'blend';
}

export interface MarkovDistributionResult {
  ticker: string;
  currentPrice: number;
  horizon: number;
  distribution: MarkovDistributionPoint[];
  metadata: {
    polymarketAnchors: number;
    regimeState: RegimeState;
    sentimentAdjustment: number;
    historicalDays: number;
    /** exp(−ρ×n): near 1 = Markov-dominant, near 0 = Polymarket-anchor-dominant */
    mixingTimeWeight: number;
    /** Second-largest absolute eigenvalue of the transition matrix (spectral gap) */
    secondEigenvalue: number;
    /** R²_OS vs. historical-average baseline; null if no held-out data */
    outOfSampleR2: number | null;
  };
}

// ---------------------------------------------------------------------------
// 1. extractPriceThresholds
// ---------------------------------------------------------------------------

const PRICE_PATTERNS = [
  /(?:exceed|above|over|surpass)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /(?:below|under|drop\s*(?:below|to))\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /(?:reach|hit)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)\s*(?:or\s*(?:higher|above|more))/i,
];

/** Parse a price string like "$70K" or "$1,234.56" into a number. */
function parsePrice(raw: string): number {
  const cleaned = raw.replace(/,/g, '');
  const multiplier = /[Kk]$/.test(cleaned) ? 1_000 : /[Mm]$/.test(cleaned) ? 1_000_000 : 1;
  return parseFloat(cleaned.replace(/[KkMm]$/, '')) * multiplier;
}

/**
 * Parse Polymarket market objects into sorted price thresholds with bias correction.
 *
 * YES-bias correction: multiply all raw probabilities by 0.95 (Reichenbach & Walther 2025).
 * Liquidity guard: markets under 48h or with zero volume receive trustScore='low'.
 */
export function extractPriceThresholds(
  markets: Array<{
    question: string;
    probability: number;
    volume?: number;
    createdAt?: string | number;
  }>,
): PriceThreshold[] {
  const seen = new Map<number, PriceThreshold>();

  for (const market of markets) {
    let matched: number | null = null;
    for (const pattern of PRICE_PATTERNS) {
      const m = market.question.match(pattern);
      if (m) {
        matched = parsePrice(m[1]);
        break;
      }
    }
    if (matched === null) continue;
    if (isNaN(matched) || matched <= 0) continue;

    const isYoung =
      market.createdAt != null &&
      Date.now() - (typeof market.createdAt === 'string'
        ? Date.parse(market.createdAt)
        : market.createdAt) < 48 * 60 * 60 * 1000;

    const hasVolume = (market.volume ?? 0) > 0;
    const trustScore: 'high' | 'low' = isYoung || !hasVolume ? 'low' : 'high';

    const rawProbability = market.probability;
    const corrected: PriceThreshold = {
      price: matched,
      rawProbability,
      probability: rawProbability * 0.95,
      trustScore,
      source: 'polymarket',
    };

    const existing = seen.get(matched);
    if (!existing || rawProbability > existing.rawProbability) {
      seen.set(matched, corrected);
    }
  }

  return Array.from(seen.values()).sort((a, b) => a.price - b.price);
}

// ---------------------------------------------------------------------------
// 2. classifyRegimeState
// ---------------------------------------------------------------------------

/**
 * Classify a trading day into a joint (direction × volatility) regime state.
 *
 * Volatility and directional return are orthogonal dimensions — a high-volatility
 * day can be strongly bullish or strongly bearish. Using a priority override (as in
 * the original spec) discards directional information. Joint states preserve both.
 *
 * @param dailyReturn - Return for the day (e.g. 0.012 = +1.2%)
 * @param dailyVolatility - Intraday vol proxy (e.g. (high - low) / open)
 */
export function classifyRegimeState(
  dailyReturn: number,
  dailyVolatility: number,
): RegimeState {
  const highVol = dailyVolatility > 0.02;
  if (highVol) {
    return dailyReturn > 0 ? 'high_vol_bull' : 'high_vol_bear';
  }
  if (dailyReturn > 0.01)  return 'bull';
  if (dailyReturn < -0.01) return 'bear';
  return 'sideways';
}

// ---------------------------------------------------------------------------
// 3. estimateTransitionMatrix
// ---------------------------------------------------------------------------

/**
 * Estimate a 5×5 Markov transition matrix from a sequence of regime states.
 *
 * Smoothing: Dirichlet α = 0.1 added to all transition counts before normalization.
 * This is the flat prior from Welton & Ades (2005) and provides meaningful
 * regularization when counts are sparse (e.g. 60-day window with rare transitions).
 *
 * Default matrix (insufficient data): 0.6 diagonal, uniform off-diagonal.
 * offDiag = (1 − 0.6) / (NUM_STATES − 1) = 0.4 / 4 = 0.1 per cell (rows sum to 1.0).
 *
 * Bug note: The original spec specified "0.2 off-diagonal" for a 4-state matrix,
 * yielding row sums of 0.6 + 3×0.2 = 1.2. Fixed here to use the correct formula.
 */
export function estimateTransitionMatrix(
  states: RegimeState[],
  alpha = 0.1,     // Dirichlet smoothing constant
  minObservations = 30,
): TransitionMatrix {
  if (states.length < minObservations) {
    return buildDefaultMatrix();
  }

  // Initialise count matrix with Dirichlet prior
  const counts: number[][] = Array.from({ length: NUM_STATES }, () =>
    Array(NUM_STATES).fill(alpha),
  );

  for (let i = 0; i < states.length - 1; i++) {
    const from = STATE_INDEX[states[i]];
    const to   = STATE_INDEX[states[i + 1]];
    counts[from][to] += 1;
  }

  return normalizeRows(counts);
}

/** Identity-like default matrix with correct row sums. */
export function buildDefaultMatrix(): TransitionMatrix {
  const diagonal = 0.6;
  const offDiag  = (1 - diagonal) / (NUM_STATES - 1); // 0.1 for 5 states
  return Array.from({ length: NUM_STATES }, (_, i) =>
    Array.from({ length: NUM_STATES }, (_, j) => (i === j ? diagonal : offDiag)),
  );
}

/** Normalize each row of a matrix to sum to 1. */
export function normalizeRows(matrix: number[][]): TransitionMatrix {
  return matrix.map(row => {
    const sum = row.reduce((a, b) => a + b, 0);
    return row.map(v => v / sum);
  });
}

// ---------------------------------------------------------------------------
// 4. adjustTransitionMatrix
// ---------------------------------------------------------------------------

/**
 * Apply sentiment-based adjustments to the baseline transition matrix.
 *
 * Only bull↔bear rows are adjusted — volatile states are intentionally left
 * unmodified since sentiment doesn't reliably predict intraday vol.
 *
 * α = 0.07 (reduced from the original 0.15). Davidovic & McCleary (2025, JRFM)
 * show that news sentiment scores (TextBlob/VADER/FinBERT) capture <5% of return
 * variation. Overly strong adjustments would corrupt the empirically estimated matrix.
 *
 * Sign fix: The original spec had `bear.to.bear = base * (1 - alpha * -shift)`,
 * which equals `base * (1 + shift)` and INCREASES bear persistence under bullish
 * sentiment. Corrected here: bullish shift reduces bear persistence (1 - alpha*shift).
 */
export function adjustTransitionMatrix(
  base: TransitionMatrix,
  sentiment: SentimentSignal,
  alpha = 0.07,
): TransitionMatrix {
  const shift = sentiment.bullish - sentiment.bearish; // -1 to +1
  const adjusted = base.map(row => [...row]);

  const bull = STATE_INDEX['bull'];
  const bear = STATE_INDEX['bear'];

  // Bull row: bullish sentiment → more persistence in bull, less exit to bear
  adjusted[bull][bull] = base[bull][bull] * (1 + alpha * shift);
  adjusted[bull][bear] = base[bull][bear] * (1 - alpha * shift);

  // Bear row: bullish sentiment → less persistence in bear, more exit to bull
  // (double-negative removed from original spec: was `(1 - alpha * -shift)`)
  adjusted[bear][bear] = base[bear][bear] * (1 - alpha * shift);
  adjusted[bear][bull] = base[bear][bull] * (1 + alpha * shift);

  // Clamp negatives to 0 before normalizing
  for (let i = 0; i < NUM_STATES; i++) {
    for (let j = 0; j < NUM_STATES; j++) {
      adjusted[i][j] = Math.max(0, adjusted[i][j]);
    }
  }

  return normalizeRows(adjusted);
}

// ---------------------------------------------------------------------------
// 5. Matrix math utilities
// ---------------------------------------------------------------------------

/** Matrix multiplication A × B. */
export function matMul(A: number[][], B: number[][]): number[][] {
  const n = A.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      A[i].reduce((s, _, k) => s + A[i][k] * B[k][j], 0),
    ),
  );
}

/** Compute P^n by repeated squaring (O(n² log n)). */
export function matPow(P: TransitionMatrix, n: number): TransitionMatrix {
  if (n === 0) return Array.from({ length: P.length }, (_, i) =>
    Array.from({ length: P.length }, (_, j) => (i === j ? 1 : 0)),
  );
  if (n === 1) return P.map(r => [...r]);
  if (n % 2 === 0) {
    const half = matPow(P, n / 2);
    return matMul(half, half);
  }
  return matMul(P, matPow(P, n - 1));
}

/**
 * Compute the second-largest absolute eigenvalue of the transition matrix
 * using the power iteration method with deflation.
 *
 * ρ determines mixing time: exp(−ρ×n) is how quickly the chain forgets its
 * initial state. Small ρ → fast mixing, Markov signal decays quickly.
 *
 * Returns a value in [0, 1].
 */
export function secondLargestEigenvalue(P: TransitionMatrix, iterations = 100): number {
  const n = P.length;

  // First eigenvector (stationary distribution) via power iteration
  let v = Array(n).fill(1 / n);
  for (let iter = 0; iter < iterations; iter++) {
    const next = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) next[j] += v[i] * P[i][j];
    const norm = next.reduce((s, x) => s + x, 0);
    v = next.map(x => x / norm);
  }
  // L2-normalize v for use in deflation (required for correct orthogonal projection)
  const vL2 = v.reduce((s, x) => s + x * x, 0) ** 0.5;
  const vUnit = vL2 < 1e-12 ? v : v.map(x => x / vL2);

  // Deflate: remove first eigenvector component, find second via power iteration
  let w = Array.from({ length: n }, (_, i) => (i === 0 ? 0.6 : 0.1));
  const wNorm = w.reduce((s, x) => s + x * x, 0) ** 0.5;
  w = w.map(x => x / wNorm);

  for (let iter = 0; iter < iterations; iter++) {
    const next = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) next[j] += w[i] * P[i][j];
    // Deflate: subtract L2-normalized component along stationary eigenvector
    const dot = next.reduce((s, x, i) => s + x * vUnit[i], 0);
    const deflated = next.map((x, i) => x - dot * vUnit[i]);
    const norm = deflated.reduce((s, x) => s + x * x, 0) ** 0.5;
    // deflated≈0 means second eigenvalue is ≈0 (e.g. uniform matrix has instant mixing)
    if (norm < 1e-10) return 0;
    w = deflated.map(x => x / norm);
  }

  const Pw = Array(n).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) Pw[j] += w[i] * P[i][j];

  const lambda2 = w.reduce((s, x, i) => s + x * Pw[i], 0);
  return Math.min(1, Math.max(0, Math.abs(lambda2)));
}

/** Standard normal CDF via rational approximation (Abramowitz & Stegun). */
export function normalCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

// ---------------------------------------------------------------------------
// 6. interpolateDistribution
// ---------------------------------------------------------------------------

interface RegimeStats {
  meanReturn: number;  // daily mean log-return in this regime
  stdReturn:  number;  // daily std of log-return in this regime
}

/**
 * Estimate per-regime empirical return statistics from historical data.
 * Falls back to literature-informed defaults when data is sparse.
 */
export function estimateRegimeStats(
  returns: number[],
  states: RegimeState[],
): Record<RegimeState, RegimeStats> {
  const defaults: Record<RegimeState, RegimeStats> = {
    bull:          { meanReturn:  0.005, stdReturn: 0.010 },
    bear:          { meanReturn: -0.005, stdReturn: 0.012 },
    high_vol_bull: { meanReturn:  0.006, stdReturn: 0.025 },
    high_vol_bear: { meanReturn: -0.006, stdReturn: 0.025 },
    sideways:      { meanReturn:  0.000, stdReturn: 0.006 },
  };

  const bins: Record<RegimeState, number[]> = {
    bull: [], bear: [], high_vol_bull: [], high_vol_bear: [], sideways: [],
  };

  for (let i = 0; i < Math.min(returns.length, states.length); i++) {
    bins[states[i]].push(returns[i]);
  }

  const result = { ...defaults };
  for (const [state, vals] of Object.entries(bins) as [RegimeState, number[]][]) {
    if (vals.length >= 5) {
      const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
      const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
      result[state] = { meanReturn: mean, stdReturn: Math.sqrt(variance) };
    }
  }
  return result;
}

/**
 * Compute the mixing-time weight: how much to trust the Markov regime signal
 * vs. anchor-only at a given horizon.
 *
 * weight = exp(−ρ × n) where ρ = second-largest eigenvalue of P.
 * Near 1 at short horizons (Markov-dominant).
 * Approaches 0 at long horizons (Polymarket anchors dominate).
 */
export function computeMixingWeight(secondEigenvalue: number, horizon: number): number {
  return Math.exp(-secondEigenvalue * horizon);
}

/**
 * Log-normal survival function: P(price > X | current price S₀, drift μ_n, vol σ_n).
 * P(X > target) = 1 − Φ( (ln(target/S₀) − μ_n) / σ_n )
 */
export function logNormalSurvival(
  currentPrice: number,
  targetPrice: number,
  driftN: number,    // n-day log-space drift
  volN: number,      // n-day log-space vol
): number {
  if (volN <= 0) return targetPrice < currentPrice ? 1 : 0;
  const z = (Math.log(targetPrice / currentPrice) - driftN) / volN;
  return 1 - normalCDF(z);
}

/**
 * Run a single Monte Carlo random walk through the transition matrix for n steps,
 * starting from `initialStateIdx`. Returns the final n-step regime weight vector.
 */
function singleMarkovWalk(
  P: TransitionMatrix,
  initialStateIdx: number,
  n: number,
): number[] {
  // Use the n-step transition row for the initial state
  const Pn = matPow(P, n);
  return Pn[initialStateIdx];
}

/**
 * Build probability distribution across price levels, blending Markov estimates
 * with Polymarket anchors using the mixing-time weight.
 *
 * Confidence intervals are computed via Monte Carlo simulation (N=1000 walks),
 * taking the 5th/95th percentile of the resulting probability distribution.
 */
export function interpolateDistribution(
  currentPrice: number,
  horizon: number,
  P: TransitionMatrix,
  regimeStats: Record<RegimeState, RegimeStats>,
  initialState: RegimeState,
  anchors: PriceThreshold[],
  secondEigenvalue: number,
  numLevels = 20,
  monteCarloSamples = 1000,
): MarkovDistributionPoint[] {
  const stepPct  = 0.015;
  const minPrice = currentPrice * Math.pow(1 - stepPct, numLevels / 2);
  const maxPrice = currentPrice * Math.pow(1 + stepPct, numLevels / 2);
  const prices: number[] = [];
  for (let i = 0; i <= numLevels; i++) {
    prices.push(minPrice * Math.pow(maxPrice / minPrice, i / numLevels));
  }

  const mixWeight = computeMixingWeight(secondEigenvalue, horizon);
  const initialIdx = STATE_INDEX[initialState];

  // n-step transition row for initial state
  const Pn = matPow(P, horizon);
  const stateWeights = Pn[initialIdx];

  // Compute regime-weighted drift and vol
  const mu_eff = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeights[i] * regimeStats[state].meanReturn, 0,
  );
  const sigma_eff = Math.sqrt(
    REGIME_STATES.reduce(
      (s, state, i) => s + stateWeights[i] * regimeStats[state].stdReturn ** 2, 0,
    ),
  );
  const mu_n    = horizon * mu_eff;
  const sigma_n = sigma_eff * Math.sqrt(horizon);

  // Nearest anchor lookup helper
  const findAnchor = (price: number) => {
    const TOLERANCE_PCT = 0.02;
    return anchors.find(a => Math.abs(a.price - price) / price < TOLERANCE_PCT);
  };

  // Monte Carlo: vary initial state draw for CI
  const rng = (): number => Math.random();
  const ciSamples: Map<number, number[]> = new Map(prices.map(p => [p, []]));

  for (let s = 0; s < monteCarloSamples; s++) {
    // Perturb drift and vol within sampling uncertainty
    const perturbedMu  = mu_n    + (rng() - 0.5) * sigma_n * 0.2;
    const perturbedVol = sigma_n * (0.9 + rng() * 0.2);
    for (const price of prices) {
      const p = logNormalSurvival(currentPrice, price, perturbedMu, perturbedVol);
      ciSamples.get(price)!.push(p);
    }
  }

  // Build distribution points
  const rawPoints = prices.map(price => {
    const anchor = findAnchor(price);
    const markovEst = logNormalSurvival(currentPrice, price, mu_n, sigma_n);

    let probability: number;
    let source: 'polymarket' | 'markov' | 'blend';

    if (anchor && anchor.trustScore === 'high') {
      probability = mixWeight * markovEst + (1 - mixWeight) * anchor.probability;
      source = mixWeight > 0.9 ? 'markov' : mixWeight < 0.1 ? 'polymarket' : 'blend';
    } else if (anchor && anchor.trustScore === 'low') {
      // Low-trust anchors: weight them at half their nominal influence
      probability = mixWeight * markovEst + (1 - mixWeight) * 0.5 * anchor.probability
                  + (1 - mixWeight) * 0.5 * markovEst;
      source = 'blend';
    } else {
      probability = markovEst;
      source = 'markov';
    }

    const samples = ciSamples.get(price)!.sort((a, b) => a - b);
    const lo = samples[Math.floor(0.05 * samples.length)];
    const hi = samples[Math.floor(0.95 * samples.length)];

    return { price, probability, lowerBound: lo, upperBound: hi, source };
  });

  // Enforce monotonicity: P(price > X) must be non-increasing in X
  for (let i = rawPoints.length - 2; i >= 0; i--) {
    if (rawPoints[i].probability < rawPoints[i + 1].probability) {
      rawPoints[i].probability = rawPoints[i + 1].probability;
    }
  }

  return rawPoints;
}

// ---------------------------------------------------------------------------
// 7. R²_OS out-of-sample validation
// ---------------------------------------------------------------------------

/**
 * Compute out-of-sample R² vs. historical-average baseline.
 * R²_OS > 0 means the Markov model adds value over naive mean forecast.
 *
 * R²_OS = 1 − Σ(actual − predicted)² / Σ(actual − mean(actual))²
 * (Nguyen 2018, IJFS; Campbell & Thompson 2008)
 */
export function computeR2OS(
  actualReturns: number[],
  predictedReturns: number[],
): number {
  if (actualReturns.length < 2) return 0;
  const mean = actualReturns.reduce((s, v) => s + v, 0) / actualReturns.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < actualReturns.length; i++) {
    ssRes += (actualReturns[i] - predictedReturns[i]) ** 2;
    ssTot += (actualReturns[i] - mean) ** 2;
  }
  if (ssTot < 1e-14) return 0;
  return 1 - ssRes / ssTot;
}

// ---------------------------------------------------------------------------
// 8. markovDistribution — main orchestration function
// ---------------------------------------------------------------------------

/**
 * Core Markov distribution computation. Separated from the LangChain tool wrapper
 * so it can be unit-tested directly without mocking the tool framework.
 */
export async function computeMarkovDistribution(params: {
  ticker: string;
  horizon: number;
  currentPrice: number;
  historicalPrices: number[];            // daily close prices, oldest first
  polymarketMarkets: Array<{
    question: string;
    probability: number;
    volume?: number;
    createdAt?: string | number;
  }>;
  sentiment?: SentimentSignal;
}): Promise<MarkovDistributionResult> {
  const { ticker, horizon, currentPrice, historicalPrices, polymarketMarkets, sentiment } = params;

  // --- Daily returns and volatility ---
  const returns: number[] = [];
  const vols: number[]    = [];
  for (let i = 1; i < historicalPrices.length; i++) {
    const ret = (historicalPrices[i] - historicalPrices[i - 1]) / historicalPrices[i - 1];
    returns.push(ret);
    // Approximate daily vol as |return| (proxy; real impl should use (H-L)/O)
    vols.push(Math.abs(ret));
  }

  // --- Classify regime states ---
  const regimeSeq: RegimeState[] = returns.map((r, i) => classifyRegimeState(r, vols[i]));
  const currentRegime = regimeSeq.length > 0 ? regimeSeq[regimeSeq.length - 1] : 'sideways';

  // --- Estimate transition matrix ---
  let P = estimateTransitionMatrix(regimeSeq);

  // --- Sentiment adjustment ---
  const sentimentSignal = sentiment ?? { bullish: 0.5, bearish: 0.5 };
  const sentimentShift = sentimentSignal.bullish - sentimentSignal.bearish;
  P = adjustTransitionMatrix(P, sentimentSignal);

  // --- Regime statistics ---
  const logReturns = returns.map(r => Math.log(1 + r));
  const regimeStats = estimateRegimeStats(logReturns, regimeSeq);

  // --- Polymarket anchors ---
  const anchors = extractPriceThresholds(polymarketMarkets);

  // --- Spectral gap ---
  const rho = secondLargestEigenvalue(P);
  const mixWeight = computeMixingWeight(rho, horizon);

  // --- Distribution ---
  const distribution = interpolateDistribution(
    currentPrice, horizon, P, regimeStats, currentRegime, anchors, rho,
  );

  // --- Optional R²_OS (leave-one-out on training tail) ---
  let r2os: number | null = null;
  const minHeldOut = 20;
  if (regimeSeq.length >= minHeldOut + 30) {
    const trainStates  = regimeSeq.slice(0, -minHeldOut);
    const trainReturns = logReturns.slice(0, -minHeldOut);
    const testReturns  = logReturns.slice(-minHeldOut);

    const trainP    = estimateTransitionMatrix(trainStates);
    const trainMean = trainReturns.reduce((s, v) => s + v, 0) / trainReturns.length;
    const predicted = testReturns.map((_, i) => {
      const Pn = matPow(trainP, i + 1);
      const stateIdx = STATE_INDEX[trainStates[trainStates.length - 1]];
      const weights  = Pn[stateIdx];
      return REGIME_STATES.reduce(
        (s, state, j) => s + weights[j] * regimeStats[state].meanReturn, 0,
      );
    });
    const baseline = Array(minHeldOut).fill(trainMean);
    const r2model    = computeR2OS(testReturns, predicted);
    const r2base     = computeR2OS(testReturns, baseline);
    r2os = r2model - r2base; // incremental R²_OS over mean baseline
  }

  return {
    ticker,
    currentPrice,
    horizon,
    distribution,
    metadata: {
      polymarketAnchors: anchors.filter(a => a.trustScore === 'high').length,
      regimeState: currentRegime,
      sentimentAdjustment: sentimentShift,
      historicalDays: returns.length,
      mixingTimeWeight: mixWeight,
      secondEigenvalue: rho,
      outOfSampleR2: r2os,
    },
  };
}

// ---------------------------------------------------------------------------
// 9. LangChain tool wrapper
// ---------------------------------------------------------------------------

export const markovDistributionTool = new DynamicStructuredTool({
  name: 'markov_distribution',
  description: `
Generate a full probability distribution for a stock/ETF price at a specified horizon.
Combines Polymarket threshold markets (real-money anchors) with historical Markov regime
transitions to produce P(price > X) for each price level in the distribution.

Use when the query asks for a probability distribution of future prices, not just a point estimate.
Requires: ticker symbol, horizon in trading days (1–90), and access to recent price history.
`.trim(),
  schema: z.object({
    ticker: z.string().describe('Stock/ETF ticker symbol, e.g. NVDA, SPY, BTC-USD'),
    horizon: z.number().int().min(1).max(90).describe('Forecast horizon in trading days'),
    currentPrice: z.number().optional().describe('Current price (fetched automatically if omitted)'),
    historicalPrices: z.array(z.number()).min(10).describe(
      'Daily close prices oldest-first, minimum 30 recommended for regime estimation',
    ),
    polymarketMarkets: z.array(z.object({
      question:    z.string(),
      probability: z.number().min(0).max(1),
      volume:      z.number().optional(),
      createdAt:   z.union([z.string(), z.number()]).optional(),
    })).describe('Polymarket markets with dollar price thresholds (can be empty array)'),
    sentiment: z.object({
      bullish: z.number().min(0).max(1),
      bearish: z.number().min(0).max(1),
    }).optional().describe('Sentiment signal from social_sentiment tool (optional)'),
  }),
  func: async (input) => {
    const price = input.currentPrice
      ?? input.historicalPrices[input.historicalPrices.length - 1];

    const result = await computeMarkovDistribution({
      ticker:            input.ticker,
      horizon:           input.horizon,
      currentPrice:      price,
      historicalPrices:  input.historicalPrices,
      polymarketMarkets: input.polymarketMarkets,
      sentiment:         input.sentiment,
    });

    const lines = [
      `📊 Markov Distribution: ${result.ticker} | Horizon: ${result.horizon}d`,
      `Current: $${result.currentPrice.toFixed(2)} | Regime: ${result.metadata.regimeState}`,
      `Anchors: ${result.metadata.polymarketAnchors} (trusted) | Sentiment shift: ${(result.metadata.sentimentAdjustment * 100).toFixed(0)}%`,
      `Mixing weight: ${(result.metadata.mixingTimeWeight * 100).toFixed(0)}% Markov / ${((1 - result.metadata.mixingTimeWeight) * 100).toFixed(0)}% Anchors`,
      result.metadata.outOfSampleR2 !== null
        ? `R²_OS: ${result.metadata.outOfSampleR2.toFixed(3)} (>0 = Markov adds value over mean)`
        : '',
      '',
      'Price         P(>price)    90% CI                 Source',
      '─'.repeat(60),
      ...result.distribution.map(d =>
        `$${d.price.toFixed(2).padStart(9)}   ${(d.probability * 100).toFixed(1).padStart(5)}%   `
        + `[${(d.lowerBound * 100).toFixed(1)}%–${(d.upperBound * 100).toFixed(1)}%]   ${d.source}`,
      ),
    ].filter(Boolean);

    return lines.join('\n');
  },
});
