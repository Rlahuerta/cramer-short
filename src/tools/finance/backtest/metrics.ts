/**
 * Backtest calibration metrics for the Markov distribution model.
 *
 * All metrics operate on arrays of BacktestStep records produced by the
 * walk-forward engine. Each step records a predicted probability/CI and
 * the actual realized outcome.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BacktestStep {
  /** Date index (trading day offset from start) */
  t: number;
  /** Predicted P(price > currentPrice at t+horizon) */
  predictedProb: number;
  /** Did the actual price exceed currentPrice? (1 = yes, 0 = no) */
  actualBinary: number;
  /** Predicted expected return over the horizon */
  predictedReturn: number;
  /** Actual realized return over the horizon */
  actualReturn: number;
  /** 90% CI lower bound (price) */
  ciLower: number;
  /** 90% CI upper bound (price) */
  ciUpper: number;
  /** Realized price at t + horizon */
  realizedPrice: number;
  /** Recommendation: BUY, HOLD, or SELL */
  recommendation: 'BUY' | 'HOLD' | 'SELL';
  /** Whether the GOF test passed for this window (null if not computed) */
  gofPasses: boolean | null;
}

export interface ReliabilityBin {
  /** Lower bound of predicted probability bin (e.g. 0.0, 0.1, ...) */
  binLower: number;
  /** Upper bound of predicted probability bin */
  binUpper: number;
  /** Mean predicted probability in this bin */
  meanPredicted: number;
  /** Actual frequency of positive outcomes in this bin */
  actualFrequency: number;
  /** Number of observations in this bin */
  count: number;
}

export interface BacktestReport {
  ticker: string;
  horizon: number;
  totalSteps: number;
  brierScore: number;
  ciCoverage: number;
  directionalAccuracy: number;
  expectedReturnCorrelation: number;
  sharpness: number;
  reliabilityBins: ReliabilityBin[];
  gofPassRate: number | null;
}

// ---------------------------------------------------------------------------
// Brier Score
// ---------------------------------------------------------------------------

/**
 * Brier score: mean squared error between predicted probabilities and binary outcomes.
 * Range [0, 1] — lower is better. 0.25 = random coin flip.
 */
export function brierScore(steps: BacktestStep[]): number {
  if (steps.length === 0) return 1;
  const sum = steps.reduce((s, step) => s + (step.predictedProb - step.actualBinary) ** 2, 0);
  return sum / steps.length;
}

// ---------------------------------------------------------------------------
// Reliability Diagram (binned calibration)
// ---------------------------------------------------------------------------

/**
 * Bin predictions into deciles and compute actual frequency per bin.
 * Returns 10 bins: [0,0.1), [0.1,0.2), ..., [0.9,1.0].
 * Perfect calibration: meanPredicted ≈ actualFrequency in each bin.
 */
export function reliabilityBins(steps: BacktestStep[], numBins = 10): ReliabilityBin[] {
  const binWidth = 1 / numBins;
  const bins: ReliabilityBin[] = [];

  for (let i = 0; i < numBins; i++) {
    const lower = i * binWidth;
    const upper = (i + 1) * binWidth;
    const inBin = steps.filter(s =>
      s.predictedProb >= lower && (i === numBins - 1 ? s.predictedProb <= upper : s.predictedProb < upper),
    );

    const meanPred = inBin.length > 0
      ? inBin.reduce((s, st) => s + st.predictedProb, 0) / inBin.length
      : (lower + upper) / 2;
    const actualFreq = inBin.length > 0
      ? inBin.reduce((s, st) => s + st.actualBinary, 0) / inBin.length
      : 0;

    bins.push({
      binLower: lower,
      binUpper: upper,
      meanPredicted: meanPred,
      actualFrequency: actualFreq,
      count: inBin.length,
    });
  }

  return bins;
}

/**
 * Maximum absolute deviation between predicted and actual frequency across bins.
 * Only considers bins with ≥ minCount observations.
 */
export function maxReliabilityDeviation(bins: ReliabilityBin[], minCount = 3): number {
  const populated = bins.filter(b => b.count >= minCount);
  if (populated.length === 0) return 0;
  return Math.max(...populated.map(b => Math.abs(b.meanPredicted - b.actualFrequency)));
}

// ---------------------------------------------------------------------------
// CI Coverage
// ---------------------------------------------------------------------------

/**
 * Fraction of steps where the realized price falls within the predicted CI.
 * Target for 90% CI: ~0.90 coverage.
 */
export function ciCoverage(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const covered = steps.filter(s => s.realizedPrice >= s.ciLower && s.realizedPrice <= s.ciUpper);
  return covered.length / steps.length;
}

// ---------------------------------------------------------------------------
// Directional Accuracy
// ---------------------------------------------------------------------------

/**
 * Fraction of steps where recommendation matches actual direction.
 * BUY is correct if actualReturn > 0, SELL if actualReturn < 0.
 * HOLD is correct if |actualReturn| < 3% (roughly within noise).
 */
export function directionalAccuracy(steps: BacktestStep[], holdThreshold = 0.03): number {
  if (steps.length === 0) return 0;
  const correct = steps.filter(s => {
    if (s.recommendation === 'BUY')  return s.actualReturn > 0;
    if (s.recommendation === 'SELL') return s.actualReturn < 0;
    // HOLD is correct if price didn't move much
    return Math.abs(s.actualReturn) < holdThreshold;
  });
  return correct.length / steps.length;
}

// ---------------------------------------------------------------------------
// Expected Return Correlation
// ---------------------------------------------------------------------------

/**
 * Pearson correlation between predicted and actual returns.
 * > 0 means the model has some predictive power; > 0.1 is decent.
 */
export function expectedReturnCorrelation(steps: BacktestStep[]): number {
  if (steps.length < 3) return 0;

  const n = steps.length;
  const predMean = steps.reduce((s, st) => s + st.predictedReturn, 0) / n;
  const actMean  = steps.reduce((s, st) => s + st.actualReturn, 0) / n;

  let num = 0, denPred = 0, denAct = 0;
  for (const s of steps) {
    const dp = s.predictedReturn - predMean;
    const da = s.actualReturn - actMean;
    num += dp * da;
    denPred += dp * dp;
    denAct  += da * da;
  }

  const den = Math.sqrt(denPred * denAct);
  return den < 1e-12 ? 0 : num / den;
}

// ---------------------------------------------------------------------------
// Sharpness
// ---------------------------------------------------------------------------

/**
 * Mean CI width as a fraction of current price.
 * Narrower = sharper (more informative). Not a pass/fail metric.
 */
export function sharpness(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const widths = steps.map(s => (s.ciUpper - s.ciLower) / s.realizedPrice);
  return widths.reduce((a, b) => a + b, 0) / widths.length;
}

// ---------------------------------------------------------------------------
// GOF Pass Rate
// ---------------------------------------------------------------------------

/**
 * Fraction of walk-forward windows where the chi-squared GOF test passed.
 * null if no windows had GOF results.
 */
export function gofPassRate(steps: BacktestStep[]): number | null {
  const withGof = steps.filter(s => s.gofPasses !== null);
  if (withGof.length === 0) return null;
  return withGof.filter(s => s.gofPasses === true).length / withGof.length;
}

// ---------------------------------------------------------------------------
// Aggregate Report
// ---------------------------------------------------------------------------

export function generateReport(
  ticker: string,
  horizon: number,
  steps: BacktestStep[],
): BacktestReport {
  const bins = reliabilityBins(steps);
  return {
    ticker,
    horizon,
    totalSteps: steps.length,
    brierScore: brierScore(steps),
    ciCoverage: ciCoverage(steps),
    directionalAccuracy: directionalAccuracy(steps),
    expectedReturnCorrelation: expectedReturnCorrelation(steps),
    sharpness: sharpness(steps),
    reliabilityBins: bins,
    gofPassRate: gofPassRate(steps),
  };
}
