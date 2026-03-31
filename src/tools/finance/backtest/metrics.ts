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
  /** Prediction confidence score (0–1) for selective prediction filtering */
  confidence: number;
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
// Selective Directional Accuracy (Idea M — sHMM)
// ---------------------------------------------------------------------------

/**
 * Selective directional accuracy: only count predictions above a confidence threshold.
 * Returns both accuracy (on selected predictions) and coverage (fraction selected).
 *
 * Inspired by El-Yaniv & Pidan (NeurIPS 2011): selective prediction achieves
 * lower error by abstaining on uncertain predictions. The RC (risk-coverage)
 * trade-off curve shows monotonically decreasing error with decreasing coverage.
 */
export function selectiveDirectionalAccuracy(
  steps: BacktestStep[],
  minConfidence: number,
  holdThreshold = 0.03,
): { accuracy: number; coverage: number; selected: number; total: number } {
  if (steps.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: 0 };

  const selected = steps.filter(s => s.confidence >= minConfidence);
  if (selected.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: steps.length };

  const correct = selected.filter(s => {
    if (s.recommendation === 'BUY')  return s.actualReturn > 0;
    if (s.recommendation === 'SELL') return s.actualReturn < 0;
    return Math.abs(s.actualReturn) < holdThreshold;
  });

  return {
    accuracy: correct.length / selected.length,
    coverage: selected.length / steps.length,
    selected: selected.length,
    total: steps.length,
  };
}

/**
 * Compute the full Risk-Coverage (RC) curve: accuracy at various confidence thresholds.
 * Returns points sorted by decreasing coverage (increasing threshold).
 */
export function computeRCCurve(
  steps: BacktestStep[],
  thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
): Array<{ threshold: number; accuracy: number; coverage: number; n: number }> {
  return thresholds.map(t => {
    const result = selectiveDirectionalAccuracy(steps, t);
    return { threshold: t, accuracy: result.accuracy, coverage: result.coverage, n: result.selected };
  });
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

// ---------------------------------------------------------------------------
// Threshold Optimization (development-time calibration tool)
// ---------------------------------------------------------------------------

export interface ThresholdOptResult {
  bestBuyThreshold: number;
  bestSellThreshold: number;
  bestAccuracy: number;
  /** Accuracy at each grid point for analysis */
  grid: Array<{ buy: number; sell: number; accuracy: number }>;
}

/**
 * Grid search for optimal BUY/SELL thresholds that maximize directional accuracy.
 * Recomputes recommendations from each step's `predictedReturn` with different
 * thresholds and finds the pair that maximizes accuracy.
 *
 * This is a development-time calibration tool — NOT used at runtime.
 */
export function optimizeThresholds(
  steps: BacktestStep[],
  buyRange = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07],
  sellRange = [0.005, 0.01, 0.015, 0.02, 0.03],
  holdBand = 0.03,
): ThresholdOptResult {
  let bestAccuracy = 0;
  let bestBuy = 0.03;
  let bestSell = 0.02;
  const grid: ThresholdOptResult['grid'] = [];

  for (const buyThr of buyRange) {
    for (const sellThr of sellRange) {
      let correct = 0;
      for (const step of steps) {
        let rec: 'BUY' | 'HOLD' | 'SELL';
        if (step.predictedReturn > buyThr) {
          rec = 'BUY';
        } else if (step.predictedReturn < -sellThr) {
          rec = 'SELL';
        } else {
          rec = 'HOLD';
        }

        if (rec === 'BUY' && step.actualReturn > 0) correct++;
        else if (rec === 'SELL' && step.actualReturn < 0) correct++;
        else if (rec === 'HOLD' && Math.abs(step.actualReturn) < holdBand) correct++;
      }

      const accuracy = steps.length > 0 ? correct / steps.length : 0;
      grid.push({ buy: buyThr, sell: sellThr, accuracy });

      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
        bestBuy = buyThr;
        bestSell = sellThr;
      }
    }
  }

  return { bestBuyThreshold: bestBuy, bestSellThreshold: bestSell, bestAccuracy, grid };
}
