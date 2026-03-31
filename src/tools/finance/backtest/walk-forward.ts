/**
 * Walk-forward backtest engine for the Markov distribution model.
 *
 * Slides a window through historical price data, generates predictions at
 * each step, then compares against realized outcomes.
 */

import {
  computeMarkovDistribution,
  type MarkovDistributionResult,
} from '../markov-distribution.js';
import type { BacktestStep } from './metrics.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface WalkForwardConfig {
  /** Ticker symbol for display purposes */
  ticker: string;
  /** Daily close prices, oldest first */
  prices: number[];
  /** Forecast horizon in trading days */
  horizon: number;
  /** Minimum number of historical days before first prediction (default: 120) */
  warmup?: number;
  /** Step forward by this many days between predictions (default: 5) */
  stride?: number;
}

export interface WalkForwardResult {
  ticker: string;
  horizon: number;
  steps: BacktestStep[];
  errors: Array<{ t: number; error: string }>;
}

// ---------------------------------------------------------------------------
// Walk-forward engine
// ---------------------------------------------------------------------------

/**
 * Run a walk-forward backtest.
 *
 * At each step t (from warmup to end-horizon, stepping by stride):
 * 1. Use prices[0..t] to generate a Markov distribution
 * 2. Record predicted probabilities, CI, recommendation
 * 3. Compare against actual outcome at prices[t + horizon]
 */
export async function walkForward(config: WalkForwardConfig): Promise<WalkForwardResult> {
  const {
    ticker,
    prices,
    horizon,
    warmup = 120,
    stride = 5,
  } = config;

  const steps: BacktestStep[] = [];
  const errors: Array<{ t: number; error: string }> = [];
  const maxT = prices.length - horizon - 1;

  for (let t = warmup; t <= maxT; t += stride) {
    const histPrices = prices.slice(0, t + 1);
    const currentPrice = prices[t];
    const realizedPrice = prices[t + horizon];
    const actualReturn = (realizedPrice - currentPrice) / currentPrice;

    try {
      const result: MarkovDistributionResult = await computeMarkovDistribution({
        ticker,
        horizon,
        currentPrice,
        historicalPrices: histPrices,
        polymarketMarkets: [],  // pure Markov, no anchors
      });

      // Find P(>currentPrice) from the distribution by interpolation
      const predictedProb = interpolateSurvival(result.distribution, currentPrice);

      // Extract the 90% CI for the median price level
      const { ciLower, ciUpper } = extractCI(result.distribution, currentPrice);

      steps.push({
        t,
        predictedProb,
        actualBinary: realizedPrice > currentPrice ? 1 : 0,
        predictedReturn: result.actionSignal.expectedReturn,
        actualReturn,
        ciLower,
        ciUpper,
        realizedPrice,
        recommendation: result.actionSignal.recommendation,
        gofPasses: result.metadata.goodnessOfFit?.passes ?? null,
      });
    } catch (err) {
      errors.push({ t, error: String(err) });
    }
  }

  return { ticker, horizon, steps, errors };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Interpolate P(>target) from the distribution curve.
 * The distribution is sorted by price ascending with probability descending.
 */
function interpolateSurvival(
  dist: MarkovDistributionResult['distribution'],
  target: number,
): number {
  if (dist.length === 0) return 0.5;

  // If target is below the lowest price point, P(>target) ≈ 1
  if (target <= dist[0].price) return dist[0].probability;
  // If target is above the highest price point, P(>target) ≈ 0
  if (target >= dist[dist.length - 1].price) return dist[dist.length - 1].probability;

  // Linear interpolation between bracketing points
  for (let i = 0; i < dist.length - 1; i++) {
    if (dist[i].price <= target && target <= dist[i + 1].price) {
      const frac = (target - dist[i].price) / (dist[i + 1].price - dist[i].price);
      return dist[i].probability + frac * (dist[i + 1].probability - dist[i].probability);
    }
  }

  return 0.5; // fallback
}

/**
 * Extract 90% confidence interval bounds from the distribution.
 * Uses the price at P(>price)=0.95 as lower bound and P(>price)=0.05 as upper bound.
 */
function extractCI(
  dist: MarkovDistributionResult['distribution'],
  currentPrice: number,
): { ciLower: number; ciUpper: number } {
  if (dist.length === 0) return { ciLower: currentPrice * 0.9, ciUpper: currentPrice * 1.1 };

  let ciLower = dist[0].price;
  let ciUpper = dist[dist.length - 1].price;

  // Find price where P(>price) crosses 0.95 (lower bound) and 0.05 (upper bound)
  for (let i = 0; i < dist.length - 1; i++) {
    // Lower bound: P(>price) ≈ 0.95
    if (dist[i].probability >= 0.95 && dist[i + 1].probability < 0.95) {
      const frac = (0.95 - dist[i + 1].probability) / (dist[i].probability - dist[i + 1].probability);
      ciLower = dist[i + 1].price + frac * (dist[i].price - dist[i + 1].price);
    }
    // Upper bound: P(>price) ≈ 0.05
    if (dist[i].probability >= 0.05 && dist[i + 1].probability < 0.05) {
      const frac = (0.05 - dist[i + 1].probability) / (dist[i].probability - dist[i + 1].probability);
      ciUpper = dist[i + 1].price + frac * (dist[i].price - dist[i + 1].price);
    }
  }

  return { ciLower, ciUpper };
}
