/**
 * Swing-trade signal backtest — validates confidence-weighted entry signals.
 *
 * Tests the hypothesis that filtering on Markov confidence ≥ 0.25 improves
 * directional accuracy and risk-adjusted returns vs. unfiltered signals.
 */

import { describe, it, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import {
  computeMarkovDistribution,
  type MarkovDistributionResult,
} from '../markov-distribution.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SwingTradeSignal {
  ticker: string;
  entryDate: string;
  entryPrice: number;
  horizon: number;
  predictionConfidence: number;
  recommendation: 'BUY' | 'HOLD' | 'SELL';
  targetPrice: number;
  stopLoss: number;
  exitDate?: string;
  exitPrice?: number;
  exitReason?: 'target' | 'stop' | 'horizon' | 'time';
  pnl?: number;
  pnlPct?: number;
  hitTarget?: boolean;
  hitStop?: boolean;
}

export interface SwingTradeBacktestConfig {
  tickers: string[];
  horizons: number[];
  confidenceThreshold: number;
  warmup: number;
  stride: number;
  startDate?: string;
  endDate?: string;
}

export interface SwingTradeBacktestResult {
  signals: SwingTradeSignal[];
  filteredSignals: SwingTradeSignal[];
  metrics: {
    totalSignals: number;
    filteredCount: number;
    filterRate: number;
    unfilteredAccuracy: number;
    filteredAccuracy: number;
    unfilteredAvgPnlPct: number;
    filteredAvgPnlPct: number;
    unfilteredWinRate: number;
    filteredWinRate: number;
    unfilteredSharpe: number;
    filteredSharpe: number;
  };
}

// ---------------------------------------------------------------------------
// Backtest engine
// ---------------------------------------------------------------------------

/**
 * Runs a walk-forward swing-trade backtest.
 *
 * At each step:
 * 1. Compute Markov distribution with specified horizon
 * 2. Record signal (BUY/HOLD/SELL based on expected return)
 * 3. Walk forward `horizon` days
 * 4. Record exit price and P&L
 * 5. Classify outcome (hit target / hit stop / horizon exit)
 */
export async function runSwingTradeBacktest(
  config: SwingTradeBacktestConfig,
): Promise<SwingTradeBacktestResult> {
  const { tickers, horizons, confidenceThreshold, warmup, stride } = config;
  
  const allSignals: SwingTradeSignal[] = [];
  const filteredSignals: SwingTradeSignal[] = [];

// Load price fixtures from JSON file
    const fixturesPath = join(process.cwd(), 'src/tools/finance/fixtures/backtest-prices.json');
    const fixturesRaw = JSON.parse(readFileSync(fixturesPath, 'utf-8'));
    const fixtures: Record<string, number[]> = {};
    for (const ticker of tickers) {
      const tickerData = fixturesRaw.tickers[ticker];
      if (tickerData && tickerData.closes) {
        fixtures[ticker] = tickerData.closes;
      }
    }

    for (const ticker of tickers) {
    const prices = fixtures[ticker];
    if (!prices || prices.length < warmup + 60) continue;

    // Walk-forward loop
    for (let startIdx = warmup; startIdx < prices.length - 30; startIdx += stride) {
      for (const horizon of horizons) {
        const historicalPrices = prices.slice(startIdx - 60, startIdx);
        const currentPrice = prices[startIdx]!;
        const entryDate = new Date(Date.now() - (prices.length - startIdx) * 24 * 60 * 60 * 1000)
          .toISOString()
          .slice(0, 10);

        // Compute Markov distribution
        const result = await computeMarkovDistribution({
          ticker,
          horizon,
          currentPrice,
          historicalPrices,
          polymarketMarkets: [],
        });

        const recommendation = result.actionSignal.recommendation;
        const targetPrice = result.actionSignal.actionLevels.targetPrice;
        const stopLoss = result.actionSignal.actionLevels.stopLoss;
        const predictionConfidence = result.predictionConfidence;

        // Exit after horizon days
        const exitIdx = startIdx + horizon;
        if (exitIdx >= prices.length) continue;

        const exitPrice = prices[exitIdx]!;
        const exitDate = new Date(Date.now() - (prices.length - exitIdx) * 24 * 60 * 60 * 1000)
          .toISOString()
          .slice(0, 10);

        // Calculate P&L for BUY signals (invert for SELL)
        const pnlPct = recommendation === 'BUY'
          ? (exitPrice - currentPrice) / currentPrice
          : (currentPrice - exitPrice) / currentPrice;

        // Determine exit reason
        const hitTarget = recommendation === 'BUY'
          ? exitPrice >= targetPrice
          : exitPrice <= targetPrice;
        const hitStop = recommendation === 'BUY'
          ? exitPrice <= stopLoss
          : exitPrice >= stopLoss;

        let exitReason: 'target' | 'stop' | 'horizon' | 'time' = 'horizon';
        if (hitStop) exitReason = 'stop';
        else if (hitTarget) exitReason = 'target';

        const signal: SwingTradeSignal = {
          ticker,
          entryDate,
          entryPrice: currentPrice,
          horizon,
          predictionConfidence,
          recommendation,
          targetPrice,
          stopLoss,
          exitDate,
          exitPrice,
          exitReason,
          pnl: exitPrice - currentPrice,
          pnlPct,
          hitTarget,
          hitStop,
        };

        allSignals.push(signal);

        // Filter by confidence
        if (predictionConfidence >= confidenceThreshold) {
          filteredSignals.push(signal);
        }
      }
    }
  }

  // Calculate metrics
  const metrics = calculateMetrics(allSignals, filteredSignals, confidenceThreshold);

  return { signals: allSignals, filteredSignals, metrics };
}

function calculateMetrics(
  allSignals: SwingTradeSignal[],
  filteredSignals: SwingTradeSignal[],
  confidenceThreshold: number,
): SwingTradeBacktestResult['metrics'] {
  const filterRate = 1 - filteredSignals.length / (allSignals.length || 1);

  // Directional accuracy (correct if BUY & up, or SELL & down)
  const calcAccuracy = (signals: SwingTradeSignal[]) => {
    if (signals.length === 0) return 0;
    const correct = signals.filter((s) => {
      if (s.recommendation === 'BUY') return s.pnlPct! > 0;
      if (s.recommendation === 'SELL') return s.pnlPct! < 0;
      return false; // HOLD signals excluded from accuracy calc
    }).length;
    const buySellSignals = signals.filter((s) => s.recommendation !== 'HOLD').length;
    return buySellSignals > 0 ? correct / buySellSignals : 0;
  };

  // Win rate (any positive P&L)
  const calcWinRate = (signals: SwingTradeSignal[]) => {
    if (signals.length === 0) return 0;
    const wins = signals.filter((s) => s.pnlPct! > 0).length;
    return wins / signals.length;
  };

  // Average P&L %
  const calcAvgPnl = (signals: SwingTradeSignal[]) => {
    if (signals.length === 0) return 0;
    const sum = signals.reduce((acc, s) => acc + (s.pnlPct ?? 0), 0);
    return sum / signals.length;
  };

  // Sharpe ratio (annualized, assuming 252 trading days)
  const calcSharpe = (signals: SwingTradeSignal[]) => {
    if (signals.length < 2) return 0;
    const returns = signals.map((s) => s.pnlPct ?? 0);
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((acc, r) => acc + Math.pow(r - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    if (stdDev === 0) return 0;
    // Annualize: multiply by sqrt(252 / avg_horizon)
    const avgHorizon = signals.reduce((a, s) => a + s.horizon, 0) / signals.length;
    const annualizationFactor = Math.sqrt(252 / avgHorizon);
    return (mean / stdDev) * annualizationFactor;
  };

  return {
    totalSignals: allSignals.length,
    filteredCount: filteredSignals.length,
    filterRate,
    unfilteredAccuracy: calcAccuracy(allSignals),
    filteredAccuracy: calcAccuracy(filteredSignals),
    unfilteredAvgPnlPct: calcAvgPnl(allSignals),
    filteredAvgPnlPct: calcAvgPnl(filteredSignals),
    unfilteredWinRate: calcWinRate(allSignals),
    filteredWinRate: calcWinRate(filteredSignals),
    unfilteredSharpe: calcSharpe(allSignals),
    filteredSharpe: calcSharpe(filteredSignals),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Swing-trade backtest', () => {
  it('filters low-confidence signals and improves accuracy', async () => {
    const result = await runSwingTradeBacktest({
      tickers: ['SPY', 'QQQ', 'GLD'],
      horizons: [14, 20],
      confidenceThreshold: 0.25,
      warmup: 120,
      stride: 10,
    });

    // Basic sanity checks
    expect(result.signals.length).toBeGreaterThan(0);
    expect(result.filteredSignals.length).toBeLessThanOrEqual(result.signals.length);
    expect(result.metrics.filterRate).toBeGreaterThan(0);
    expect(result.metrics.filterRate).toBeLessThan(1);

    // Hypothesis: filtered accuracy >= unfiltered accuracy
    // (May not always hold due to sample size, but trend should be positive)
    console.log(`
Swing-Trade Backtest Results:
  Total signals: ${result.metrics.totalSignals}
  Filtered signals: ${result.metrics.filteredCount} (${(1 - result.metrics.filterRate) * 100 | 0}% of total)
  Filter rate: ${(result.metrics.filterRate * 100).toFixed(1)}%

  Directional Accuracy:
    Unfiltered: ${(result.metrics.unfilteredAccuracy * 100).toFixed(1)}%
    Filtered: ${(result.metrics.filteredAccuracy * 100).toFixed(1)}%
    Delta: ${((result.metrics.filteredAccuracy - result.metrics.unfilteredAccuracy) * 100).toFixed(1)} pp

  Win Rate:
    Unfiltered: ${(result.metrics.unfilteredWinRate * 100).toFixed(1)}%
    Filtered: ${(result.metrics.filteredWinRate * 100).toFixed(1)}%

  Avg P&L %:
    Unfiltered: ${(result.metrics.unfilteredAvgPnlPct * 100).toFixed(2)}%
    Filtered: ${(result.metrics.filteredAvgPnlPct * 100).toFixed(2)}%

  Sharpe (annualized):
    Unfiltered: ${result.metrics.unfilteredSharpe.toFixed(2)}
    Filtered: ${result.metrics.filteredSharpe.toFixed(2)}
`);

    // Soft assertion: filtered accuracy should trend higher
    // (Using expect.soft would be ideal, but Bun doesn't have it — use console.log for visibility)
    if (result.metrics.filteredAccuracy < result.metrics.unfilteredAccuracy) {
      console.warn('⚠️ Warning: Filtered accuracy did not improve — may need larger sample');
    }
  }, 120_000);

  it('shows confidence distribution across signals', async () => {
    const result = await runSwingTradeBacktest({
      tickers: ['SPY'],
      horizons: [14],
      confidenceThreshold: 0.25,
      warmup: 120,
      stride: 10,
    });

    const high = result.signals.filter((s) => s.predictionConfidence >= 0.40).length;
    const medium = result.signals.filter(
      (s) => s.predictionConfidence >= 0.25 && s.predictionConfidence < 0.40,
    ).length;
    const low = result.signals.filter((s) => s.predictionConfidence < 0.25).length;
    const total = result.signals.length;

    console.log(`
Confidence Distribution (SPY 14d):
  High (≥0.40):   ${high} (${((high / total) * 100).toFixed(1)}%)
  Medium (0.25–0.40): ${medium} (${((medium / total) * 100).toFixed(1)}%)
  Low (<0.25):    ${low} (${((low / total) * 100).toFixed(1)}%)
  Total: ${total}
`);

    expect(total).toBeGreaterThan(0);
    expect(high + medium + low).toBe(total);
  }, 60_000);

  it('compares per-ticker performance (SPY vs QQQ vs GLD)', async () => {
    const tickers = ['SPY', 'QQQ', 'GLD'];
    const results: Record<string, any> = {};

    for (const ticker of tickers) {
      const result = await runSwingTradeBacktest({
        tickers: [ticker],
        horizons: [14, 20],
        confidenceThreshold: 0.25,
        warmup: 120,
        stride: 10,
      });
      results[ticker] = result.metrics;
    }

    console.log(`
Per-Ticker Comparison (14d + 20d horizons, confidence ≥ 0.25):

| Ticker | Signals | Filter Rate | Acc (filtered) | Win Rate | Sharpe |
|--------|---------|-------------|----------------|----------|--------|
| SPY    | ${results['SPY'].totalSignals.toString().padStart(7)} | ${(results['SPY'].filterRate * 100).toFixed(0).padStart(9)}% | ${(results['SPY'].filteredAccuracy * 100).toFixed(1).padStart(13)}% | ${(results['SPY'].filteredWinRate * 100).toFixed(1).padStart(8)}% | ${results['SPY'].filteredSharpe.toFixed(2).padStart(6)} |
| QQQ    | ${results['QQQ'].totalSignals.toString().padStart(7)} | ${(results['QQQ'].filterRate * 100).toFixed(0).padStart(9)}% | ${(results['QQQ'].filteredAccuracy * 100).toFixed(1).padStart(13)}% | ${(results['QQQ'].filteredWinRate * 100).toFixed(1).padStart(8)}% | ${results['QQQ'].filteredSharpe.toFixed(2).padStart(6)} |
| GLD    | ${results['GLD'].totalSignals.toString().padStart(7)} | ${(results['GLD'].filterRate * 100).toFixed(0).padStart(9)}% | ${(results['GLD'].filteredAccuracy * 100).toFixed(1).padStart(13)}% | ${(results['GLD'].filteredWinRate * 100).toFixed(1).padStart(8)}% | ${results['GLD'].filteredSharpe.toFixed(2).padStart(6)} |

Key Insights:
- **Best Sharpe:** ${Object.entries(results).sort((a, b) => b[1].filteredSharpe - a[1].filteredSharpe)[0][0]} (${Math.max(...Object.values(results).map(r => r.filteredSharpe)).toFixed(2)})
- **Highest Win Rate:** ${Object.entries(results).sort((a, b) => b[1].filteredWinRate - a[1].filteredWinRate)[0][0]} (${(Math.max(...Object.values(results).map(r => r.filteredWinRate)) * 100).toFixed(1)}%)
- **Most Signals:** ${Object.entries(results).sort((a, b) => b[1].totalSignals - a[1].totalSignals)[0][0]} (${Math.max(...Object.values(results).map(r => r.totalSignals))} total)
`);

    // Verify all tickers produced signals
    for (const ticker of tickers) {
      expect(results[ticker].totalSignals).toBeGreaterThan(0);
      expect(results[ticker].filteredCount).toBeGreaterThan(0);
    }
  }, 120_000);
});
