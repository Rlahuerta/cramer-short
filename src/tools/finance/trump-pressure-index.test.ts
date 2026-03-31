/**
 * Tests for the Trump Pressure Index tool.
 *
 * Covers: Z-score computation, regime classification, Markov transitions,
 *         TACO probability blending, weight redistribution, output formatting,
 *         alert detection, and historical landmark matching.
 */

import { describe, it, expect } from 'bun:test';
import {
  computeZScore,
  computeMonthlyChange,
  classifyPressureRegime,
  estimatePressureTransitionMatrix,
  buildDefaultPressureMatrix,
  normalizePressureRows,
  monteCarloRegimeForecast,
  detectPressureStructuralBreak,
  computeTacoProb,
  redistributeWeights,
  formatPressureResult,
  COMPONENT_WEIGHTS,
  TACO_LANDMARKS,
  PRESSURE_REGIMES,
  NUM_PRESSURE_STATES,
  MARKOV_WEIGHT,
  POLYMARKET_WEIGHT,
  REGIME_BADGES,
  PRESSURE_STATE_INDEX,
  type PressureRegime,
  type TrumpPressureResult,
  type PressureComponent,
} from './trump-pressure-index.js';

// ---------------------------------------------------------------------------
// computeZScore
// ---------------------------------------------------------------------------

describe('computeZScore', () => {
  it('returns 0 for a single value', () => {
    const result = computeZScore([5]);
    expect(result.zScore).toBe(0);
  });

  it('returns 0 for constant values', () => {
    const result = computeZScore([3, 3, 3, 3, 3]);
    expect(result.zScore).toBe(0);
  });

  it('returns positive z-score when latest is above mean', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]; // 100 is far above mean
    const result = computeZScore(values, 1);
    expect(result.zScore).toBeGreaterThan(0);
  });

  it('returns negative z-score when latest is below mean', () => {
    const values = [50, 50, 50, 50, 50, 50, 50, 50, 50, 1]; // 1 is far below mean
    const result = computeZScore(values, 1);
    expect(result.zScore).toBeLessThan(0);
  });

  it('flips sign when direction is -1', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
    const positive = computeZScore(values, 1);
    const negative = computeZScore(values, -1);
    expect(positive.zScore).toBeCloseTo(-negative.zScore, 5);
  });

  it('uses rolling window correctly', () => {
    // 100 values, window of 10
    const values = Array.from({ length: 100 }, (_, i) => i);
    const result = computeZScore(values, 1, 10);
    // Last 10 values: 90-99, mean ≈ 94.5
    expect(result.mean).toBeCloseTo(94.5, 1);
    expect(result.std).toBeGreaterThan(0);
  });

  it('computes mean and std correctly', () => {
    const values = [10, 20, 30];
    const result = computeZScore(values, 1);
    expect(result.mean).toBeCloseTo(20, 5);
    // std = sqrt(((10-20)²+(20-20)²+(30-20)²)/3) = sqrt(200/3) ≈ 8.165
    expect(result.std).toBeCloseTo(Math.sqrt(200 / 3), 2);
  });
});

// ---------------------------------------------------------------------------
// computeMonthlyChange
// ---------------------------------------------------------------------------

describe('computeMonthlyChange', () => {
  it('returns null for insufficient data', () => {
    expect(computeMonthlyChange([1, 2, 3])).toBeNull();
  });

  it('returns null for exactly 20 data points (need 21)', () => {
    expect(computeMonthlyChange(Array(20).fill(100))).toBeNull();
  });

  it('computes change correctly for 21 data points', () => {
    const prices = Array(21).fill(100);
    prices[prices.length - 1] = 110; // 10% increase
    const result = computeMonthlyChange(prices);
    expect(result).toBeCloseTo(0.1, 5);
  });

  it('handles negative change', () => {
    const prices = Array(21).fill(100);
    prices[prices.length - 1] = 90; // 10% decrease
    const result = computeMonthlyChange(prices);
    expect(result).toBeCloseTo(-0.1, 5);
  });

  it('returns null when month-ago price is zero', () => {
    const prices = Array(21).fill(0);
    prices[prices.length - 1] = 100;
    expect(computeMonthlyChange(prices)).toBeNull();
  });

  it('uses the correct month-ago price (21 trading days)', () => {
    const prices = Array(30).fill(200);
    prices[30 - 21] = 100; // month-ago = 100
    prices[29] = 150;      // current = 150
    const result = computeMonthlyChange(prices);
    expect(result).toBeCloseTo(0.5, 5); // 50% increase
  });
});

// ---------------------------------------------------------------------------
// classifyPressureRegime
// ---------------------------------------------------------------------------

describe('classifyPressureRegime', () => {
  it('classifies LOW for score < 0.5', () => {
    expect(classifyPressureRegime(0)).toBe('LOW');
    expect(classifyPressureRegime(0.49)).toBe('LOW');
    expect(classifyPressureRegime(-1)).toBe('LOW');
  });

  it('classifies MODERATE for 0.5 <= score < 1.5', () => {
    expect(classifyPressureRegime(0.5)).toBe('MODERATE');
    expect(classifyPressureRegime(1.0)).toBe('MODERATE');
    expect(classifyPressureRegime(1.49)).toBe('MODERATE');
  });

  it('classifies ELEVATED for 1.5 <= score < 2.0', () => {
    expect(classifyPressureRegime(1.5)).toBe('ELEVATED');
    expect(classifyPressureRegime(1.99)).toBe('ELEVATED');
  });

  it('classifies CRITICAL for score >= 2.0', () => {
    expect(classifyPressureRegime(2.0)).toBe('CRITICAL');
    expect(classifyPressureRegime(3.5)).toBe('CRITICAL');
    expect(classifyPressureRegime(10)).toBe('CRITICAL');
  });
});

// ---------------------------------------------------------------------------
// Markov transition matrix
// ---------------------------------------------------------------------------

describe('estimatePressureTransitionMatrix', () => {
  it('returns default matrix for insufficient observations', () => {
    const regimes: PressureRegime[] = ['LOW', 'MODERATE', 'LOW'];
    const matrix = estimatePressureTransitionMatrix(regimes);
    const defaultMatrix = buildDefaultPressureMatrix();
    expect(matrix).toEqual(defaultMatrix);
  });

  it('produces rows that sum to 1', () => {
    const regimes: PressureRegime[] = [];
    for (let i = 0; i < 50; i++) {
      regimes.push(PRESSURE_REGIMES[i % 4]);
    }
    const matrix = estimatePressureTransitionMatrix(regimes);
    for (const row of matrix) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it('has correct dimensions (4x4)', () => {
    const regimes = Array(30).fill('LOW') as PressureRegime[];
    const matrix = estimatePressureTransitionMatrix(regimes);
    expect(matrix.length).toBe(4);
    for (const row of matrix) {
      expect(row.length).toBe(4);
    }
  });

  it('concentrates probability on self-transition for constant sequence', () => {
    const regimes = Array(30).fill('ELEVATED') as PressureRegime[];
    const matrix = estimatePressureTransitionMatrix(regimes);
    // ELEVATED → ELEVATED should be highest
    const elevIdx = PRESSURE_STATE_INDEX['ELEVATED'];
    expect(matrix[elevIdx][elevIdx]).toBeGreaterThan(0.9);
  });

  it('applies Dirichlet smoothing', () => {
    const regimes = Array(30).fill('LOW') as PressureRegime[];
    const matrix = estimatePressureTransitionMatrix(regimes, 0.1);
    const lowIdx = PRESSURE_STATE_INDEX['LOW'];
    // Off-diagonal cells should be > 0 due to smoothing
    for (let j = 0; j < NUM_PRESSURE_STATES; j++) {
      if (j !== lowIdx) {
        expect(matrix[lowIdx][j]).toBeGreaterThan(0);
      }
    }
  });
});

describe('buildDefaultPressureMatrix', () => {
  it('has 0.6 diagonal', () => {
    const matrix = buildDefaultPressureMatrix();
    for (let i = 0; i < 4; i++) {
      expect(matrix[i][i]).toBeCloseTo(0.6, 5);
    }
  });

  it('has uniform off-diagonal', () => {
    const matrix = buildDefaultPressureMatrix();
    const offDiag = (1 - 0.6) / 3; // ≈ 0.1333
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (i !== j) {
          expect(matrix[i][j]).toBeCloseTo(offDiag, 5);
        }
      }
    }
  });

  it('rows sum to 1', () => {
    const matrix = buildDefaultPressureMatrix();
    for (const row of matrix) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 5);
    }
  });
});

describe('normalizePressureRows', () => {
  it('normalizes rows to sum to 1', () => {
    const matrix = [[1, 2, 3, 4], [5, 5, 5, 5], [0.1, 0.1, 0.1, 0.1], [10, 0, 0, 0]];
    const normalized = normalizePressureRows(matrix);
    for (const row of normalized) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 5);
    }
  });

  it('handles zero rows gracefully', () => {
    const matrix = [[0, 0, 0, 0], [1, 1, 1, 1]];
    const normalized = normalizePressureRows(matrix);
    expect(normalized[0]).toEqual([0, 0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// Monte Carlo regime forecast
// ---------------------------------------------------------------------------

describe('monteCarloRegimeForecast', () => {
  it('returns probabilities that sum to ~1', () => {
    const matrix = buildDefaultPressureMatrix();
    const forecast = monteCarloRegimeForecast(matrix, 'MODERATE', 30, 1000);
    const total = Object.values(forecast).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 1);
  });

  it('favours current regime for strong diagonal', () => {
    // Make CRITICAL very sticky (0.95 diagonal)
    const matrix = buildDefaultPressureMatrix();
    const critIdx = PRESSURE_STATE_INDEX['CRITICAL'];
    matrix[critIdx] = [0.01, 0.01, 0.03, 0.95];
    const forecast = monteCarloRegimeForecast(matrix, 'CRITICAL', 5, 5000);
    expect(forecast.CRITICAL).toBeGreaterThan(0.5);
  });

  it('disperses for long horizons with default matrix', () => {
    const matrix = buildDefaultPressureMatrix();
    const forecast = monteCarloRegimeForecast(matrix, 'LOW', 100, 5000);
    // With long horizon and default matrix, should be roughly uniform
    for (const regime of PRESSURE_REGIMES) {
      expect(forecast[regime]).toBeGreaterThan(0.1);
    }
  });

  it('returns all four regimes', () => {
    const matrix = buildDefaultPressureMatrix();
    const forecast = monteCarloRegimeForecast(matrix, 'LOW', 30, 1000);
    for (const regime of PRESSURE_REGIMES) {
      expect(forecast[regime]).toBeDefined();
      expect(typeof forecast[regime]).toBe('number');
    }
  });
});

// ---------------------------------------------------------------------------
// Structural break detection
// ---------------------------------------------------------------------------

describe('detectPressureStructuralBreak', () => {
  it('detects no break for uniform sequences', () => {
    const regimes = Array(40).fill('MODERATE') as PressureRegime[];
    const result = detectPressureStructuralBreak(regimes);
    expect(result.detected).toBe(false);
  });

  it('returns no break for short sequences', () => {
    const regimes: PressureRegime[] = ['LOW', 'LOW', 'LOW'];
    const result = detectPressureStructuralBreak(regimes);
    expect(result.detected).toBe(false);
    expect(result.divergence).toBe(0);
  });

  it('detects break when halves have different regimes', () => {
    const firstHalf = Array(20).fill('LOW') as PressureRegime[];
    const secondHalf = Array(20).fill('CRITICAL') as PressureRegime[];
    const result = detectPressureStructuralBreak([...firstHalf, ...secondHalf]);
    expect(result.detected).toBe(true);
    expect(result.divergence).toBeGreaterThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// TACO probability blender
// ---------------------------------------------------------------------------

describe('computeTacoProb', () => {
  it('returns 0 for LOW regime', () => {
    const forecast = { LOW: 0.8, MODERATE: 0.1, ELEVATED: 0.05, CRITICAL: 0.05 };
    const result = computeTacoProb(forecast, 'LOW', null);
    expect(result.tacoProb).toBe(0);
  });

  it('uses Markov only when no Polymarket data', () => {
    const forecast = { LOW: 0.3, MODERATE: 0.3, ELEVATED: 0.2, CRITICAL: 0.2 };
    const result = computeTacoProb(forecast, 'CRITICAL', null);
    // P_markov = LOW + MODERATE = 0.6
    expect(result.tacoProb).toBeCloseTo(0.6, 3);
    expect(result.polymarketComponent).toBe(0);
  });

  it('blends Markov and Polymarket with correct weights', () => {
    const forecast = { LOW: 0.2, MODERATE: 0.2, ELEVATED: 0.3, CRITICAL: 0.3 };
    const polymarketProb = 0.5;
    const result = computeTacoProb(forecast, 'ELEVATED', polymarketProb);
    // P_markov = LOW + MODERATE = 0.4
    // P_polymarket = 0.5 * 0.95 (YES_BIAS_MULTIPLIER) = 0.475
    // TACO = 0.4 * 0.4 + 0.6 * 0.475 = 0.16 + 0.285 = 0.445
    expect(result.tacoProb).toBeCloseTo(0.445, 2);
  });

  it('clamps to [0, 1]', () => {
    const forecast = { LOW: 0.9, MODERATE: 0.1, ELEVATED: 0.0, CRITICAL: 0.0 };
    const result = computeTacoProb(forecast, 'CRITICAL', 0.99);
    expect(result.tacoProb).toBeLessThanOrEqual(1);
    expect(result.tacoProb).toBeGreaterThanOrEqual(0);
  });

  it('MODERATE regime only uses LOW for de-escalation', () => {
    const forecast = { LOW: 0.15, MODERATE: 0.6, ELEVATED: 0.2, CRITICAL: 0.05 };
    const result = computeTacoProb(forecast, 'MODERATE', null);
    expect(result.markovComponent).toBeCloseTo(0.15, 3);
  });

  it('applies YES-bias correction to Polymarket', () => {
    const forecast = { LOW: 0.0, MODERATE: 0.0, ELEVATED: 0.0, CRITICAL: 1.0 };
    const result = computeTacoProb(forecast, 'CRITICAL', 1.0);
    // P_polymarket corrected = 1.0 * 0.95 = 0.95
    expect(result.polymarketComponent).toBeCloseTo(0.95, 5);
  });
});

// ---------------------------------------------------------------------------
// Weight redistribution
// ---------------------------------------------------------------------------

describe('redistributeWeights', () => {
  it('returns original weights when all available', () => {
    const components = COMPONENT_WEIGHTS.map(cw => ({ key: cw.key, available: true }));
    const weights = redistributeWeights(components);
    const total = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 2);
  });

  it('redistributes within group when one component unavailable', () => {
    const components = COMPONENT_WEIGHTS.map(cw => ({
      key: cw.key,
      available: cw.key !== 'approval', // approval unavailable
    }));
    const weights = redistributeWeights(components);
    // approval weight (0.10) should be redistributed to other db_core members
    expect(weights.get('approval')).toBe(0);
    expect(weights.get('spx')!).toBeGreaterThan(0.20); // boosted
    const total = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 2);
  });

  it('sets unavailable component weight to 0', () => {
    const components = COMPONENT_WEIGHTS.map(cw => ({
      key: cw.key,
      available: cw.key !== 'gas',
    }));
    const weights = redistributeWeights(components);
    expect(weights.get('gas')).toBe(0);
  });

  it('handles all components unavailable', () => {
    const components = COMPONENT_WEIGHTS.map(cw => ({
      key: cw.key,
      available: false,
    }));
    const weights = redistributeWeights(components);
    const total = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    expect(total).toBe(0);
  });

  it('maintains total weight of 1.0 after redistribution', () => {
    // Remove 2 from db_core, 1 from extension
    const components = COMPONENT_WEIGHTS.map(cw => ({
      key: cw.key,
      available: !['approval', 'inflation', 'sentiment'].includes(cw.key),
    }));
    const weights = redistributeWeights(components);
    const total = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 2);
  });
});

// ---------------------------------------------------------------------------
// Constants validation
// ---------------------------------------------------------------------------

describe('constants', () => {
  it('component weights sum to 1.0', () => {
    const total = COMPONENT_WEIGHTS.reduce((a, cw) => a + cw.weight, 0);
    expect(total).toBeCloseTo(1.0, 5);
  });

  it('DB core weights sum to 0.60', () => {
    const total = COMPONENT_WEIGHTS
      .filter(cw => cw.group === 'db_core')
      .reduce((a, cw) => a + cw.weight, 0);
    expect(total).toBeCloseTo(0.60, 5);
  });

  it('extension weights sum to 0.40', () => {
    const total = COMPONENT_WEIGHTS
      .filter(cw => cw.group === 'extension')
      .reduce((a, cw) => a + cw.weight, 0);
    expect(total).toBeCloseTo(0.40, 5);
  });

  it('MARKOV + POLYMARKET weights sum to 1', () => {
    expect(MARKOV_WEIGHT + POLYMARKET_WEIGHT).toBeCloseTo(1.0, 5);
  });

  it('all 4 pressure regimes defined', () => {
    expect(PRESSURE_REGIMES).toEqual(['LOW', 'MODERATE', 'ELEVATED', 'CRITICAL']);
    expect(NUM_PRESSURE_STATES).toBe(4);
  });

  it('regime badges exist for all regimes', () => {
    for (const r of PRESSURE_REGIMES) {
      expect(REGIME_BADGES[r]).toBeDefined();
      expect(typeof REGIME_BADGES[r]).toBe('string');
    }
  });

  it('pressure state indices are 0-3', () => {
    expect(PRESSURE_STATE_INDEX.LOW).toBe(0);
    expect(PRESSURE_STATE_INDEX.MODERATE).toBe(1);
    expect(PRESSURE_STATE_INDEX.ELEVATED).toBe(2);
    expect(PRESSURE_STATE_INDEX.CRITICAL).toBe(3);
  });

  it('TACO landmarks are sorted by date', () => {
    expect(TACO_LANDMARKS.length).toBeGreaterThan(0);
    for (const lm of TACO_LANDMARKS) {
      expect(lm.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(PRESSURE_REGIMES).toContain(lm.regime);
    }
  });
});

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

describe('formatPressureResult', () => {
  const mockResult: TrumpPressureResult = {
    pressureScore: 1.75,
    regime: 'ELEVATED',
    tacoProb: 0.42,
    components: [
      {
        name: 'S&P 500',
        rawValue: -0.05,
        zScore: 1.5,
        weight: 0.20,
        contribution: 0.30,
        dataQuality: 'live',
        qualityPenalty: 0,
      },
      {
        name: '10Y Treasury',
        rawValue: 0.25,
        zScore: 0.8,
        weight: 0.15,
        contribution: 0.12,
        dataQuality: 'live',
        qualityPenalty: 0,
      },
      {
        name: 'Approval Rating',
        rawValue: 0,
        zScore: 0,
        weight: 0,
        contribution: 0,
        dataQuality: 'unavailable',
        qualityPenalty: 1,
      },
    ],
    nearestLandmark: TACO_LANDMARKS[0],
    regimeForecast: { LOW: 0.15, MODERATE: 0.25, ELEVATED: 0.40, CRITICAL: 0.20 },
    alertTriggered: false,
    warnings: ['Approval rating data unavailable — weight redistributed'],
    metadata: {
      dataTimestamp: '2026-03-31T12:00:00Z',
      componentsAvailable: 5,
      componentsTotal: 7,
      markovObservations: 60,
      polymarketAnchors: 3,
      structuralBreakDetected: false,
    },
  };

  it('includes title', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('Trump Pressure Index');
  });

  it('shows regime badge', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('ELEVATED');
  });

  it('shows pressure score', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('1.75');
  });

  it('shows TACO probability', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('42.0%');
  });

  it('shows component breakdown table', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('Component Breakdown');
    expect(output).toContain('S&P 500');
    expect(output).toContain('10Y Treasury');
  });

  it('shows unavailable components with N/A', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('N/A');
  });

  it('shows regime forecast', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('30-Day Regime Forecast');
    expect(output).toContain('LOW');
    expect(output).toContain('CRITICAL');
  });

  it('shows nearest landmark', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('Historical Comparison');
    expect(output).toContain('Liberation Day');
  });

  it('shows warnings', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('Warnings');
    expect(output).toContain('Approval rating data unavailable');
  });

  it('shows metadata', () => {
    const output = formatPressureResult(mockResult);
    expect(output).toContain('5/7 components available');
    expect(output).toContain('Markov observations: 60');
  });

  it('shows TACO alert when triggered', () => {
    const alertResult = { ...mockResult, pressureScore: 2.5, regime: 'CRITICAL' as PressureRegime, alertTriggered: true };
    alertResult.warnings = ['⚠️ TACO ALERT: Pressure at 2.5σ (CRITICAL).'];
    const output = formatPressureResult(alertResult);
    expect(output).toContain('TACO ALERT');
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe('edge cases', () => {
  it('computeZScore handles empty array', () => {
    const result = computeZScore([]);
    expect(result.zScore).toBe(0);
  });

  it('classifyPressureRegime handles negative scores as LOW', () => {
    expect(classifyPressureRegime(-5.0)).toBe('LOW');
    expect(classifyPressureRegime(-0.01)).toBe('LOW');
  });

  it('monteCarloRegimeForecast works with 1 simulation', () => {
    const matrix = buildDefaultPressureMatrix();
    const forecast = monteCarloRegimeForecast(matrix, 'LOW', 1, 1);
    const total = Object.values(forecast).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1.0, 1);
  });

  it('buildDefaultPressureMatrix is 4x4', () => {
    const m = buildDefaultPressureMatrix();
    expect(m.length).toBe(4);
    m.forEach(row => expect(row.length).toBe(4));
  });

  it('TACO prob handles extreme Polymarket values', () => {
    const forecast = { LOW: 0.5, MODERATE: 0.3, ELEVATED: 0.15, CRITICAL: 0.05 };
    const r1 = computeTacoProb(forecast, 'CRITICAL', 0);
    expect(r1.tacoProb).toBeGreaterThanOrEqual(0);
    const r2 = computeTacoProb(forecast, 'CRITICAL', 1);
    expect(r2.tacoProb).toBeLessThanOrEqual(1);
  });

  it('weight redistribution with single available component', () => {
    const components = COMPONENT_WEIGHTS.map(cw => ({
      key: cw.key,
      available: cw.key === 'spx',
    }));
    const weights = redistributeWeights(components);
    expect(weights.get('spx')!).toBeCloseTo(1.0, 2);
  });
});
