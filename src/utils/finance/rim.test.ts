import { describe, it, expect } from 'bun:test';
import {
  computeAbnormalEarnings,
  rimTerminalValue,
  computeRimEquityValue,
  validateCleanSurplus,
  type RimInputs,
} from './rim.js';

// ─── computeAbnormalEarnings ────────────────────────────────────────────────

describe('computeAbnormalEarnings', () => {
  it('happy path: NI=120, r_e=10%, BV_{t-1}=1000 → AE=20', () => {
    expect(computeAbnormalEarnings(120, 0.10, 1000)).toBeCloseTo(20, 4);
  });

  it('zero net income → negative abnormal earnings', () => {
    expect(computeAbnormalEarnings(0, 0.10, 1000)).toBeCloseTo(-100, 4);
  });
});

// ─── rimTerminalValue ───────────────────────────────────────────────────────

describe('rimTerminalValue', () => {
  it('happy path: AE=20, g=2%, r_e=10% → TV = 20×1.02 / (0.10−0.02) = 255', () => {
    expect(rimTerminalValue(20, 0.02, 0.10)).toBeCloseTo(255, 4);
  });

  it('throws when cost of equity <= terminal growth', () => {
    expect(() => rimTerminalValue(20, 0.10, 0.10)).toThrow();
    expect(() => rimTerminalValue(20, 0.12, 0.10)).toThrow();
  });
});

// ─── computeRimEquityValue ──────────────────────────────────────────────────

describe('computeRimEquityValue', () => {
  const base: RimInputs = {
    beginningBookValue: 1000,
    projectedNetIncome: [120],
    costOfEquity: 0.10,
    terminalGrowthRate: 0.02,
    years: 1,
    dilutedShares: 100,
  };

  it('basic happy path: 1-year horizon', () => {
    // AE = 120 − 0.10×1000 = 20
    // PV(AE) = 20 / 1.10 = 18.1818...
    // TV = 20×1.02 / (0.10−0.02) = 255
    // PV(TV) = 255 / 1.10 = 231.8181...
    // V₀ = 1000 + 18.1818 + 231.8181 = 1250
    const result = computeRimEquityValue(base);
    expect(result.equityValue).toBeCloseTo(1250, 4);
    expect(result.equityValuePerShare).toBeCloseTo(12.5, 4);
    expect(result.projectedAbnormalEarnings).toEqual([20]);
    expect(result.rimTerminalValue).toBeCloseTo(255, 4);
    expect(result.pvRimTerminalValue).toBeCloseTo(231.8182, 4);
  });

  it('zero abnormal earnings → value equals beginning book value', () => {
    // AE = 100 − 0.10×1000 = 0 → no TV, no PV stream
    const zeroAe: RimInputs = {
      ...base,
      projectedNetIncome: [100],
    };
    const result = computeRimEquityValue(zeroAe);
    expect(result.equityValue).toBeCloseTo(1000, 4);
    expect(result.equityValuePerShare).toBeCloseTo(10, 4);
  });

  it('monotonicity: higher NI → higher equity value', () => {
    const low = computeRimEquityValue({ ...base, projectedNetIncome: [100] });
    const high = computeRimEquityValue({ ...base, projectedNetIncome: [150] });
    expect(high.equityValue).toBeGreaterThan(low.equityValue);
    expect(high.equityValuePerShare).toBeGreaterThan(low.equityValuePerShare);
  });

  it('monotonicity: higher cost of equity → lower equity value', () => {
    const lowRe = computeRimEquityValue({ ...base, costOfEquity: 0.08 });
    const highRe = computeRimEquityValue({ ...base, costOfEquity: 0.15 });
    expect(highRe.equityValue).toBeLessThan(lowRe.equityValue);
  });

  it('monotonicity: higher terminal growth → higher equity value', () => {
    const lowG = computeRimEquityValue({ ...base, terminalGrowthRate: 0.01 });
    const highG = computeRimEquityValue({ ...base, terminalGrowthRate: 0.03 });
    expect(highG.equityValue).toBeGreaterThan(lowG.equityValue);
  });

  it('2-year horizon with retention: book value evolves', () => {
    // Year 0: BV=1000, NI=120, AE=20, BV₁=1120
    // Year 1: BV=1120, NI=130, AE=130−0.10×1120=18, TV=18×1.02/0.08=229.5
    // PV = 20/1.10 + 18/1.10² + 229.5/1.10²
    //    = 18.1818 + 14.8760 + 189.6694 = 222.7272
    // V₀ = 1000 + 222.7272 = 1222.7272
    const twoYear: RimInputs = {
      ...base,
      projectedNetIncome: [120, 130],
      years: 2,
    };
    const result = computeRimEquityValue(twoYear);
    expect(result.equityValue).toBeCloseTo(1222.7273, 4);
    expect(result.projectedAbnormalEarnings).toEqual([20, 18]);
  });

  it('internal consistency: equityValue = equityValuePerShare × dilutedShares', () => {
    const result = computeRimEquityValue(base);
    expect(result.equityValue).toBeCloseTo(
      result.equityValuePerShare * base.dilutedShares,
      4,
    );
  });
});

// ─── validateCleanSurplus ───────────────────────────────────────────────────

describe('validateCleanSurplus', () => {
  it('pass case: ΔBV = NI − div for every period', () => {
    // Period 1: ΔBV=100, NI=120, div=20 → 120−20=100 ✓
    // Period 2: ΔBV=110, NI=130, div=20 → 130−20=110 ✓
    expect(validateCleanSurplus([100, 110], [120, 130], [20, 20])).toBe(true);
  });

  it('failure case: clean surplus violation (ΔBV ≠ NI − div)', () => {
    // ΔBV=100, NI=120, div=10 → 120−10=110 ≠ 100
    expect(validateCleanSurplus([100], [120], [10])).toBe(false);
  });

  it('returns false when arrays have mismatched lengths', () => {
    expect(validateCleanSurplus([100, 110], [120], [20])).toBe(false);
  });

  it('returns true for empty arrays (trivially consistent)', () => {
    expect(validateCleanSurplus([], [], [])).toBe(true);
  });
});
