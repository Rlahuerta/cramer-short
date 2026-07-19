import { describe, it, expect } from 'bun:test';
import {
  sensitivityGrid,
  formatSensitivityGrid,
  type SensitivityGrid,
} from './dcf-sensitivity.js';
import { computeFairValuePerShare, type DcfInputs } from './dcf.js';

const baseInputs: DcfInputs = {
  baseFcf: 100,
  wacc: 0.10,
  growthRate: 0.10,
  terminalGrowthRate: 0.025,
  years: 3,
  decay: 0.05,
  netDebt: 50,
  dilutedShares: 10,
};

// ─── Grid shape ───────────────────────────────────────────────────────────────

describe('grid shape', () => {
  it('produces a 5×5 grid for 5 wacc values × 5 growth values', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    expect(grid.waccValues).toHaveLength(5);
    expect(grid.growthValues).toHaveLength(5);
    expect(grid.values).toHaveLength(5);
    for (const row of grid.values) {
      expect(row).toHaveLength(5);
    }
  });
});

// ─── Monotonicity within a row (fixed WACC, varying growth) ──────────────────

describe('monotonicity within a row', () => {
  it('higher terminal growth → higher fair value for fixed WACC', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    for (let i = 0; i < grid.waccValues.length; i++) {
      for (let j = 0; j < grid.growthValues.length - 1; j++) {
        expect(grid.values[i][j + 1]).toBeGreaterThan(grid.values[i][j]);
      }
    }
  });
});

// ─── Monotonicity within a column (fixed growth, varying WACC) ────────────────

describe('monotonicity within a column', () => {
  it('higher WACC → lower fair value for fixed terminal growth', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    for (let j = 0; j < grid.growthValues.length; j++) {
      for (let i = 0; i < grid.waccValues.length - 1; i++) {
        expect(grid.values[i + 1][j]).toBeLessThan(grid.values[i][j]);
      }
    }
  });
});

// ─── Center cell matches base case ────────────────────────────────────────────

describe('center cell matches base case', () => {
  it('center cell equals computeFairValuePerShare(baseInputs)', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    // waccValues = [0.08, 0.09, 0.10, 0.11, 0.12] → center index 2
    // growthValues = [0.015, 0.02, 0.025, 0.03, 0.035] → center index 2
    const baseResult = computeFairValuePerShare(baseInputs);
    expect(grid.values[2][2]).toBeCloseTo(baseResult.fairValuePerShare, 4);
  });
});

// ─── All cells finite positive ────────────────────────────────────────────────

describe('all cells finite positive', () => {
  it('every value in the grid is a finite positive number', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    for (let i = 0; i < grid.values.length; i++) {
      for (let j = 0; j < grid.values[i].length; j++) {
        expect(Number.isFinite(grid.values[i][j])).toBe(true);
        expect(grid.values[i][j]).toBeGreaterThan(0);
      }
    }
  });
});

// ─── formatSensitivityGrid output shape ─────────────────────────────────────

describe('formatSensitivityGrid output shape', () => {
  it('returns string[][] with header row plus data rows', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    const formatted = formatSensitivityGrid(grid);
    // rows = 1 header + 5 data = 6
    expect(formatted).toHaveLength(6);
    // cols = 1 label + 5 growth values = 6
    for (const row of formatted) {
      expect(row).toHaveLength(6);
    }
  });

  it('header row starts with WACC label and contains formatted growth rates', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    const formatted = formatSensitivityGrid(grid);
    expect(formatted[0][0]).toBe('WACC \\ Terminal Growth');
    expect(formatted[0][1]).toBe('1.50');
    expect(formatted[0][2]).toBe('2.00');
    expect(formatted[0][3]).toBe('2.50');
    expect(formatted[0][4]).toBe('3.00');
    expect(formatted[0][5]).toBe('3.50');
  });

  it('data rows have WACC label and 2-decimal values', () => {
    const grid = sensitivityGrid(
      baseInputs,
      { min: 0.08, max: 0.12, step: 0.01 },
      { min: 0.015, max: 0.035, step: 0.005 },
    );
    const formatted = formatSensitivityGrid(grid);
    expect(formatted[1][0]).toBe('8.00%');
    expect(formatted[3][0]).toBe('10.00%');
    // spot-check a value cell is numeric with 2 decimals
    const val = parseFloat(formatted[1][1]);
    expect(Number.isFinite(val)).toBe(true);
    expect(formatted[1][1]).toMatch(/^\d+\.\d{2}$/);
  });
});

// ─── Adversarial: malformed / empty ranges ────────────────────────────────────

describe('adversarial ranges', () => {
  it('throws when wacc step is zero', () => {
    expect(() =>
      sensitivityGrid(baseInputs, { min: 0.08, max: 0.12, step: 0 }, { min: 0.015, max: 0.035, step: 0.005 }),
    ).toThrow('step must be positive');
  });

  it('throws when wacc step is negative', () => {
    expect(() =>
      sensitivityGrid(baseInputs, { min: 0.08, max: 0.12, step: -0.01 }, { min: 0.015, max: 0.035, step: 0.005 }),
    ).toThrow('step must be positive');
  });

  it('throws when growth step is zero', () => {
    expect(() =>
      sensitivityGrid(baseInputs, { min: 0.08, max: 0.12, step: 0.01 }, { min: 0.015, max: 0.035, step: 0 }),
    ).toThrow('step must be positive');
  });

  it('throws when min > max for wacc', () => {
    expect(() =>
      sensitivityGrid(baseInputs, { min: 0.12, max: 0.08, step: 0.01 }, { min: 0.015, max: 0.035, step: 0.005 }),
    ).toThrow('min must be less than or equal to max');
  });

  it('throws when min > max for growth', () => {
    expect(() =>
      sensitivityGrid(baseInputs, { min: 0.08, max: 0.12, step: 0.01 }, { min: 0.035, max: 0.015, step: 0.005 }),
    ).toThrow('min must be less than or equal to max');
  });
});
