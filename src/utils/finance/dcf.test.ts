import { describe, it, expect } from 'bun:test';
import {
  projectFcf,
  gordonGrowthTv,
  exitMultipleTv,
  discountPV,
  computeFairValuePerShare,
  computeNetDebt,
  type DcfInputs,
} from './dcf.js';

// ─── gordonGrowthTv ───────────────────────────────────────────────────────────

describe('gordonGrowthTv', () => {
  it('gordon growth TV = finalFcf * (1+g) / (wacc - g)', () => {
    // finalFcf=100, g=2.5%, wacc=8% → TV = 100*1.025/0.055 = 1863.636...
    expect(gordonGrowthTv(100, 0.025, 0.08)).toBeCloseTo(1863.6364, 4);
  });

  it('wacc less than terminal growth throws', () => {
    expect(() => gordonGrowthTv(100, 0.05, 0.04)).toThrow(
      'WACC must be greater than terminal growth rate',
    );
  });

  it('wacc equal to terminal growth throws (division by zero risk)', () => {
    expect(() => gordonGrowthTv(100, 0.05, 0.05)).toThrow(
      'WACC must be greater than terminal growth rate',
    );
  });
});

// ─── exitMultipleTv ───────────────────────────────────────────────────────────

describe('exitMultipleTv', () => {
  it('exit multiple TV = finalEbitda * multiple', () => {
    expect(exitMultipleTv(100, 12)).toBe(1200);
  });

  it('zero multiple yields zero TV', () => {
    expect(exitMultipleTv(100, 0)).toBe(0);
  });
});

// ─── projectFcf ───────────────────────────────────────────────────────────────

describe('projectFcf', () => {
  it('returns an array of length equal to years', () => {
    const fcf = projectFcf(100, 0.10, 3, 0.05);
    expect(fcf).toHaveLength(3);
  });

  it('each projected FCF is positive for positive base and growth', () => {
    const fcf = projectFcf(100, 0.10, 5, 0.05);
    for (const v of fcf) {
      expect(v).toBeGreaterThan(0);
    }
  });

  it('each year is distinct (growth changes with decay)', () => {
    const fcf = projectFcf(100, 0.10, 3, 0.05);
    expect(new Set(fcf).size).toBe(3);
  });

  it('zero growth produces flat FCF (all values equal base)', () => {
    const fcf = projectFcf(100, 0, 3, 0.05);
    for (const v of fcf) {
      expect(v).toBeCloseTo(100, 6);
    }
  });
});

// ─── discountPV ───────────────────────────────────────────────────────────────

describe('discountPV', () => {
  it('discounts cash flows at the given WACC', () => {
    const pv = discountPV([100, 110, 120], 0.10);
    expect(pv[0]).toBeCloseTo(90.9091, 4);
    expect(pv[1]).toBeCloseTo(90.9091, 4);
    expect(pv[2]).toBeCloseTo(90.1578, 4);
  });

  it('zero WACC returns original cash flows', () => {
    const pv = discountPV([100, 200, 300], 0);
    expect(pv[0]).toBe(100);
    expect(pv[1]).toBe(200);
    expect(pv[2]).toBe(300);
  });
});

// ─── computeNetDebt ───────────────────────────────────────────────────────────

describe('computeNetDebt', () => {
  it('standard net debt = totalDebt - cash - shortTermInvestments', () => {
    const nd = computeNetDebt({
      totalDebt: 500,
      operatingLeaseLiabilities: 0,
      cash: 100,
      shortTermInvestments: 50,
    });
    expect(nd).toBe(350);
  });

  it('net-cash company: negative net debt', () => {
    const nd = computeNetDebt({
      totalDebt: 50,
      operatingLeaseLiabilities: 0,
      cash: 200,
      shortTermInvestments: 100,
    });
    expect(nd).toBe(-250);
  });

  it('lease-heavy company: operating leases added to net debt', () => {
    const nd = computeNetDebt({
      totalDebt: 500,
      operatingLeaseLiabilities: 300,
      cash: 100,
      shortTermInvestments: 50,
    });
    expect(nd).toBe(650);
  });

  it('preferred stock treated as debt', () => {
    const nd = computeNetDebt({
      totalDebt: 500,
      operatingLeaseLiabilities: 0,
      cash: 100,
      shortTermInvestments: 50,
      preferredStock: 200,
    });
    expect(nd).toBe(550);
  });

  it('convertible notes face value added to net debt', () => {
    const nd = computeNetDebt({
      totalDebt: 500,
      operatingLeaseLiabilities: 0,
      cash: 100,
      shortTermInvestments: 50,
      convertiblesFace: 150,
    });
    expect(nd).toBe(500);
  });

  it('pension deficit added to net debt', () => {
    const nd = computeNetDebt({
      totalDebt: 500,
      operatingLeaseLiabilities: 0,
      cash: 100,
      shortTermInvestments: 50,
      pensionDeficit: 80,
    });
    expect(nd).toBe(430);
  });

  it('restricted cash subtracted only if provided', () => {
    const ndWith = computeNetDebt({
      totalDebt: 500,
      operatingLeaseLiabilities: 0,
      cash: 100,
      shortTermInvestments: 50,
      restrictedCash: 30,
    });
    const ndWithout = computeNetDebt({
      totalDebt: 500,
      operatingLeaseLiabilities: 0,
      cash: 100,
      shortTermInvestments: 50,
    });
    expect(ndWith).toBe(320);
    expect(ndWithout).toBe(350);
  });
});

// ─── computeFairValuePerShare ─────────────────────────────────────────────────

describe('computeFairValuePerShare', () => {
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

  it('happy path: positive netDebt, positive fair value', () => {
    const result = computeFairValuePerShare(baseInputs);
    expect(result.fairValuePerShare).toBeGreaterThan(0);
    expect(result.enterpriseValue).toBeGreaterThan(0);
    expect(result.equityValue).toBeGreaterThan(0);
    expect(result.projectedFcf).toHaveLength(3);
    expect(result.terminalValue).toBeGreaterThan(0);
    expect(result.pvTerminalValue).toBeGreaterThan(0);
    expect(result.tvMethod).toBe('gordon');
    expect(result.tvProportionOfEv).toBeGreaterThan(0);
    expect(result.tvProportionOfEv).toBeLessThan(1);
  });

  it('net-cash case: negative netDebt → fairValuePerShare > (EV/dilutedShares)', () => {
    const inputs: DcfInputs = { ...baseInputs, netDebt: -50 };
    const result = computeFairValuePerShare(inputs);
    const evPerShare = result.enterpriseValue / inputs.dilutedShares;
    expect(result.fairValuePerShare).toBeGreaterThan(evPerShare);
  });

  it('lease-heavy case: higher netDebt → lower fair value', () => {
    const lowDebt = computeFairValuePerShare({ ...baseInputs, netDebt: 50 });
    const highDebt = computeFairValuePerShare({ ...baseInputs, netDebt: 500 });
    expect(highDebt.fairValuePerShare).toBeLessThan(lowDebt.fairValuePerShare);
  });

  it('exit-multiple cross-check: tvMethod=both and divergencePct populated', () => {
    const inputs: DcfInputs = { ...baseInputs, exitMultiple: 12 };
    const result = computeFairValuePerShare(inputs);
    expect(result.tvMethod).toBe('both');
    expect(result.exitMultipleTv).toBeDefined();
    expect(result.exitMultipleTv).toBeGreaterThan(0);
    expect(result.divergencePct).toBeDefined();
    expect(result.divergencePct!).toBeGreaterThanOrEqual(0);
  });

  it('monotonicity: higher growth → higher fair value', () => {
    const low = computeFairValuePerShare({ ...baseInputs, growthRate: 0.05 });
    const high = computeFairValuePerShare({ ...baseInputs, growthRate: 0.15 });
    expect(high.fairValuePerShare).toBeGreaterThan(low.fairValuePerShare);
  });

  it('monotonicity: higher wacc → lower fair value', () => {
    const low = computeFairValuePerShare({ ...baseInputs, wacc: 0.08 });
    const high = computeFairValuePerShare({ ...baseInputs, wacc: 0.12 });
    expect(high.fairValuePerShare).toBeLessThan(low.fairValuePerShare);
  });

  it('monotonicity: higher terminalGrowth → higher fair value', () => {
    const low = computeFairValuePerShare({ ...baseInputs, terminalGrowthRate: 0.01 });
    const high = computeFairValuePerShare({ ...baseInputs, terminalGrowthRate: 0.03 });
    expect(high.fairValuePerShare).toBeGreaterThan(low.fairValuePerShare);
  });
});
