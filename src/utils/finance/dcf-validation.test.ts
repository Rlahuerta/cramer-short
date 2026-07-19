import { describe, it, expect } from 'bun:test';
import {
  validateWaccGreaterThanGrowth,
  validateTerminalGrowthVsRfr,
  validateTerminalGrowthCap,
  validateTvProportion,
  validateWaccVsRoic,
  validateEvSanity,
  validateMarketValueWeights,
  runDcfValidation,
  type DcfValidationInput,
} from './dcf-validation.js';

// ─── validateWaccGreaterThanGrowth ────────────────────────────────────────────

describe('validateWaccGreaterThanGrowth', () => {
  it('passes when WACC exceeds terminal growth by more than 0.5%', () => {
    const result = validateWaccGreaterThanGrowth(0.08, 0.025);
    expect(result.ok).toBe(true);
    expect(result.message).toContain('WACC');
  });

  it('fails on exact boundary (wacc = terminalGrowth + 0.005)', () => {
    const result = validateWaccGreaterThanGrowth(0.025 + 0.005, 0.025);
    expect(result.ok).toBe(false);
    expect(result.message).toContain('WACC');
  });

  it('fails when margin is insufficient (wacc = terminalGrowth + 0.003)', () => {
    const result = validateWaccGreaterThanGrowth(0.028, 0.025);
    expect(result.ok).toBe(false);
  });
});

// ─── validateTerminalGrowthVsRfr ────────────────────────────────────────────

describe('validateTerminalGrowthVsRfr', () => {
  it('passes when terminal growth does not exceed risk-free rate', () => {
    const result = validateTerminalGrowthVsRfr(0.025, 0.043);
    expect(result.ok).toBe(true);
  });

  it('fails when terminal growth exceeds risk-free rate', () => {
    const result = validateTerminalGrowthVsRfr(0.05, 0.043);
    expect(result.ok).toBe(false);
    expect(result.message).toContain('risk-free');
  });
});

// ─── validateTerminalGrowthCap ──────────────────────────────────────────────

describe('validateTerminalGrowthCap', () => {
  it('passes for developed market at 3.0%', () => {
    const result = validateTerminalGrowthCap(0.03, true);
    expect(result.ok).toBe(true);
  });

  it('fails for developed market at 4.0% (cap is 3.5%)', () => {
    const result = validateTerminalGrowthCap(0.04, true);
    expect(result.ok).toBe(false);
    expect(result.message).toContain('3.5%');
  });

  it('passes for EM at 4.0%', () => {
    const result = validateTerminalGrowthCap(0.04, false);
    expect(result.ok).toBe(true);
  });

  it('fails for EM at 4.1% (cap is 4.0%)', () => {
    const result = validateTerminalGrowthCap(0.041, false);
    expect(result.ok).toBe(false);
    expect(result.message).toContain('4.0%');
  });
});

// ─── validateTvProportion ───────────────────────────────────────────────────

describe('validateTvProportion', () => {
  it('passes silently when TV/EV is 0.84', () => {
    const result = validateTvProportion(840, 1000);
    expect(result.ok).toBe(true);
    expect(result.message).toBe('');
    expect(result.ratio).toBeCloseTo(0.84, 4);
  });

  it('warns when TV/EV is 0.87', () => {
    const result = validateTvProportion(870, 1000);
    expect(result.ok).toBe(true);
    expect(result.message).toContain('85%');
    expect(result.ratio).toBeCloseTo(0.87, 4);
  });

  it('fails when TV/EV is 0.91', () => {
    const result = validateTvProportion(910, 1000);
    expect(result.ok).toBe(false);
    expect(result.message).toContain('90%');
    expect(result.ratio).toBeCloseTo(0.91, 4);
  });

  it('handles zero or negative enterpriseValue gracefully (ratio=0, ok=true)', () => {
    const result = validateTvProportion(100, 0);
    expect(result.ratio).toBe(0);
    expect(result.ok).toBe(false);
  });
});

// ─── validateWaccVsRoic ─────────────────────────────────────────────────────

describe('validateWaccVsRoic', () => {
  it('passes positively when WACC <= ROIC (value-creating)', () => {
    const result = validateWaccVsRoic(0.08, 0.10);
    expect(result.ok).toBe(true);
    expect(result.message).not.toContain('destruction');
  });

  it('warns when WACC > ROIC (value destruction)', () => {
    const result = validateWaccVsRoic(0.10, 0.08);
    expect(result.ok).toBe(true);
    expect(result.message).toContain('destruction');
  });
});

// ─── validateEvSanity ─────────────────────────────────────────────────────────

describe('validateEvSanity', () => {
  it('passes when deviation is 23.08%', () => {
    const result = validateEvSanity(100, 130);
    expect(result.ok).toBe(true);
    expect(result.deviationPct).toBeCloseTo(23.0769, 2);
  });

  it('fails when deviation exceeds 30%', () => {
    const result = validateEvSanity(100, 200);
    expect(result.ok).toBe(false);
    expect(result.deviationPct).toBeCloseTo(50, 2);
  });
});

// ─── validateMarketValueWeights ───────────────────────────────────────────────

describe('validateMarketValueWeights', () => {
  it('warns when source is book', () => {
    const result = validateMarketValueWeights(0.5, 'book');
    expect(result.ok).toBe(true);
    expect(result.message).toContain('book');
  });

  it('passes neutrally when source is market', () => {
    const result = validateMarketValueWeights(0.5, 'market');
    expect(result.ok).toBe(true);
    expect(result.message).not.toContain('book');
  });

  it('passes neutrally when source is unknown', () => {
    const result = validateMarketValueWeights(0.5, 'unknown');
    expect(result.ok).toBe(true);
  });
});

// ─── runDcfValidation (aggregator) ──────────────────────────────────────────

describe('runDcfValidation', () => {
  const happyInput: DcfValidationInput = {
    wacc: 0.08,
    terminalGrowth: 0.025,
    riskFreeRate: 0.043,
    isDevelopedMarket: true,
    tvPV: 500,
    enterpriseValue: 1000,
    roic: 0.10,
    computedEv: 105,
    reportedEv: 130,
    debtToEquity: 0.5,
    source: 'market',
  };

  it('happy path: ok=true, no errors', () => {
    const result = runDcfValidation(happyInput);
    expect(result.ok).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('aggregates one error when a single check fails', () => {
    const bad = { ...happyInput, tvPV: 910 };
    const result = runDcfValidation(bad);
    expect(result.ok).toBe(false);
    expect(result.errors.length).toBeGreaterThanOrEqual(1);
  });

  it('collects warnings from ok=true validators with concerns', () => {
    const warn = { ...happyInput, roic: 0.06 }; // WACC > ROIC triggers warning
    const result = runDcfValidation(warn);
    expect(result.warnings.length).toBeGreaterThanOrEqual(1);
  });
});
