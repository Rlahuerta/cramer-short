import { describe, test, expect } from 'bun:test';

import { reverseDcfTool } from './reverse-dcf.js';
import { computeFairValuePerShare } from '../../utils/finance/dcf.js';

function parseResult(raw: unknown): { data: Record<string, unknown> } {
  return JSON.parse(raw as string) as { data: Record<string, unknown> };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('reverseDcfTool — known solutions', () => {
  test('recovers growth rate from synthetic DCF with g=0.10', async () => {
    const baseFcf = 100;
    const wacc = 0.10;
    const terminalGrowthRate = 0.025;
    const years = 5;
    const decay = 0.05;
    const netDebt = 50;
    const dilutedShares = 10;

    // Forward-compute fair value at g=0.10
    const { fairValuePerShare } = computeFairValuePerShare({
      baseFcf,
      wacc,
      growthRate: 0.10,
      terminalGrowthRate,
      years,
      decay,
      netDebt,
      dilutedShares,
    });

    // Invert
    const result = await reverseDcfTool.invoke({
      ticker: 'SYNTH',
      current_price: fairValuePerShare,
      base_fcf: baseFcf,
      wacc,
      terminal_growth_rate: terminalGrowthRate,
      years,
      decay,
      net_debt: netDebt,
      diluted_shares: dilutedShares,
    });
    const parsed = parseResult(result);
    expect(parsed.data.resultStatus).toBe('converged');
    expect(parsed.data.impliedGrowthRate).toBeCloseTo(0.10, 3);
  });

  test('recovers negative growth rate g=-0.05', async () => {
    const baseFcf = 200;
    const wacc = 0.12;
    const terminalGrowthRate = 0.02;
    const years = 5;
    const decay = 0.05;
    const netDebt = 0;
    const dilutedShares = 1;

    const { fairValuePerShare } = computeFairValuePerShare({
      baseFcf,
      wacc,
      growthRate: -0.05,
      terminalGrowthRate,
      years,
      decay,
      netDebt,
      dilutedShares,
    });

    const result = await reverseDcfTool.invoke({
      ticker: 'NGROW',
      current_price: fairValuePerShare,
      base_fcf: baseFcf,
      wacc,
      terminal_growth_rate: terminalGrowthRate,
      years,
      decay,
      net_debt: netDebt,
      diluted_shares: dilutedShares,
    });
    const parsed = parseResult(result);
    expect(parsed.data.resultStatus).toBe('converged');
    expect(parsed.data.impliedGrowthRate).toBeCloseTo(-0.05, 3);
  });
});

describe('reverseDcfTool — boundary and edge cases', () => {
  test('boundary at zero growth (price equals no-growth FCF-only value)', async () => {
    const baseFcf = 100;
    const wacc = 0.10;
    const terminalGrowthRate = 0.025;
    const years = 5;
    const decay = 0.05;
    const netDebt = 0;
    const dilutedShares = 10;

    const { fairValuePerShare } = computeFairValuePerShare({
      baseFcf,
      wacc,
      growthRate: 0,
      terminalGrowthRate,
      years,
      decay,
      netDebt,
      dilutedShares,
    });

    const result = await reverseDcfTool.invoke({
      ticker: 'ZERO',
      current_price: fairValuePerShare,
      base_fcf: baseFcf,
      wacc,
      terminal_growth_rate: terminalGrowthRate,
      years,
      decay,
      net_debt: netDebt,
      diluted_shares: dilutedShares,
    });
    const parsed = parseResult(result);
    expect(parsed.data.resultStatus).toBe('converged');
    expect(parsed.data.impliedGrowthRate).toBeCloseTo(0, 3);
  });

  test('out of range when price is impossibly high', async () => {
    const result = await reverseDcfTool.invoke({
      ticker: 'HIGH',
      current_price: 1e12,
      base_fcf: 100,
      wacc: 0.10,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
      net_debt: 0,
      diluted_shares: 1,
    });
    const parsed = parseResult(result);
    expect(parsed.data.resultStatus).toBe('out_of_range');
    expect(parsed.data.impliedGrowthRate).toBeNull();
    expect(parsed.data.impliedTv).toBeNull();
  });

  test('no solution when current price is non-positive', async () => {
    const result = await reverseDcfTool.invoke({
      ticker: 'BAD',
      current_price: -100,
      base_fcf: 100,
      wacc: 0.10,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
      net_debt: 0,
      diluted_shares: 1,
    });
    const parsed = parseResult(result);
    expect(parsed.data.resultStatus).toBe('no_solution');
    expect(parsed.data.impliedGrowthRate).toBeNull();
    expect(parsed.data.impliedTv).toBeNull();
    expect(Array.isArray(parsed.data.impliedFcfProjection)).toBe(true);
    expect((parsed.data.impliedFcfProjection as unknown[]).length).toBe(0);
  });

  test('ticker is normalized to uppercase', async () => {
    const baseFcf = 100;
    const wacc = 0.10;
    const terminalGrowthRate = 0.025;
    const years = 5;
    const decay = 0.05;
    const netDebt = 0;
    const dilutedShares = 1;

    const { fairValuePerShare } = computeFairValuePerShare({
      baseFcf,
      wacc,
      growthRate: 0.05,
      terminalGrowthRate,
      years,
      decay,
      netDebt,
      dilutedShares,
    });

    const result = await reverseDcfTool.invoke({
      ticker: 'lowercase',
      current_price: fairValuePerShare,
      base_fcf: baseFcf,
      wacc,
      terminal_growth_rate: terminalGrowthRate,
      years,
      decay,
      net_debt: netDebt,
      diluted_shares: dilutedShares,
    });
    const parsed = parseResult(result);
    expect(parsed.data.ticker).toBe('LOWERCASE');
  });
});

describe('reverseDcfTool — structural', () => {
  test('tool name is reverse_dcf', () => {
    expect(reverseDcfTool.name).toBe('reverse_dcf');
  });

  test('returns all expected output fields', async () => {
    const baseFcf = 100;
    const wacc = 0.10;
    const terminalGrowthRate = 0.025;
    const years = 5;
    const decay = 0.05;
    const netDebt = 0;
    const dilutedShares = 1;

    const { fairValuePerShare } = computeFairValuePerShare({
      baseFcf,
      wacc,
      growthRate: 0.08,
      terminalGrowthRate,
      years,
      decay,
      netDebt,
      dilutedShares,
    });

    const result = await reverseDcfTool.invoke({
      ticker: 'FIELD',
      current_price: fairValuePerShare,
      base_fcf: baseFcf,
      wacc,
      terminal_growth_rate: terminalGrowthRate,
      years,
      decay,
      net_debt: netDebt,
      diluted_shares: dilutedShares,
    });
    const parsed = parseResult(result);
    const expected = [
      'ticker',
      'currentPrice',
      'impliedGrowthRate',
      'impliedFcfProjection',
      'impliedTv',
      'resultStatus',
    ];
    for (const field of expected) {
      expect(parsed.data).toHaveProperty(field);
    }
  });
});
