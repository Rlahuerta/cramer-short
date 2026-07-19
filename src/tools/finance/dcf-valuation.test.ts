import { describe, test, expect } from 'bun:test';
import { dcfValuationTool } from './dcf-valuation.js';

function parseResult(raw: unknown): { data: Record<string, unknown>; sourceUrls?: string[] } {
  return JSON.parse(raw as string) as { data: Record<string, unknown>; sourceUrls?: string[] };
}

describe('dcfValuationTool', () => {
  test('tool name is dcf_valuation', () => {
    expect(dcfValuationTool.name).toBe('dcf_valuation');
  });

  test('returns fair value per share', async () => {
    const result = await dcfValuationTool.invoke({
      ticker: 'AAPL',
      base_fcf: 100,
      net_debt: 0,
      diluted_shares: 10,
      wacc: 0.10,
      growth_rate: 0.15,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
    });
    const parsed = parseResult(result);
    expect(typeof parsed.data.fairValuePerShare).toBe('number');
    expect(Number(parsed.data.fairValuePerShare)).toBeGreaterThan(0);
    expect(Number(parsed.data.fairValuePerShare)).toBeFinite();
  });

  test('WACC less than terminal growth returns validation error', async () => {
    const result = await dcfValuationTool.invoke({
      ticker: 'AAPL',
      base_fcf: 100,
      net_debt: 0,
      diluted_shares: 10,
      wacc: 0.03,
      growth_rate: 0.15,
      terminal_growth_rate: 0.035,
      years: 5,
      decay: 0.05,
    });
    const parsed = parseResult(result);
    const validation = parsed.data.validation as { errors?: string[]; warnings?: string[] };
    expect(Array.isArray(validation?.errors)).toBe(true);
    expect(
      (validation?.errors ?? []).some((e) => e.toLowerCase().includes('wacc')),
    ).toBe(true);
  });

  test('exit multiple cross-check produces divergence field', async () => {
    const result = await dcfValuationTool.invoke({
      ticker: 'AAPL',
      base_fcf: 100,
      net_debt: 0,
      diluted_shares: 10,
      wacc: 0.10,
      growth_rate: 0.15,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
      exit_multiple: 10,
    });
    const parsed = parseResult(result);
    expect(typeof parsed.data.divergencePct).toBe('number');
    expect(Number(parsed.data.divergencePct)).toBeGreaterThanOrEqual(0);
    expect(parsed.data.tvMethod).toBe('both');
  });

  test('sensitivity grid present in output', async () => {
    const result = await dcfValuationTool.invoke({
      ticker: 'AAPL',
      base_fcf: 100,
      net_debt: 0,
      diluted_shares: 10,
      wacc: 0.10,
      growth_rate: 0.15,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
    });
    const parsed = parseResult(result);
    expect(Array.isArray(parsed.data.sensitivityGrid)).toBe(true);
    expect((parsed.data.sensitivityGrid as unknown[]).length).toBeGreaterThan(0);
  });

  test('ticker is normalized to uppercase', async () => {
    const result = await dcfValuationTool.invoke({
      ticker: 'aapl',
      base_fcf: 100,
      net_debt: 0,
      diluted_shares: 10,
      wacc: 0.10,
      growth_rate: 0.15,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
    });
    const parsed = parseResult(result);
    expect(parsed.data.ticker).toBe('AAPL');
  });

  test('all expected output fields are present', async () => {
    const result = await dcfValuationTool.invoke({
      ticker: 'MSFT',
      base_fcf: 200,
      net_debt: 50,
      diluted_shares: 20,
      wacc: 0.10,
      growth_rate: 0.15,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
    });
    const parsed = parseResult(result);
    const expected = [
      'ticker',
      'fairValuePerShare',
      'enterpriseValue',
      'equityValue',
      'netDebt',
      'wacc',
      'growthRate',
      'terminalGrowthRate',
      'years',
      'tvMethod',
      'tvProportionOfEv',
      'sensitivityGrid',
      'validation',
    ];
    for (const field of expected) {
      expect(parsed.data).toHaveProperty(field);
    }
  });
});
