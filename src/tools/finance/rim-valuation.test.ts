import { describe, test, expect } from 'bun:test';
import { rimValuationTool } from './rim-valuation.js';

function parseResult(raw: unknown): { data: Record<string, unknown>; sourceUrls?: string[] } {
  return JSON.parse(raw as string) as { data: Record<string, unknown>; sourceUrls?: string[] };
}

describe('rimValuationTool', () => {
  test('basic invocation returns all expected fields', async () => {
    const result = await rimValuationTool.invoke({
      ticker: 'AAPL',
      cost_of_equity: 0.10,
      beginning_book_value: 1000,
      projected_net_income: [120, 130, 140],
      terminal_growth_rate: 0.02,
      diluted_shares: 100,
    });
    const parsed = parseResult(result);
    expect(parsed.data.ticker).toBe('AAPL');
    expect(typeof parsed.data.equityValuePerShare).toBe('number');
    expect(typeof parsed.data.beginningBookValuePerShare).toBe('number');
    expect(Array.isArray(parsed.data.projectedAbnormalEarnings)).toBe(true);
    expect(typeof parsed.data.rimTerminalValue).toBe('number');
    expect(typeof parsed.data.pvRimTerminalValue).toBe('number');
    expect(parsed.data.cleanSurplusOk).toBeNull();
    expect(parsed.data.validation).toBeDefined();
    const validation = parsed.data.validation as { warnings: string[]; errors: string[] };
    expect(validation.errors).toEqual([]);
    expect(validation.warnings).toEqual([]);
  });

  test('clean surplus pass case sets cleanSurplusOk true and no warnings', async () => {
    const result = await rimValuationTool.invoke({
      ticker: 'AAPL',
      cost_of_equity: 0.10,
      beginning_book_value: 1000,
      projected_net_income: [120, 130],
      dividends: [20, 20],
      book_value_changes: [100, 110],
      terminal_growth_rate: 0.02,
      diluted_shares: 100,
    });
    const parsed = parseResult(result);
    expect(parsed.data.cleanSurplusOk).toBe(true);
    const validation = parsed.data.validation as { warnings: string[]; errors: string[] };
    expect(validation.warnings).toEqual([]);
    expect(validation.errors).toEqual([]);
  });

  test('clean surplus violation warns', async () => {
    const result = await rimValuationTool.invoke({
      ticker: 'AAPL',
      cost_of_equity: 0.10,
      beginning_book_value: 1000,
      projected_net_income: [120],
      dividends: [10],
      book_value_changes: [100], // 120 - 10 = 110 != 100
      terminal_growth_rate: 0.02,
      diluted_shares: 100,
    });
    const parsed = parseResult(result);
    expect(parsed.data.cleanSurplusOk).toBe(false);
    const validation = parsed.data.validation as { warnings: string[]; errors: string[] };
    expect(validation.warnings.length).toBeGreaterThan(0);
    expect(
      validation.warnings.some((w: string) => w.toLowerCase().includes('clean surplus')),
    ).toBe(true);
    expect(validation.errors).toEqual([]);
  });

  test('cost of equity less than or equal to terminal growth returns error', async () => {
    const result = await rimValuationTool.invoke({
      ticker: 'AAPL',
      cost_of_equity: 0.03,
      beginning_book_value: 1000,
      projected_net_income: [120],
      terminal_growth_rate: 0.03,
      diluted_shares: 100,
    });
    const parsed = parseResult(result);
    expect(parsed.data.equityValuePerShare).toBeNull();
    expect(Number(parsed.data.beginningBookValuePerShare)).toBeCloseTo(10, 4);
    const validation = parsed.data.validation as { warnings: string[]; errors: string[] };
    expect(validation.errors.length).toBeGreaterThan(0);
    expect(
      validation.errors.some((e: string) => e.toLowerCase().includes('terminal growth')),
    ).toBe(true);
  });

  test('zero abnormal earnings equals book value', async () => {
    const result = await rimValuationTool.invoke({
      ticker: 'AAPL',
      cost_of_equity: 0.10,
      beginning_book_value: 1000,
      projected_net_income: [100], // 100 = 0.10 * 1000 -> AE = 0
      terminal_growth_rate: 0.02,
      diluted_shares: 100,
    });
    const parsed = parseResult(result);
    expect(Number(parsed.data.equityValuePerShare)).toBeCloseTo(10, 4); // 1000 / 100
    expect(Number(parsed.data.beginningBookValuePerShare)).toBeCloseTo(10, 4);
  });

  test('monotonicity: higher net income produces higher value', async () => {
    const low = await rimValuationTool.invoke({
      ticker: 'AAPL',
      cost_of_equity: 0.10,
      beginning_book_value: 1000,
      projected_net_income: [100],
      terminal_growth_rate: 0.02,
      diluted_shares: 100,
    });
    const high = await rimValuationTool.invoke({
      ticker: 'AAPL',
      cost_of_equity: 0.10,
      beginning_book_value: 1000,
      projected_net_income: [150],
      terminal_growth_rate: 0.02,
      diluted_shares: 100,
    });
    const lowParsed = parseResult(low);
    const highParsed = parseResult(high);
    expect(Number(highParsed.data.equityValuePerShare)).toBeGreaterThan(
      Number(lowParsed.data.equityValuePerShare),
    );
  });
});
