import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import {
  computeRimEquityValue,
  rimTerminalValue,
  validateCleanSurplus,
  type RimInputs,
} from '../../utils/finance/rim.js';

export const RIM_VALUATION_DESCRIPTION = `
Performs a Residual Income Model (RIM) / Abnormal Earnings valuation for a given ticker.

## What this tool does
1. Computes abnormal earnings for each explicit forecast year:
   AE_t = NI_t − r_e × BV_{t−1}
2. Computes a Gordon-growth terminal value for abnormal earnings at the horizon.
3. Discounts both the abnormal earnings stream and the terminal value using cost of equity.
4. Optionally validates the clean-surplus relation if dividends and book-value changes are provided.

## When to use
- In the DCF skill Step 5 (RIM cross-check) to compare RIM fair value to DCF fair value.
- Any time the user asks for an equity valuation via residual income or abnormal earnings.

## Inputs
- **ticker** (required): stock ticker symbol
- **cost_of_equity**: cost of equity as a decimal (default 0.10 = 10%)
- **beginning_book_value**: book value of equity at the start of the forecast (t=0)
- **projected_net_income**: array of projected net income per year (default [100, 110, 120, 130, 140])
- **terminal_growth_rate**: terminal growth rate of abnormal earnings (default 0.025 = 2.5%)
- **diluted_shares**: fully diluted shares outstanding
- **dividends**: optional array of dividends per year (for clean-surplus validation)
- **book_value_changes**: optional array of year-over-year book value changes (for clean-surplus validation)

## Output
- ticker, equityValuePerShare, beginningBookValuePerShare
- projectedAbnormalEarnings[], rimTerminalValue, pvRimTerminalValue
- cleanSurplusOk (boolean | null)
- validation { warnings, errors }
`.trim();

const RimValuationSchema = z.object({
  ticker: z
    .string()
    .max(128)
    .describe("Stock ticker symbol, e.g. 'AAPL'."),
  cost_of_equity: z
    .number()
    .min(0.03)
    .max(0.30)
    .default(0.10)
    .describe('Cost of equity as a decimal (default 0.10 = 10%).'),
  beginning_book_value: z
    .number()
    .describe('Book value of equity at the start of the forecast (t=0).'),
  projected_net_income: z
    .array(z.number())
    .min(1)
    .max(20)
    .default([100, 110, 120, 130, 140])
    .describe('Projected net income per year (1-20 periods).'),
  terminal_growth_rate: z
    .number()
    .min(0)
    .max(0.10)
    .default(0.025)
    .describe('Terminal growth rate of abnormal earnings as a decimal (default 0.025 = 2.5%).'),
  diluted_shares: z
    .number()
    .positive()
    .describe('Fully diluted shares outstanding.'),
  dividends: z
    .array(z.number())
    .optional()
    .describe('Optional dividends per year for clean-surplus validation.'),
  book_value_changes: z
    .array(z.number())
    .optional()
    .describe('Optional year-over-year book value changes for clean-surplus validation.'),
});

export const rimValuationTool = new DynamicStructuredTool({
  name: 'rim_valuation',
  description: RIM_VALUATION_DESCRIPTION,
  schema: RimValuationSchema,
  func: async (input) => {
    const ticker = input.ticker.trim().toUpperCase();
    const warnings: string[] = [];
    const errors: string[] = [];

    // Clean-surplus validation (only when both optional arrays are provided)
    let cleanSurplusOk: boolean | null = null;
    if (input.dividends && input.book_value_changes) {
      cleanSurplusOk = validateCleanSurplus(
        input.book_value_changes,
        input.projected_net_income,
        input.dividends,
      );
      if (!cleanSurplusOk) {
        warnings.push(
          'Clean surplus relation violated: book value changes do not match net income minus dividends.',
        );
      }
    }

    // Build RimInputs from schema
    const years = input.projected_net_income.length;
    const rimInputs: RimInputs = {
      beginningBookValue: input.beginning_book_value,
      projectedNetIncome: input.projected_net_income,
      costOfEquity: input.cost_of_equity,
      terminalGrowthRate: input.terminal_growth_rate,
      years,
      dilutedShares: input.diluted_shares,
    };

    let result: {
      equityValuePerShare: number | null;
      beginningBookValuePerShare: number;
      projectedAbnormalEarnings: number[];
      rimTerminalValue: number | null;
      pvRimTerminalValue: number | null;
      equityValue: number | null;
    } = {
      equityValuePerShare: null,
      beginningBookValuePerShare: input.beginning_book_value / input.diluted_shares,
      projectedAbnormalEarnings: [],
      rimTerminalValue: null,
      pvRimTerminalValue: null,
      equityValue: null,
    };

    try {
      const computed = computeRimEquityValue(rimInputs);
      result = {
        equityValuePerShare: computed.equityValuePerShare,
        beginningBookValuePerShare: input.beginning_book_value / input.diluted_shares,
        projectedAbnormalEarnings: computed.projectedAbnormalEarnings,
        rimTerminalValue: computed.rimTerminalValue,
        pvRimTerminalValue: computed.pvRimTerminalValue,
        equityValue: computed.equityValue,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // Convert terminal-value error (r_e <= g) into a structured error
      if (message.toLowerCase().includes('terminal growth') || message.toLowerCase().includes('must exceed')) {
        errors.push(message);
      } else {
        errors.push(`RIM computation failed: ${message}`);
      }
    }

    return formatToolResult(
      {
        ticker,
        equityValuePerShare: result.equityValuePerShare,
        beginningBookValuePerShare: result.beginningBookValuePerShare,
        projectedAbnormalEarnings: result.projectedAbnormalEarnings,
        rimTerminalValue: result.rimTerminalValue,
        pvRimTerminalValue: result.pvRimTerminalValue,
        cleanSurplusOk,
        validation: { warnings, errors },
      },
      [],
    );
  },
});
