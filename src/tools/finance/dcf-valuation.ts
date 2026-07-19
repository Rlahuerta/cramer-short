import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { computeFairValuePerShare, type DcfInputs } from '../../utils/finance/dcf.js';
import { runDcfValidation } from '../../utils/finance/dcf-validation.js';
import { sensitivityGrid, formatSensitivityGrid } from '../../utils/finance/dcf-sensitivity.js';

export const DCF_VALUATION_DESCRIPTION = `
Runs a discounted cash flow (DCF) valuation using pre-computed inputs.

## What this tool does
1. Projects free cash flows (FCF) over the chosen horizon using a growth rate
   that decays linearly each year.
2. Computes terminal value via the Gordon-growth model.
3. If an exit_multiple is provided, also computes an exit-multiple TV and
   reports the divergence between the two methods.
4. Discounts projected FCFs and terminal value back to present value.
5. Runs validation checks (WACC > terminal growth, TV proportion of EV, etc.).
6. Generates a 2-way sensitivity grid (WACC × terminal growth).

## Inputs
- **ticker** (required): stock ticker symbol
- **base_fcf** (required): current-year free cash flow (currency units)
- **net_debt** (required): total debt minus cash and equivalents
- **diluted_shares** (required): fully diluted shares outstanding
- **wacc**: weighted average cost of capital as a decimal (default 0.10)
- **growth_rate**: near-term FCF growth rate as a decimal (default 0.15)
- **terminal_growth_rate**: long-run terminal growth rate as a decimal (default 0.025)
- **years**: projection horizon in years (default 5)
- **exit_multiple**: optional exit multiple for TV cross-check (e.g. 10)
- **decay**: annual decay applied to growth_rate (default 0.05)

## Output
- fairValuePerShare, enterpriseValue, equityValue
- tvMethod, tvProportionOfEv, divergencePct (if exit_multiple given)
- sensitivityGrid: formatted 2-way grid
- validation: { warnings, errors }
`.trim();

const DcfValuationSchema = z.object({
  ticker: z
    .string()
    .max(128)
    .describe("Stock ticker symbol, e.g. 'AAPL'."),
  base_fcf: z
    .number()
    .gt(0)
    .describe('Current-year free cash flow in currency units (must be > 0).'),
  net_debt: z
    .number()
    .describe('Total debt minus cash and equivalents (can be negative for net-cash companies).'),
  diluted_shares: z
    .number()
    .gt(0)
    .describe('Fully diluted shares outstanding (must be > 0).'),
  wacc: z
    .number()
    .min(0.03)
    .max(0.30)
    .default(0.10)
    .describe('Weighted average cost of capital as a decimal (default 0.10 = 10%).'),
  growth_rate: z
    .number()
    .min(-0.20)
    .max(2.00)
    .default(0.15)
    .describe('Near-term FCF growth rate as a decimal (default 0.15 = 15%).'),
  terminal_growth_rate: z
    .number()
    .min(0)
    .max(0.10)
    .default(0.025)
    .describe('Long-run terminal growth rate as a decimal (default 0.025 = 2.5%).'),
  years: z
    .number()
    .min(1)
    .max(20)
    .default(5)
    .describe('Projection horizon in years (default 5).'),
  exit_multiple: z
    .number()
    .min(3)
    .max(30)
    .optional()
    .describe('Optional exit multiple for TV cross-check (e.g. 10).'),
  decay: z
    .number()
    .min(0)
    .max(0.20)
    .default(0.05)
    .describe('Annual decay applied to growth_rate (default 0.05 = 5% per year).'),
});

/**
 * DCF valuation tool — pure computation, no external API calls.
 * The skill orchestrates fetching inputs via get_financials + wacc_inputs.
 */
export const dcfValuationTool = new DynamicStructuredTool({
  name: 'dcf_valuation',
  description: DCF_VALUATION_DESCRIPTION,
  schema: DcfValuationSchema,
  func: async (input) => {
    const ticker = input.ticker.trim().toUpperCase();

    const dcfInputs: DcfInputs = {
      baseFcf: input.base_fcf,
      wacc: input.wacc,
      growthRate: input.growth_rate,
      terminalGrowthRate: input.terminal_growth_rate,
      years: input.years,
      decay: input.decay,
      netDebt: input.net_debt,
      dilutedShares: input.diluted_shares,
      exitMultiple: input.exit_multiple,
    };

    let result: ReturnType<typeof computeFairValuePerShare>;
    try {
      result = computeFairValuePerShare(dcfInputs);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      return formatToolResult({
        ticker,
        fairValuePerShare: null,
        enterpriseValue: null,
        equityValue: null,
        netDebt: input.net_debt,
        wacc: input.wacc,
        growthRate: input.growth_rate,
        terminalGrowthRate: input.terminal_growth_rate,
        years: input.years,
        tvMethod: null,
        tvProportionOfEv: null,
        divergencePct: null,
        sensitivityGrid: null,
        validation: {
          warnings: [],
          errors: [message],
        },
      });
    }

    // 2. Run validation
    const validation = runDcfValidation({
      wacc: input.wacc,
      terminalGrowth: input.terminal_growth_rate,
      riskFreeRate: 0.043,
      isDevelopedMarket: true,
      tvPV: result.pvTerminalValue,
      enterpriseValue: result.enterpriseValue,
      roic: 0,
      computedEv: result.enterpriseValue,
      reportedEv: 0,
      debtToEquity: 0,
      source: 'unknown',
    });

    // 3. Sensitivity grid — default ranges per plan
    let formattedGrid: string[][] | null = null;
    try {
      const waccRange = { min: input.wacc - 0.01, max: input.wacc + 0.01, step: 0.005 };
      const growthRange = {
        min: input.terminal_growth_rate - 0.005,
        max: input.terminal_growth_rate + 0.005,
        step: 0.0025,
      };
      const grid = sensitivityGrid(dcfInputs, waccRange, growthRange);
      formattedGrid = formatSensitivityGrid(grid);
    } catch (e) {
      validation.warnings.push(`Sensitivity grid skipped: ${e instanceof Error ? e.message : String(e)}`);
    }

    return formatToolResult({
      ticker,
      fairValuePerShare: result.fairValuePerShare,
      enterpriseValue: result.enterpriseValue,
      equityValue: result.equityValue,
      netDebt: input.net_debt,
      wacc: input.wacc,
      growthRate: input.growth_rate,
      terminalGrowthRate: input.terminal_growth_rate,
      years: input.years,
      tvMethod: result.tvMethod,
      tvProportionOfEv: result.tvProportionOfEv,
      divergencePct: result.divergencePct ?? null,
      sensitivityGrid: formattedGrid,
      validation: {
        warnings: validation.warnings,
        errors: validation.errors,
      },
    });
  },
});
