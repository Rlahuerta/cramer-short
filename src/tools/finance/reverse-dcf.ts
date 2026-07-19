import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import {
  computeFairValuePerShare,
  projectFcf,
  gordonGrowthTv,
} from '../../utils/finance/dcf.js';

const ReverseDcfSchema = z.object({
  ticker: z
    .string()
    .max(128)
    .describe("Stock ticker symbol, e.g. 'AAPL'."),
  current_price: z
    .number()
    .describe('Current market price per share (must be > 0 for a valid solution).'),
  wacc: z
    .number()
    .min(0.03)
    .max(0.30)
    .default(0.10)
    .describe('Weighted average cost of capital as a decimal (default 0.10 = 10%).'),
  terminal_growth_rate: z
    .number()
    .min(0)
    .max(0.10)
    .default(0.025)
    .describe('Perpetual terminal growth rate as a decimal (default 0.025 = 2.5%).'),
  years: z
    .number()
    .min(1)
    .max(20)
    .default(5)
    .describe('Projection horizon in years (default 5).'),
  decay: z
    .number()
    .min(0)
    .max(0.20)
    .default(0.05)
    .describe('Annual decay applied to growth rate (default 0.05 = 5%).'),
  base_fcf: z.number().describe('Base-year free cash flow (required).'),
  net_debt: z
    .number()
    .describe('Net debt (can be negative for net-cash companies).'),
  diluted_shares: z
    .number()
    .min(0.01)
    .describe('Fully diluted shares outstanding (must be > 0).'),
});

export interface ReverseDcfResult {
  ticker: string;
  currentPrice: number;
  impliedGrowthRate: number | null;
  impliedFcfProjection: number[];
  impliedTv: number | null;
  resultStatus: 'converged' | 'out_of_range' | 'no_solution';
}

export const REVERSE_DCF_DESCRIPTION = `
Solves for the market-implied FCF growth rate that makes the DCF fair value equal to the current stock price.

## What this tool does
1. Takes the current stock price and a set of DCF assumptions (WACC, terminal growth, projection horizon, base FCF, net debt, shares).
2. Uses bisection search over growth_rate ∈ [-0.20, 2.00] to find the growth rate that makes computeFairValuePerShare ≈ current_price.
3. Returns the implied growth rate, the implied FCF projection, the implied terminal value, and a convergence status.

## When to use
- In the DCF skill Step 6 (Reverse DCF) to surface market-implied assumptions.
- Any time you want to understand what growth the market is pricing in.

## Inputs
- ticker, current_price, wacc, terminal_growth_rate, years, decay, base_fcf, net_debt, diluted_shares

## Output
- impliedGrowthRate: number | null
- impliedFcfProjection: number[]
- impliedTv: number | null
- resultStatus: 'converged' | 'out_of_range' | 'no_solution'
`.trim();

function solveReverseDcf(
  ticker: string,
  currentPrice: number,
  baseFcf: number,
  wacc: number,
  terminalGrowthRate: number,
  years: number,
  decay: number,
  netDebt: number,
  dilutedShares: number,
): ReverseDcfResult {
  // No finite growth rate can produce a non-positive fair value when base FCF is positive.
  if (!isFinite(currentPrice) || currentPrice <= 0) {
    return {
      ticker,
      currentPrice,
      impliedGrowthRate: null,
      impliedFcfProjection: [],
      impliedTv: null,
      resultStatus: 'no_solution',
    };
  }

  const LOW = -0.20;
  const HIGH = 2.00;
  const TOLERANCE = 0.01;
  const MAX_ITER = 100;

  function fairValue(g: number): number {
    return computeFairValuePerShare({
      baseFcf,
      wacc,
      growthRate: g,
      terminalGrowthRate,
      years,
      decay,
      netDebt,
      dilutedShares,
    }).fairValuePerShare;
  }

  let fLow = fairValue(LOW) - currentPrice;
  let fHigh = fairValue(HIGH) - currentPrice;

  // Exact hits at boundaries
  if (Math.abs(fLow) <= TOLERANCE) {
    const proj = projectFcf(baseFcf, LOW, years, decay);
    const finalFcf = proj[proj.length - 1];
    return {
      ticker,
      currentPrice,
      impliedGrowthRate: LOW,
      impliedFcfProjection: proj,
      impliedTv: gordonGrowthTv(finalFcf, terminalGrowthRate, wacc),
      resultStatus: 'converged',
    };
  }

  if (Math.abs(fHigh) <= TOLERANCE) {
    const proj = projectFcf(baseFcf, HIGH, years, decay);
    const finalFcf = proj[proj.length - 1];
    return {
      ticker,
      currentPrice,
      impliedGrowthRate: HIGH,
      impliedFcfProjection: proj,
      impliedTv: gordonGrowthTv(finalFcf, terminalGrowthRate, wacc),
      resultStatus: 'converged',
    };
  }

  // Monotonic increasing function: at most one root.
  // If both endpoints are on the same side of zero, no root in [LOW, HIGH].
  if (fLow > 0 && fHigh > 0) {
    return {
      ticker,
      currentPrice,
      impliedGrowthRate: null,
      impliedFcfProjection: [],
      impliedTv: null,
      resultStatus: 'out_of_range',
    };
  }

  if (fLow < 0 && fHigh < 0) {
    return {
      ticker,
      currentPrice,
      impliedGrowthRate: null,
      impliedFcfProjection: [],
      impliedTv: null,
      resultStatus: 'out_of_range',
    };
  }

  // Bracket exists: fLow < 0 and fHigh > 0
  let a = LOW;
  let b = HIGH;
  let fA = fLow;
  let mid = 0;
  let fMid = 0;

  for (let i = 0; i < MAX_ITER; i++) {
    mid = (a + b) / 2;
    fMid = fairValue(mid) - currentPrice;

    if (Math.abs(fMid) <= TOLERANCE) {
      break;
    }

    if (fA * fMid <= 0) {
      b = mid;
    } else {
      a = mid;
      fA = fMid;
    }
  }

  const impliedGrowthRate = mid;
  const impliedFcfProjection = projectFcf(baseFcf, impliedGrowthRate, years, decay);
  const finalFcf = impliedFcfProjection[years - 1];
  const impliedTv = gordonGrowthTv(finalFcf, terminalGrowthRate, wacc);

  return {
    ticker,
    currentPrice,
    impliedGrowthRate,
    impliedFcfProjection,
    impliedTv,
    resultStatus: 'converged',
  };
}

export const reverseDcfTool = new DynamicStructuredTool({
  name: 'reverse_dcf',
  description: REVERSE_DCF_DESCRIPTION,
  schema: ReverseDcfSchema,
  func: async (input) => {
    const ticker = input.ticker.trim().toUpperCase();
    const result = solveReverseDcf(
      ticker,
      input.current_price,
      input.base_fcf,
      input.wacc,
      input.terminal_growth_rate,
      input.years,
      input.decay,
      input.net_debt,
      input.diluted_shares,
    );
    return formatToolResult(result);
  },
});
