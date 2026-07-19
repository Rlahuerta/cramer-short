// ponytail: fcfe mode TBD when skill needs it

// ─── Types ───────────────────────────────────────────────────────────────────

export interface DcfInputs {
  /** Base year free cash flow (e.g. latest annual FCF). */
  baseFcf: number;
  /** Discount rate as a decimal (e.g. 0.10 for 10%). */
  wacc: number;
  /** Near-term FCF growth rate as a decimal (e.g. 0.10 for 10%). */
  growthRate: number;
  /** Long-run terminal growth rate as a decimal (e.g. 0.025 for 2.5%). */
  terminalGrowthRate: number;
  /** Projection horizon in years (e.g. 5). */
  years: number;
  /** Annual decay applied to growthRate (default 0.05 → 5% per year). */
  decay: number;
  /** Net debt (can be negative for net-cash companies). */
  netDebt: number;
  /** Fully diluted shares outstanding. */
  dilutedShares: number;
  /** Optional exit multiple for cross-check TV. */
  exitMultiple?: number;
}

export interface DcfResult {
  fairValuePerShare: number;
  enterpriseValue: number;
  equityValue: number;
  projectedFcf: number[];
  terminalValue: number;
  pvTerminalValue: number;
  tvMethod: 'gordon' | 'exit-multiple' | 'both';
  tvProportionOfEv: number;
  exitMultipleTv?: number;
  divergencePct?: number;
}

// ─── Core functions ──────────────────────────────────────────────────────────

/**
 * Project FCF for years 1..N with decayed growth rate.
 * Year 1 gets full growthRate; each subsequent year the growth rate is
 * reduced by `decay` (e.g. decay=0.05 → 95%, 90%, 85%, 80% …).
 * Reflects competitive dynamics per SKILL.md §4.
 */
export function projectFcf(
  baseFcf: number,
  growthRate: number,
  years: number,
  decay: number = 0.05,
): number[] {
  const result: number[] = [];
  let current = baseFcf;
  for (let i = 0; i < years; i++) {
    const factor = Math.max(0, 1.0 - decay * i);
    current = current * (1 + growthRate * factor);
    result.push(current);
  }
  return result;
}

/**
 * Terminal value via the Gordon Growth Model.
 *   TV = finalFcf × (1 + g) / (wacc − g)
 * Throws when wacc ≤ terminalGrowth (division by zero or negative denominator).
 */
export function gordonGrowthTv(
  finalFcf: number,
  terminalGrowth: number,
  wacc: number,
): number {
  if (wacc <= terminalGrowth) {
    throw new Error('WACC must be greater than terminal growth rate');
  }
  return (finalFcf * (1 + terminalGrowth)) / (wacc - terminalGrowth);
}

/**
 * Terminal value via exit multiple on final-year EBITDA.
 *   TV = finalEbitda × multiple
 */
export function exitMultipleTv(
  finalEbitda: number,
  exitMultiple: number,
): number {
  return finalEbitda * exitMultiple;
}

/**
 * Discount a stream of cash flows back to present value.
 *   PV_i = cashFlows[i] / (1 + wacc)^(i+1)
 */
export function discountPV(
  cashFlows: number[],
  wacc: number,
): number[] {
  return cashFlows.map((cf, i) => cf / Math.pow(1 + wacc, i + 1));
}

/**
 * Compute net debt honoring lease / pension / convertible / preferred
 * edge cases per SKILL.md §5.
 */
export function computeNetDebt(parts: {
  totalDebt: number;
  operatingLeaseLiabilities: number;
  cash: number;
  shortTermInvestments: number;
  restrictedCash?: number;
  pensionDeficit?: number;
  preferredStock?: number;
  convertiblesFace?: number;
}): number {
  return (
    parts.totalDebt +
    parts.operatingLeaseLiabilities -
    parts.cash -
    parts.shortTermInvestments -
    (parts.restrictedCash ?? 0) +
    (parts.pensionDeficit ?? 0) +
    (parts.preferredStock ?? 0) +
    (parts.convertiblesFace ?? 0)
  );
}

/**
 * Full DCF fair-value-per-share calculation.
 * Always computes Gordon-growth TV; if exitMultiple is provided
 * also computes exit-multiple TV and reports divergence.
 */
export function computeFairValuePerShare(inputs: DcfInputs): DcfResult {
  const {
    baseFcf,
    wacc,
    growthRate,
    terminalGrowthRate,
    years,
    decay,
    netDebt,
    dilutedShares,
    exitMultiple,
  } = inputs;

  // 1. Project FCFs
  const projectedFcf = projectFcf(baseFcf, growthRate, years, decay);
  const finalFcf = projectedFcf[years - 1];

  // 2. Terminal values
  const gordonTv = gordonGrowthTv(finalFcf, terminalGrowthRate, wacc);
  let exitTv: number | undefined;
  let tvMethod: DcfResult['tvMethod'] = 'gordon';

  if (exitMultiple !== undefined) {
    exitTv = exitMultipleTv(finalFcf, exitMultiple);
    tvMethod = 'both';
  }

  // 3. Discounting
  const pvFcf = discountPV(projectedFcf, wacc);
  const sumPvFcf = pvFcf.reduce((a, b) => a + b, 0);

  const primaryTv = gordonTv;
  const pvTerminalValue = primaryTv / Math.pow(1 + wacc, years);

  const enterpriseValue = sumPvFcf + pvTerminalValue;
  const equityValue = enterpriseValue - netDebt;
  const fairValuePerShare = equityValue / dilutedShares;

  const tvProportionOfEv =
    enterpriseValue > 0 ? pvTerminalValue / enterpriseValue : 0;

  const result: DcfResult = {
    fairValuePerShare,
    enterpriseValue,
    equityValue,
    projectedFcf,
    terminalValue: primaryTv,
    pvTerminalValue,
    tvMethod,
    tvProportionOfEv,
  };

  if (exitTv !== undefined) {
    result.exitMultipleTv = exitTv;
    const avg = (gordonTv + exitTv) / 2;
    result.divergencePct =
      avg > 0 ? (Math.abs(gordonTv - exitTv) / avg) * 100 : 0;
  }

  return result;
}
