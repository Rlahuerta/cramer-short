/**
 * Pure Residual Income Model (RIM) / Abnormal Earnings utilities.
 *
 * All functions are side-effect free and accept plain numbers so they can be
 * unit-tested without any API or file-system dependencies.
 *
 * Formulas used (Francis, Olsson, & Oswald 2000; Penman 1998):
 *   AE_t = NI_t − r_e × BV_{t−1}
 *   TV   = AE_final × (1 + g) / (r_e − g)
 *   V_0  = BV_0 + Σ PV(AE_t) + PV(TV)
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface RimInputs {
  /** Book value of equity at the start of the forecast (t=0). */
  beginningBookValue: number;
  /** Array of projected net income per year (length must equal years). */
  projectedNetIncome: number[];
  /** Cost of equity as a decimal (e.g. 0.10 for 10%). */
  costOfEquity: number;
  /** Terminal growth rate of abnormal earnings as a decimal (e.g. 0.02). */
  terminalGrowthRate: number;
  /** Number of explicit forecast years. */
  years: number;
  /** Fully diluted shares outstanding. */
  dilutedShares: number;
}

export interface RimResult {
  /** Equity value per diluted share. */
  equityValuePerShare: number;
  /** Total equity value (V_0). */
  equityValue: number;
  /** Abnormal earnings for each explicit forecast year. */
  projectedAbnormalEarnings: number[];
  /** Terminal value of abnormal earnings at the end of the horizon. */
  rimTerminalValue: number;
  /** Present value of the terminal value. */
  pvRimTerminalValue: number;
}

// ─── Core functions ────────────────────────────────────────────────────────────

/**
 * Compute abnormal earnings for a single period.
 *   AE = NI − r_e × BV_{t−1}
 */
export function computeAbnormalEarnings(
  netIncome: number,
  costOfEquity: number,
  beginningBookValue: number,
): number {
  return netIncome - costOfEquity * beginningBookValue;
}

/**
 * Gordon-growth terminal value for abnormal earnings.
 *   TV = AE × (1 + g) / (r_e − g)
 *
 * Throws when r_e ≤ g because the perpetuity would diverge.
 */
export function rimTerminalValue(
  finalAbnormalEarnings: number,
  terminalGrowth: number,
  costOfEquity: number,
): number {
  if (costOfEquity <= terminalGrowth) {
    throw new Error(
      `Cost of equity (${costOfEquity}) must exceed terminal growth (${terminalGrowth})`,
    );
  }
  return (finalAbnormalEarnings * (1 + terminalGrowth)) / (costOfEquity - terminalGrowth);
}

/**
 * Full RIM equity valuation.
 *
 * Book value is assumed to evolve via clean surplus:
 *   BV_t = BV_{t−1} + NI_t − div_t
 *
 * For simplicity, we retain all earnings (dividends = 0) inside the forecast
 * horizon, so BV_t = BV_{t−1} + NI_t.  This is the standard academic
 * simplification when dividends are not provided separately.
 *
 * Discounting uses costOfEquity (NOT WACC) because RIM is an equity model.
 */
export function computeRimEquityValue(inputs: RimInputs): RimResult {
  const {
    beginningBookValue,
    projectedNetIncome,
    costOfEquity,
    terminalGrowthRate,
    years,
    dilutedShares,
  } = inputs;

  let bookValue = beginningBookValue;
  const projectedAbnormalEarnings: number[] = [];
  let pvAbnormalEarnings = 0;

  for (let t = 0; t < years; t++) {
    const ni = projectedNetIncome[t] ?? 0;
    const ae = computeAbnormalEarnings(ni, costOfEquity, bookValue);
    projectedAbnormalEarnings.push(ae);
    pvAbnormalEarnings += ae / Math.pow(1 + costOfEquity, t + 1);

    // Clean-surplus evolution: retain all earnings for next period's BV
    bookValue += ni;
  }

  const finalAe = projectedAbnormalEarnings[years - 1] ?? 0;
  const rimTerminalValue_ = rimTerminalValue(finalAe, terminalGrowthRate, costOfEquity);
  const pvRimTerminalValue = rimTerminalValue_ / Math.pow(1 + costOfEquity, years);

  const equityValue = beginningBookValue + pvAbnormalEarnings + pvRimTerminalValue;
  const equityValuePerShare = equityValue / dilutedShares;

  return {
    equityValue,
    equityValuePerShare,
    projectedAbnormalEarnings,
    rimTerminalValue: rimTerminalValue_,
    pvRimTerminalValue,
  };
}

/**
 * Validate the clean-surplus relation for a set of historical periods.
 *
 * Returns true iff ΔBV_t = NI_t − div_t for every period.
 * Returns false if arrays have mismatched lengths or any period violates the identity.
 */
export function validateCleanSurplus(
  bookValueChanges: number[],
  netIncome: number[],
  dividends: number[],
): boolean {
  if (
    bookValueChanges.length !== netIncome.length ||
    netIncome.length !== dividends.length
  ) {
    return false;
  }

  for (let i = 0; i < bookValueChanges.length; i++) {
    if (bookValueChanges[i] !== netIncome[i] - dividends[i]) {
      return false;
    }
  }

  return true;
}
