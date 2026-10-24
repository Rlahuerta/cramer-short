/**
 * DCF validation guards -- pure functions.
 *
 * All functions are side-effect free and return structured { ok, message }
 * (never throw).
 */

// -- Types -------------------------------------------------------------------

export interface DcfValidationInput {
  wacc: number;
  terminalGrowth: number;
  riskFreeRate: number;
  isDevelopedMarket?: boolean;
  tvPV: number;
  enterpriseValue: number;
  roic: number;
  computedEv: number;
  reportedEv: number;
  debtToEquity: number;
  source?: 'book' | 'market' | 'unknown';
}

export interface DcfValidationResult {
  ok: boolean;
  warnings: string[];
  errors: string[];
}

// -- Individual validators ---------------------------------------------------

/**
 * ok if wacc > terminalGrowth + 0.005 (0.5% safety margin).
 */
export function validateWaccGreaterThanGrowth(
  wacc: number,
  terminalGrowth: number,
): { ok: boolean; message: string } {
  if (wacc > terminalGrowth + 0.005) {
    return { ok: true, message: 'WACC exceeds terminal growth by adequate margin' };
  }
  return {
    ok: false,
    message: `WACC (${(wacc * 100).toFixed(2)}%) must be at least 0.5% above terminal growth (${(terminalGrowth * 100).toFixed(2)}%)`,
  };
}

/**
 * ok if terminalGrowth <= riskFreeRate.
 */
export function validateTerminalGrowthVsRfr(
  terminalGrowth: number,
  riskFreeRate: number,
): { ok: boolean; message: string } {
  if (terminalGrowth <= riskFreeRate) {
    return { ok: true, message: 'Terminal growth does not exceed risk-free rate' };
  }
  return {
    ok: false,
    message: `Terminal growth (${(terminalGrowth * 100).toFixed(2)}%) exceeds risk-free rate (${(riskFreeRate * 100).toFixed(2)}%)`,
  };
}

/**
 * ok if terminalGrowth <= 3.5% (developed) or <= 4.0% (EM).
 */
export function validateTerminalGrowthCap(
  terminalGrowth: number,
  isDevelopedMarket: boolean = true,
): { ok: boolean; message: string } {
  const cap = isDevelopedMarket ? 0.035 : 0.04;
  const capLabel = isDevelopedMarket ? '3.5%' : '4.0%';
  if (terminalGrowth <= cap) {
    return {
      ok: true,
      message: `Terminal growth is within ${isDevelopedMarket ? 'developed-market' : 'emerging-market'} cap (${capLabel})`,
    };
  }
  return {
    ok: false,
    message: `Terminal growth (${(terminalGrowth * 100).toFixed(2)}%) exceeds ${capLabel} ${isDevelopedMarket ? 'developed-market' : 'emerging-market'} cap`,
  };
}

/**
 * Flag if TV/EV > 0.85 (warn) or > 0.90 (fail).
 */
export function validateTvProportion(
  tvPV: number,
  enterpriseValue: number,
): { ok: boolean; message: string; ratio: number } {
  if (enterpriseValue <= 0) {
    return {
      ok: false,
      message: 'Enterprise value must be positive to compute TV proportion',
      ratio: 0,
    };
  }
  const ratio = tvPV / enterpriseValue;
  if (ratio > 0.90) {
    return {
      ok: false,
      message: `Terminal value (${(ratio * 100).toFixed(1)}%) exceeds 90% of enterprise value -- reduce growth assumption`,
      ratio,
    };
  }
  if (ratio > 0.85) {
    return {
      ok: true,
      message: `Terminal value (${(ratio * 100).toFixed(1)}%) is above 85% of enterprise value -- consider lowering growth`,
      ratio,
    };
  }
  return { ok: true, message: '', ratio };
}

/**
 * Warn if WACC > ROIC (value destruction).
 */
export function validateWaccVsRoic(
  wacc: number,
  roic: number,
): { ok: boolean; message: string } {
  if (wacc > roic) {
    return {
      ok: true,
      message: `WACC (${(wacc * 100).toFixed(2)}%) exceeds ROIC (${(roic * 100).toFixed(2)}%) -- potential value destruction`,
    };
  }
  return {
    ok: true,
    message: `WACC (${(wacc * 100).toFixed(2)}%) is at or below ROIC (${(roic * 100).toFixed(2)}%) -- value-creating`,
  };
}

/**
 * ok if |computedEv - reportedEv| / reportedEv <= 0.30.
 */
export function validateEvSanity(
  computedEv: number,
  reportedEv: number,
): { ok: boolean; message: string; deviationPct: number } {
  if (reportedEv <= 0) {
    return {
      ok: false,
      message: 'Reported enterprise value must be positive',
      deviationPct: 0,
    };
  }
  const deviationPct = (Math.abs(computedEv - reportedEv) / reportedEv) * 100;
  if (deviationPct <= 30) {
    return {
      ok: true,
      message: `EV deviation ${deviationPct.toFixed(2)}% is within 30% sanity threshold`,
      deviationPct,
    };
  }
  return {
    ok: false,
    message: `EV deviation ${deviationPct.toFixed(2)}% exceeds 30% -- revisit WACC or growth assumptions`,
    deviationPct,
  };
}

/**
 * Warn if source is 'book' (WACC is forward-looking).
 */
export function validateMarketValueWeights(
  debtToEquity: number,
  source: 'book' | 'market' | 'unknown' = 'unknown',
): { ok: boolean; message: string } {
  if (source === 'book') {
    return {
      ok: true,
      message: `Using book-value weights (source=${source}) -- WACC is forward-looking; consider market-value weights`,
    };
  }
  return {
    ok: true,
    message: `Capital-structure weights sourced from ${source} values`,
  };
}

// -- Aggregator --------------------------------------------------------------

/**
 * Runs all 7 validators and aggregates errors + warnings.
 */
export function runDcfValidation(checks: DcfValidationInput): DcfValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const r1 = validateWaccGreaterThanGrowth(checks.wacc, checks.terminalGrowth);
  if (!r1.ok) errors.push(r1.message);

  const r2 = validateTerminalGrowthVsRfr(checks.terminalGrowth, checks.riskFreeRate);
  if (!r2.ok) errors.push(r2.message);

  const r3 = validateTerminalGrowthCap(
    checks.terminalGrowth,
    checks.isDevelopedMarket ?? true,
  );
  if (!r3.ok) errors.push(r3.message);

  const r4 = validateTvProportion(checks.tvPV, checks.enterpriseValue);
  if (!r4.ok) {
    errors.push(r4.message);
  } else if (r4.message) {
    warnings.push(r4.message);
  }

  const r5 = validateWaccVsRoic(checks.wacc, checks.roic);
  if (r5.ok && r5.message.toLowerCase().includes('destruction')) {
    warnings.push(r5.message);
  }

  const r6 = validateEvSanity(checks.computedEv, checks.reportedEv);
  if (!r6.ok) {
    errors.push(r6.message);
  }

  const r7 = validateMarketValueWeights(checks.debtToEquity, checks.source);
  if (r7.ok && r7.message.toLowerCase().includes('book')) {
    warnings.push(r7.message);
  }

  return { ok: errors.length === 0, warnings, errors };
}
