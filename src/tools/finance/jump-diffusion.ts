/**
 * Jump-Diffusion helpers (Idea 2 — Polymarket-informed Merton jumps).
 *
 * Mirrors `research/models/jump_diffusion.py`. Pure math; no side effects.
 *
 * The Merton (1976) jump-diffusion model adds a compound Poisson term to GBM:
 *
 *   dS_t / S_t = (μ − λ·κ) dt + σ dW_t + (J − 1) dN_t
 *
 * where:
 *   - N_t is a Poisson process with intensity λ (jumps per unit time);
 *   - log(J) ~ N(μ_J, σ_J²) — the jump multiplier is log-normal;
 *   - κ = E[J − 1] = exp(μ_J + σ_J²/2) − 1 is the expected percentage jump;
 *   - λ·κ is the **drift compensator** that keeps E[dS/S] = μ dt.
 *
 * Without the compensator the simulated drift drifts upward (or down) by
 * λ·κ per unit time, double-counting the jump's expected impact.
 *
 * In daily-discretised form, with N independent jump events e (each having
 * its own intensity λ_e, log-mean μ_J,e, log-vol σ_J,e):
 *
 *   r_t = (μ_t − Σ_e λ_e · κ_e) Δt + σ_t √Δt · Z + Σ_e Bern(λ_e Δt) · N(μ_J,e, σ_J,e²)
 */

import { transformQToP } from './rnd-integration.js';
import type { JumpDirection } from './polymarket.js';

export type { JumpDirection };

/**
 * P2a — Apply directional sign-flip to a prior log-jump mean.
 *
 * Asset-class priors in {@link JUMP_DEFAULTS} default to **negative**
 * means because Polymarket tail-risk markets cluster on the downside.
 * When a market explicitly implies an *up* catalyst (rate cut, deal,
 * approval, breakthrough), flip the prior's magnitude positive.
 * `unknown` and `undefined` preserve the prior unchanged.
 */
export function effectiveJumpMean(
  priorMean: number,
  direction: JumpDirection | undefined,
): number {
  if (direction === 'up') return Math.abs(priorMean);
  if (direction === 'down') return -Math.abs(priorMean);
  return priorMean;
}

/**
 * Per-asset-class jump-magnitude defaults.
 *
 * Calibration source: rolling 90-day max-abs-daily-log-return percentiles
 * across SPY/QQQ (etf, equity), BTC/ETH (crypto), GLD/USO (commodity)
 * over 2020-01-01..2024-12-31 (n ≈ 1,250 daily obs per series).
 *
 * Magnitudes are intentionally conservative — they describe the *typical*
 * tail event, not a once-in-a-decade crash. Override at the call site if
 * the Polymarket question carries a more specific implied magnitude
 * (e.g., a "Will SPY close < 4,000?" market implies a known down-jump).
 *
 * `meanLogJump` defaults are negative because Polymarket tail-risk
 * markets (war, default, recession, hack) cluster on the downside.
 */
export interface JumpPrior {
  /** Mean of log(J) — negative ⇒ down-jump (selloff) */
  meanLogJump: number;
  /** Std of log(J) — wider ⇒ more uncertainty about the size */
  stdLogJump: number;
}

export const JUMP_DEFAULTS: Record<'etf' | 'equity' | 'crypto' | 'commodity' | 'geopolitics', JumpPrior> = {
  etf:         { meanLogJump: -0.04, stdLogJump: 0.02 },
  equity:      { meanLogJump: -0.05, stdLogJump: 0.03 },
  crypto:      { meanLogJump: -0.08, stdLogJump: 0.05 },
  commodity:   { meanLogJump: -0.05, stdLogJump: 0.03 },
  /** War, sanctions, political shock — spec: ±10% expected impact, wide uncertainty. */
  geopolitics: { meanLogJump: -0.10, stdLogJump: 0.06 },
};

/**
 * Specification for a single jump-event source.
 *
 * `dailyIntensity` is λ_e in the Merton SDE; it is the *physical-measure*
 * Poisson rate per *daily* step, already converted from the Q-measure
 * Polymarket probability via {@link polymarketProbToHazard} +
 * {@link transformQToP}. The compensator μ_t − λ·κ uses this value.
 */
export interface JumpEventSpec {
  /** Identifier for provenance (e.g., the Polymarket market slug). */
  id: string;
  /** Physical-measure daily Poisson intensity (jumps/day). 0 ≤ λ ≤ 0.95. */
  dailyIntensity: number;
  /** Mean of log(J) for this event. */
  meanLogJump: number;
  /** Std of log(J) for this event. */
  stdLogJump: number;
  /** P2a — direction implied by the source question, for provenance. */
  jumpDirection?: JumpDirection;
}

/**
 * Convert a Polymarket-implied total settlement probability to a daily
 * Poisson hazard rate using the survival-function relation:
 *
 *   1 − p = exp(−λ_total) ⇒ λ_total = −ln(1 − p)
 *
 * Then split λ_total uniformly across the `horizonDays` settlement window
 * to obtain the per-day intensity:
 *
 *   λ_daily = λ_total / horizonDays
 *
 * This is the **correct** continuous-time hazard, not the linear
 * approximation `p / horizonDays` (which only matches for p ≪ 1 and
 * over-estimates λ for large p).
 *
 * @param p Probability that the event occurs *at all* over the horizon
 *          (0 < p < 1). Must already be in physical measure (apply
 *          {@link transformQToP} first).
 * @param horizonDays Days until settlement (≥ 1).
 * @returns Per-day jump intensity, capped at 0.95 to keep
 *          Bern(λ_daily) ≪ 1 per step.
 */
export function polymarketProbToHazard(p: number, horizonDays: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 0.95; // saturate at the per-day cap
  const days = Math.max(1, horizonDays);
  const lambdaTotal = -Math.log(1 - p);
  const lambdaDaily = lambdaTotal / days;
  return Math.min(0.95, Math.max(0, lambdaDaily));
}

/**
 * One-shot helper: convert raw Polymarket Q-prob into a fully-specified
 * `JumpEventSpec` using physical-measure conversion and asset-class priors.
 *
 * @param raw                  Raw Polymarket YES price (Q-measure).
 * @param horizonDays          Days to settlement.
 * @param historicalDriftAnnual Annualised historical drift for Q→P.
 * @param riskFreeRate          Annualised risk-free rate for Q→P.
 * @param volatilityAnnual      Annualised vol for Q→P.
 * @param prior                Per-asset jump-size prior.
 * @param id                   Free-form identifier.
 */
export function buildJumpEventSpec(
  raw: number,
  horizonDays: number,
  historicalDriftAnnual: number,
  riskFreeRate: number,
  volatilityAnnual: number,
  prior: JumpPrior,
  id: string,
  jumpDirection: JumpDirection = 'unknown',
): JumpEventSpec {
  const pPhysical = transformQToP(raw, historicalDriftAnnual, riskFreeRate, volatilityAnnual, horizonDays);
  const dailyIntensity = polymarketProbToHazard(pPhysical, horizonDays);
  return {
    id,
    dailyIntensity,
    meanLogJump: effectiveJumpMean(prior.meanLogJump, jumpDirection),
    stdLogJump: prior.stdLogJump,
    jumpDirection,
  };
}

/**
 * Daily Merton drift compensator:
 *
 *   c_t = Σ_e λ_e · (exp(μ_J,e + σ_J,e² / 2) − 1)
 *
 * Subtract from `μ_t · Δt` (in log-space, Δt = 1 day) to keep the
 * post-jump expected return equal to `μ_t`.
 */
export function jumpDriftCompensator(events: readonly JumpEventSpec[]): number {
  let c = 0;
  for (const e of events) {
    const kappa = Math.exp(e.meanLogJump + (e.stdLogJump * e.stdLogJump) / 2) - 1;
    c += e.dailyIntensity * kappa;
  }
  return c;
}
