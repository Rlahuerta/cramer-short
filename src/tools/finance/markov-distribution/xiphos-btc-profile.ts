/** Mirrors `research/models/markov/xiphos_btc_profile.py`. */

/**
 * BTC/USD Markov parameters ported from the xiphos optimizer at
 * `proopsios/trading/xiphos/markov/backtest/_parameter_sources.py`.
 *
 * These are BTC/USD-tuned design levers. `vbparam_1d` is the optimizer-precise
 * 1-day profile, which is cramer-short's finest daily forecast horizon, so it is
 * the profile used here. The 6h profile is optimizer-precise too but sub-daily,
 * so it is only recorded for provenance (cramer-short does not forecast at 6h).
 *
 * `break_ci_slope` / `break_ci_max` and `target_steps` have no direct cramer-short
 * equivalent (cramer-short widens CIs via `entropyKappa` / conformal calibration),
 * so they are recorded here for provenance but not wired into the forecast.
 */
export interface XiphosMarkovProfile {
  /** Calendar days of training history before the first prediction (backtest lever). */
  readonly warmupDays: number;
  /** Exponential recency weight for transition-matrix counting (↔ transitionDecay). */
  readonly decayRate: number;
  /** Sensitivity of bull/bear vs sideways regime classification. */
  readonly returnThresholdMultiplier: number;
  /** Frobenius-divergence threshold for structural-break detection. */
  readonly breakDivergenceThreshold: number;
  /** Minimum |p_up - 0.5|*2 required before emitting a directional call. */
  readonly abstentionThreshold: number;
  /** Degrees of freedom for Student-t trajectory MC innovations. */
  readonly studentTNu: number;
  /** GARCH horizon after which scaling soft-blends back toward 1.0. */
  readonly garchHorizonCap: number;
  /** Upper clamp for GARCH scaling in calm regimes. */
  readonly garchCalmCeiling: number;
  /** Upper clamp for GARCH scaling in turbulent regimes. */
  readonly garchTurbulentCeiling: number;
  /** Rolling window for entropy-based CI modulation. */
  readonly entropyWindowSize: number;
  /** Number of HMM states for HMM-based regime detection. */
  readonly hmmNumStates: number;
  /** Lookback window for meta-regime volatility classification. */
  readonly metaVolWindow: number;
  /** Percentile threshold separating low/high-volatility meta-regimes. */
  readonly metaHighVolThreshold: number;
  /** Lookback window for volume-regime classification. */
  readonly volumeLookback: number;
  /** Sensitivity of high/low volume-regime classification. */
  readonly volumeThresholdMultiplier: number;
  /** Apply GARCH volatility scaling in the trajectory interval. */
  readonly enableGarchVol: boolean;
  /** Scale confidence intervals by transition-entropy z-score. */
  readonly enableEntropyCiModulation: boolean;
  /** Use HMM-based regime detection instead of threshold classification. */
  readonly useHmmRegime: boolean;
  /** Enable AH-HMM meta-regime conditioning. */
  readonly useMetaRegime: boolean;
  /** Number of Monte Carlo samples for trajectory simulation. */
  readonly trajectoryNSamples: number;
}

/**
 * xiphos `vbparam_1d` (BTC/USD, optimizer-precise 1-day profile).
 * Source: `_parameter_sources.py` `vbparam_1d` (large simulation, dir_acc 55.6%).
 */
export const XIPHOS_BTC_MARKOV_PROFILE_1D: XiphosMarkovProfile = {
  warmupDays: 54,
  decayRate: 0.9494,
  returnThresholdMultiplier: 0.6771,
  breakDivergenceThreshold: 0.1165,
  abstentionThreshold: 0.002,
  studentTNu: 16,
  garchHorizonCap: 19,
  garchCalmCeiling: 1.3603,
  garchTurbulentCeiling: 3.097,
  entropyWindowSize: 38,
  hmmNumStates: 2,
  metaVolWindow: 40,
  metaHighVolThreshold: 0.7824,
  volumeLookback: 8,
  volumeThresholdMultiplier: 1.4051,
  enableGarchVol: true,
  enableEntropyCiModulation: true,
  useHmmRegime: true,
  useMetaRegime: true,
  trajectoryNSamples: 200,
};

/**
 * xiphos `vbparam_6h` (BTC/USD, optimizer-precise 6-hour profile).
 * Recorded for provenance; NOT applied because cramer-short forecasts daily.
 */
export const XIPHOS_BTC_MARKOV_PROFILE_6H: XiphosMarkovProfile = {
  warmupDays: 50,
  decayRate: 0.93621,
  returnThresholdMultiplier: 0.43254,
  breakDivergenceThreshold: 0.23088,
  abstentionThreshold: 0.00366,
  studentTNu: 12,
  garchHorizonCap: 10,
  garchCalmCeiling: 1.4054,
  garchTurbulentCeiling: 3.61353,
  entropyWindowSize: 87,
  hmmNumStates: 4,
  metaVolWindow: 11,
  metaHighVolThreshold: 0.7989,
  volumeLookback: 45,
  volumeThresholdMultiplier: 1.8659,
  enableGarchVol: true,
  enableEntropyCiModulation: true,
  useHmmRegime: true,
  useMetaRegime: true,
  trajectoryNSamples: 200,
};

/** The BTC profile cramer-short uses (the optimizer-precise 1-day profile). */
export const XIPHOS_BTC_MARKOV_PROFILE: XiphosMarkovProfile = XIPHOS_BTC_MARKOV_PROFILE_1D;

/**
 * The xiphos BTC forecast levers to apply in the live Markov tool for BTC.
 *
 * Only the levers whose benefit is verified by the walk-forward backtest are
 * applied (transition decay, regime-classification threshold, GARCH vol scaling,
 * entropy CI modulation, meta-regime conditioning). Deliberately excluded:
 * - `studentTNu` / `trajectoryNSamples`: the backtest improvement was obtained
 *   WITHOUT changing these, and `nu` conflicts with cramer-short's crypto
 *   asset-class profile (nu=3, fatter tails) — so they are left as-is.
 * - `breakDivergenceThreshold`: cramer-short's per-horizon BTC live policy is
 *   kept (it is more granular than xiphos's single value).
 * - `warmup` / `randomState`: backtest-only concerns.
 */
export function xiphosBtcLiveParams(profile: XiphosMarkovProfile = XIPHOS_BTC_MARKOV_PROFILE): {
  readonly transitionDecayOverride: number;
  readonly btcReturnThresholdMultiplier: number;
  readonly enableGarchVol: boolean;
  readonly garchHorizonCap: number;
  readonly garchRegimeCeiling: { readonly calm: number; readonly turbulent: number };
  readonly enableEntropyCiModulation: boolean;
  readonly useMetaRegime: boolean;
  readonly useConditionalTransitions: boolean;
} {
  return {
    transitionDecayOverride: profile.decayRate,
    btcReturnThresholdMultiplier: profile.returnThresholdMultiplier,
    enableGarchVol: profile.enableGarchVol,
    garchHorizonCap: profile.garchHorizonCap,
    garchRegimeCeiling: { calm: profile.garchCalmCeiling, turbulent: profile.garchTurbulentCeiling },
    enableEntropyCiModulation: profile.enableEntropyCiModulation,
    useMetaRegime: profile.useMetaRegime,
    useConditionalTransitions: profile.useMetaRegime,
  };
}
