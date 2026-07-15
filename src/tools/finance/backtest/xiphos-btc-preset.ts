import { XIPHOS_BTC_MARKOV_PROFILE, type XiphosMarkovProfile } from '../markov-distribution/xiphos-btc-profile.js';
import type { WalkForwardConfig } from './walk-forward.js';

/**
 * Map the xiphos BTC Markov profile onto the walk-forward config fields that
 * cramer-short already exposes. Levers not yet exposed as walk-forward config
 * (`studentTNu`, `hmmNumStates`, `abstentionThreshold`, meta window/threshold,
 * `trajectoryNSamples`) are omitted here and threaded separately.
 *
 * A fixed `randomState` is supplied so backtest comparisons are reproducible;
 * pass a `randomState` override to change the seed.
 */
export function xiphosBtcWalkForwardConfig(
  profile: XiphosMarkovProfile = XIPHOS_BTC_MARKOV_PROFILE,
  randomState = 42,
): Partial<WalkForwardConfig> {
  return {
    warmup: profile.warmupDays,
    transitionDecayOverride: profile.decayRate,
    btcReturnThresholdMultiplier: profile.returnThresholdMultiplier,
    btcBreakDivergenceThreshold: profile.breakDivergenceThreshold,
    enableGarchVol: profile.enableGarchVol,
    garchHorizonCap: profile.garchHorizonCap,
    garchRegimeCeiling: { calm: profile.garchCalmCeiling, turbulent: profile.garchTurbulentCeiling },
    enableEntropyCiModulation: profile.enableEntropyCiModulation,
    entropyWindowSize: profile.entropyWindowSize,
    useMetaRegime: profile.useMetaRegime,
    useConditionalTransitions: profile.useMetaRegime,
    randomState,
  };
}
