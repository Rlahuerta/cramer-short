import { describe, expect, it } from 'bun:test';
import {
  XIPHOS_BTC_MARKOV_PROFILE,
  XIPHOS_BTC_MARKOV_PROFILE_1D,
  XIPHOS_BTC_MARKOV_PROFILE_6H,
} from './xiphos-btc-profile.js';

describe('XIPHOS_BTC_MARKOV_PROFILE', () => {
  it('matches xiphos vbparam_1d values (provenance lock)', () => {
    expect(XIPHOS_BTC_MARKOV_PROFILE_1D).toEqual({
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
    });
  });

  it('uses the 1d profile as the active BTC profile', () => {
    expect(XIPHOS_BTC_MARKOV_PROFILE).toBe(XIPHOS_BTC_MARKOV_PROFILE_1D);
  });

  it('holds valid invariants (both profiles)', () => {
    for (const p of [XIPHOS_BTC_MARKOV_PROFILE_1D, XIPHOS_BTC_MARKOV_PROFILE_6H]) {
      expect(p.decayRate).toBeGreaterThan(0);
      expect(p.decayRate).toBeLessThan(1);
      expect(Number.isInteger(p.studentTNu)).toBe(true);
      expect(p.studentTNu).toBeGreaterThan(2);
      expect(Number.isInteger(p.hmmNumStates)).toBe(true);
      expect(p.hmmNumStates).toBeGreaterThanOrEqual(2);
      expect(p.garchCalmCeiling).toBeLessThan(p.garchTurbulentCeiling);
      expect(p.metaHighVolThreshold).toBeGreaterThan(0);
      expect(p.metaHighVolThreshold).toBeLessThan(1);
      expect(p.volumeThresholdMultiplier).toBeGreaterThan(1);
      expect(p.enableGarchVol).toBe(true);
      expect(p.useMetaRegime).toBe(true);
    }
  });
});
