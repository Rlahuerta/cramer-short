import { describe, expect, it } from 'bun:test';
import { listForecastLabStructuredMutations } from './profiles.js';
import {
  FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
  resolveForecastLabMarkovParameterDefaults,
} from '../../tools/finance/markov-distribution.js';
import { resolveForecastLabConformalParameterDefaults } from '../../tools/finance/conformal.js';
import { resolveForecastLabRegimeCalibratorDefaults } from '../../tools/finance/regime-calibrator.js';

function getMutation(profileId: 'sol-markov-short-horizon' | 'hype-markov-short-horizon', mutationId: string) {
  const mutation = listForecastLabStructuredMutations(profileId).find((candidate) => candidate.id === mutationId);
  if (!mutation) {
    throw new Error(`Missing mutation fixture: ${profileId}/${mutationId}`);
  }
  return mutation;
}

function getAfterValue(profileId: 'sol-markov-short-horizon' | 'hype-markov-short-horizon', mutationId: string, parameterId: string) {
  const edit = getMutation(profileId, mutationId).edits.find((candidate) => candidate.parameterId === parameterId);
  if (!edit) {
    throw new Error(`Missing parameter ${parameterId} for ${profileId}/${mutationId}`);
  }
  return edit.afterValue;
}

function getNumericAfterValue(profileId: 'sol-markov-short-horizon' | 'hype-markov-short-horizon', mutationId: string, parameterId: string): number {
  const value = getAfterValue(profileId, mutationId, parameterId);
  if (typeof value !== 'number') {
    throw new Error(`Expected numeric parameter ${parameterId} for ${profileId}/${mutationId}`);
  }
  return value;
}

describe('forecast-lab promoted runtime defaults', () => {
  it('ships the promoted SOL shorter-reactive-window as the live SOL baseline', () => {
    const markov = resolveForecastLabMarkovParameterDefaults('sol');
    const conformal = resolveForecastLabConformalParameterDefaults('sol');
    const regime = resolveForecastLabRegimeCalibratorDefaults('sol');

    expect(markov.transitionMinObservations).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'transitionMinObservations'),
    );
    expect(markov.structuralBreakMinLength).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'structuralBreakMinLength'),
    );
    expect(markov.momentumLookback).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'momentumLookback'),
    );
    expect(markov.momentumAdjustmentScale).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'momentumAdjustmentScale'),
    );
    expect(markov.momentumAdjustmentClamp).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'momentumAdjustmentClamp'),
    );
    expect(conformal.scoreAggregationMinSamples).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'scoreAggregationMinSamples'),
    );
    expect(conformal.scoreAggregationCalibrationWindow).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'scoreAggregationCalibrationWindow'),
    );
    expect(regime.minSamplesPerRegime).toBe(
      getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'minSamplesPerRegime'),
    );
  });

  it('ships the promoted HYPE lower-confidence mutator as the live HYPE baseline', () => {
    const hype = resolveForecastLabMarkovParameterDefaults('hype');

    expect(hype.recommendedConfidenceThreshold).toBe(
      getNumericAfterValue('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty', 'recommendedConfidenceThreshold'),
    );
    expect(hype.momentumAdjustmentScale).toBe(
      getNumericAfterValue('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty', 'momentumAdjustmentScale'),
    );
    expect(hype.momentumAdjustmentClamp).toBe(
      getNumericAfterValue('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty', 'momentumAdjustmentClamp'),
    );
  });

  it('keeps promoted SOL/HYPE defaults isolated from shared, BTC, and GOLD scopes', () => {
    const shared = resolveForecastLabMarkovParameterDefaults('shared');
    const btc = resolveForecastLabMarkovParameterDefaults('btc');
    const gold = resolveForecastLabMarkovParameterDefaults('gold');
    const sol = resolveForecastLabMarkovParameterDefaults('sol');
    const hype = resolveForecastLabMarkovParameterDefaults('hype');

    expect(shared.recommendedConfidenceThreshold).toBe(FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold);
    expect(btc.recommendedConfidenceThreshold).toBe(FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold);
    expect(gold.recommendedConfidenceThreshold).toBe(FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold);
    expect(sol.momentumLookback).not.toBe(shared.momentumLookback);
    expect(hype.recommendedConfidenceThreshold).not.toBe(shared.recommendedConfidenceThreshold);
    expect(sol.momentumLookback).not.toBe(gold.momentumLookback);
    expect(hype.recommendedConfidenceThreshold).not.toBe(btc.recommendedConfidenceThreshold);
  });
});
