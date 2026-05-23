import { describe, expect } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
import { E2E_TIMEOUT_MS, runAgentE2EWithTimeoutRetry } from '@/utils/e2e-helpers.js';
import type { ToolEndEvent, ToolStartEvent } from '@/agent/types.js';
import { listForecastLabStructuredMutations } from '@/experiments/forecast-lab/profiles.js';
import { resolveForecastLabMarkovParameterDefaults, setForecastLabMarkovRuntimeDefaults, PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS } from '@/tools/finance/markov-distribution.js';
import { resolveForecastLabConformalParameterDefaults } from '@/tools/finance/conformal.js';
import { resolveForecastLabRegimeCalibratorDefaults } from '@/tools/finance/regime-calibrator.js';

const SOL_MUTATOR_QUERY =
  'Improve the SOL 1d/2d/3d Markov forecast workflow using mutator markov-shorter-reactive-window.';
const HYPE_MUTATOR_QUERY =
  'Improve the HYPE 1d/2d/3d Markov forecast workflow using mutator markov-shorter-reactive-window.';
const ORDINARY_SOL_HYPE_FORECAST_QUERY = [
  'Give me an ordinary Markov forecast for SOLUSD and HYPEUSDT over the next 1 to 3 days.',
  'Do not use any skill and do not use forecast_lab_run.',
  'Use markov_distribution once for SOLUSD and once for HYPEUSDT.',
  'Briefly compare the two assets and avoid BTC or GOLD substitutions.',
].join(' ');

interface ForecastLabRunPayload {
  readonly profileId?: string;
  readonly answer?: string;
}

function findToolStartEvent(result: { events: unknown[] }, tool: string): ToolStartEvent | undefined {
  return result.events.find((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_start' && candidate.tool === tool;
  });
}

function findToolStartEvents(result: { events: unknown[] }, tool: string): ToolStartEvent[] {
  return result.events.filter((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_start' && candidate.tool === tool;
  });
}

function findToolEndEvent(result: { events: unknown[] }, tool: string): ToolEndEvent | undefined {
  return result.events.find((event): event is ToolEndEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_end' && candidate.tool === tool;
  });
}

function parseForecastLabRunPayload(result: string): ForecastLabRunPayload {
  try {
    const parsed = JSON.parse(result) as { data?: ForecastLabRunPayload };
    return parsed.data ?? {};
  } catch {
    return {};
  }
}

function getAfterValue(profileId: 'sol-markov-short-horizon' | 'hype-markov-short-horizon', mutationId: string, parameterId: string) {
  const mutation = listForecastLabStructuredMutations(profileId).find((candidate) => candidate.id === mutationId);
  if (!mutation) {
    throw new Error(`Missing mutation fixture: ${profileId}/${mutationId}`);
  }

  const edit = mutation.edits.find((candidate) => candidate.parameterId === parameterId);
  if (!edit) {
    throw new Error(`Missing mutation parameter: ${profileId}/${mutationId}/${parameterId}`);
  }

  return edit.afterValue;
}

function getNumericAfterValue(profileId: 'sol-markov-short-horizon' | 'hype-markov-short-horizon', mutationId: string, parameterId: string): number {
  const value = getAfterValue(profileId, mutationId, parameterId);
  if (typeof value !== 'number') {
    throw new Error(`Expected numeric mutation value: ${profileId}/${mutationId}/${parameterId}`);
  }
  return value;
}

describe('forecast-lab asset-scope E2E', () => {
  e2eIt(
    'routes SOL mutation prompts into the SOL profile without BTC/GOLD profile leakage',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(SOL_MUTATOR_QUERY, {
        maxIterations: 6,
      });

      expect(result.toolsCalled).toContain('forecast_lab_run');
      expect(result.toolsCalled).toContain('skill');

      const runStart = findToolStartEvent(result, 'forecast_lab_run');
      expect(runStart?.args?.mutator).toBe('markov-shorter-reactive-window');

      const runEnd = findToolEndEvent(result, 'forecast_lab_run');
      expect(runEnd).toBeDefined();
      const payload = parseForecastLabRunPayload(runEnd!.result);
      expect(payload.profileId).toBe('sol-markov-short-horizon');
      expect(payload.answer?.toLowerCase()).toContain('sol-markov-short-horizon');
      expect(payload.answer?.toLowerCase()).not.toContain('btc-markov-ultra-short-horizon');
      expect(payload.answer?.toLowerCase()).not.toContain('gold-markov-short-horizon');
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'routes HYPE mutation prompts into the HYPE profile without BTC/GOLD profile leakage',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(HYPE_MUTATOR_QUERY, {
        maxIterations: 6,
      });

      expect(result.toolsCalled).toContain('forecast_lab_run');
      expect(result.toolsCalled).toContain('skill');

      const runStart = findToolStartEvent(result, 'forecast_lab_run');
      expect(runStart?.args?.mutator).toBe('markov-shorter-reactive-window');

      const runEnd = findToolEndEvent(result, 'forecast_lab_run');
      expect(runEnd).toBeDefined();
      const payload = parseForecastLabRunPayload(runEnd!.result);
      expect(payload.profileId).toBe('hype-markov-short-horizon');
      expect(payload.answer?.toLowerCase()).toContain('hype-markov-short-horizon');
      expect(payload.answer?.toLowerCase()).not.toContain('btc-markov-ultra-short-horizon');
      expect(payload.answer?.toLowerCase()).not.toContain('gold-markov-short-horizon');
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    'keeps the promoted SOL/HYPE defaults active on the ordinary forecast path without BTC/GOLD leakage',
    async () => {
      // Re-apply promoted defaults — runner.test.ts may have overwritten them
      // with snapshot values that don't match the current promoted config.
      setForecastLabMarkovRuntimeDefaults('sol', PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS);

      const solMarkov = resolveForecastLabMarkovParameterDefaults('sol');
      const solConformal = resolveForecastLabConformalParameterDefaults('sol');
      const solRegime = resolveForecastLabRegimeCalibratorDefaults('sol');
      const hypeMarkov = resolveForecastLabMarkovParameterDefaults('hype');

      expect(solMarkov.transitionMinObservations).toBe(
        getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'transitionMinObservations'),
      );
      expect(solMarkov.momentumLookback).toBe(
        getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'momentumLookback'),
      );
      expect(solConformal.scoreAggregationCalibrationWindow).toBe(
        getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'scoreAggregationCalibrationWindow'),
      );
      expect(solRegime.minSamplesPerRegime).toBe(
        getNumericAfterValue('sol-markov-short-horizon', 'markov-shorter-reactive-window', 'minSamplesPerRegime'),
      );
      expect(hypeMarkov.recommendedConfidenceThreshold).toBe(
        getNumericAfterValue('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty', 'recommendedConfidenceThreshold'),
      );
      expect(hypeMarkov.momentumAdjustmentScale).toBe(
        getNumericAfterValue('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty', 'momentumAdjustmentScale'),
      );
      expect(hypeMarkov.momentumAdjustmentClamp).toBe(
        getNumericAfterValue('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty', 'momentumAdjustmentClamp'),
      );
      expect(resolveForecastLabMarkovParameterDefaults('btc').recommendedConfidenceThreshold).not.toBe(
        hypeMarkov.recommendedConfidenceThreshold,
      );
      expect(resolveForecastLabMarkovParameterDefaults('gold').momentumLookback).not.toBe(
        solMarkov.momentumLookback,
      );

      const result = await runAgentE2EWithTimeoutRetry(ORDINARY_SOL_HYPE_FORECAST_QUERY, {
        maxIterations: 6,
      });

      expect(result.toolsCalled).toContain('markov_distribution');
      expect(result.toolsCalled).not.toContain('forecast_lab_run');
      expect(result.toolsCalled).not.toContain('skill');

      const markovCalls = findToolStartEvents(result, 'markov_distribution');
      expect(markovCalls.length).toBeGreaterThanOrEqual(2);
      const serializedArgs = JSON.stringify(markovCalls.map((call) => call.args));
      expect(serializedArgs).toContain('SOLUSD');
      expect(serializedArgs).toContain('HYPEUSDT');
      expect(result.answer.toLowerCase()).not.toContain('btc-markov-ultra-short-horizon');
      expect(result.answer.toLowerCase()).not.toContain('gold-markov-short-horizon');
    },
    E2E_TIMEOUT_MS,
  );
});
