import { describe, expect } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
import { E2E_TIMEOUT_MS, runAgentE2EWithTimeoutRetry } from '@/utils/e2e-helpers.js';
import type { ToolEndEvent, ToolStartEvent } from '@/agent/types.js';

const SOL_MUTATOR_QUERY =
  'Improve the SOL 1d/2d/3d Markov forecast workflow using mutator markov-shorter-reactive-window.';
const HYPE_MUTATOR_QUERY =
  'Improve the HYPE 1d/2d/3d Markov forecast workflow using mutator markov-shorter-reactive-window.';

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
});
