import { Scratchpad } from './scratchpad.js';
import { TokenCounter } from './token-counter.js';

export interface ForecastLabGuard {
  readonly recommendedProfileId: string | null;
}

/**
 * Mutable state for a single agent run.
 */
export interface RunContext {
  readonly query: string;
  readonly scratchpad: Scratchpad;
  readonly tokenCounter: TokenCounter;
  readonly startTime: number;
  iteration: number;
  forecastLabGuard?: ForecastLabGuard;
}

export function createRunContext(query: string): RunContext {
  return {
    query,
    scratchpad: new Scratchpad(query),
    tokenCounter: new TokenCounter(),
    startTime: Date.now(),
    iteration: 0,
  };
}
