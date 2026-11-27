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
  /** True when the run continues a conversation, so `query` may refer to earlier turns. */
  readonly followUp: boolean;
  readonly scratchpad: Scratchpad;
  readonly tokenCounter: TokenCounter;
  readonly startTime: number;
  iteration: number;
  forecastLabGuard?: ForecastLabGuard;
  /** Initial prompt context (history, memory, Polymarket) reused by every iteration prompt. */
  promptContext?: string;
  /** `tool:args` keys of forced tool calls already made this run; never forced twice. */
  readonly forcedToolCalls: Set<string>;
}

export function createRunContext(query: string, followUp = false): RunContext {
  return {
    query,
    followUp,
    scratchpad: new Scratchpad(query),
    tokenCounter: new TokenCounter(),
    startTime: Date.now(),
    iteration: 0,
    forcedToolCalls: new Set(),
  };
}
