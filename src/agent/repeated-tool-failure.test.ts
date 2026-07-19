/**
 * TDD tests for the repeated-tool-failure loop breaker.
 *
 * Defense-in-depth: if the model re-emits a tool call whose exact (tool, args)
 * already produced an error result, re-running it cannot make progress and
 * risks burning the whole iteration budget. These helpers detect such repeats
 * so the agent loop can skip them (and break to synthesis when only repeats
 * remain).
 */
import { describe, it, expect } from 'bun:test';
import { AIMessage } from '@langchain/core/messages';
import type { ToolCallRecord } from './scratchpad.js';
import {
  isRepeatedFailingToolCall,
  partitionRepeatedFailingToolCalls,
} from './repeated-tool-failure.js';

const ERROR_RESULT = '{"data":{"_tool":"markov_distribution","error":"Could not fetch prices for PLAN"}}';
const OK_RESULT = '{"data":{"_tool":"markov_distribution","status":"ok","expectedReturn":0.01}}';

function record(tool: string, args: Record<string, unknown>, result: string): ToolCallRecord {
  return { tool, args, result };
}

function aiMsg(calls: Array<{ name: string; args: Record<string, unknown> }>): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: calls.map((c, i) => ({ id: `c${i}`, name: c.name, args: c.args, type: 'tool_call' as const })),
    additional_kwargs: {},
  });
}

describe('isRepeatedFailingToolCall', () => {
  it('flags a call that exactly repeats a prior errored call', () => {
    const records = [record('markov_distribution', { ticker: 'PLAN', horizon: 1 }, ERROR_RESULT)];
    const call = { id: 'c1', name: 'markov_distribution', args: { ticker: 'PLAN', horizon: 1 }, type: 'tool_call' as const };

    expect(isRepeatedFailingToolCall(call, records)).toBe(true);
  });

  it('is order-independent on argument keys', () => {
    const records = [record('markov_distribution', { ticker: 'PLAN', horizon: 1 }, ERROR_RESULT)];
    const call = { id: 'c1', name: 'markov_distribution', args: { horizon: 1, ticker: 'PLAN' }, type: 'tool_call' as const };

    expect(isRepeatedFailingToolCall(call, records)).toBe(true);
  });

  it('does not flag when args differ', () => {
    const records = [record('markov_distribution', { ticker: 'PLAN', horizon: 1 }, ERROR_RESULT)];
    const call = { id: 'c1', name: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 1 }, type: 'tool_call' as const };

    expect(isRepeatedFailingToolCall(call, records)).toBe(false);
  });

  it('does not flag when the prior identical call succeeded', () => {
    const records = [record('markov_distribution', { ticker: 'PLAN', horizon: 1 }, OK_RESULT)];
    const call = { id: 'c1', name: 'markov_distribution', args: { ticker: 'PLAN', horizon: 1 }, type: 'tool_call' as const };

    expect(isRepeatedFailingToolCall(call, records)).toBe(false);
  });

  it('does not flag when the tool name differs', () => {
    const records = [record('markov_distribution', { ticker: 'PLAN' }, ERROR_RESULT)];
    const call = { id: 'c1', name: 'polymarket_forecast', args: { ticker: 'PLAN' }, type: 'tool_call' as const };

    expect(isRepeatedFailingToolCall(call, records)).toBe(false);
  });

  it('recognizes an "Error:"-prefixed prior result as a failure', () => {
    const records = [record('get_market_data', { ticker: 'PLAN' }, 'Error: not found')];
    const call = { id: 'c1', name: 'get_market_data', args: { ticker: 'PLAN' }, type: 'tool_call' as const };

    expect(isRepeatedFailingToolCall(call, records)).toBe(true);
  });
});

describe('partitionRepeatedFailingToolCalls', () => {
  it('splits calls into executable and skipped', () => {
    const records = [record('markov_distribution', { ticker: 'PLAN', horizon: 1 }, ERROR_RESULT)];
    const response = aiMsg([
      { name: 'markov_distribution', args: { ticker: 'PLAN', horizon: 1 } },
      { name: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 1 } },
    ]);

    const { executable, skipped } = partitionRepeatedFailingToolCalls(response, records);

    expect(skipped.map((c) => c.name)).toEqual(['markov_distribution']);
    expect(executable.map((c) => c.name)).toEqual(['polymarket_forecast']);
  });

  it('marks every call executable when none previously failed', () => {
    const response = aiMsg([{ name: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 1 } }]);

    const { executable, skipped } = partitionRepeatedFailingToolCalls(response, []);

    expect(skipped).toHaveLength(0);
    expect(executable).toHaveLength(1);
  });

  it('marks all calls skipped when all are repeated failures', () => {
    const records = [record('markov_distribution', { ticker: 'PLAN', horizon: 1 }, ERROR_RESULT)];
    const response = aiMsg([{ name: 'markov_distribution', args: { ticker: 'PLAN', horizon: 1 } }]);

    const { executable, skipped } = partitionRepeatedFailingToolCalls(response, records);

    expect(executable).toHaveLength(0);
    expect(skipped).toHaveLength(1);
  });

  it('returns empty partitions for a response with no tool calls', () => {
    const response = new AIMessage({ content: 'done', additional_kwargs: {} });

    const { executable, skipped } = partitionRepeatedFailingToolCalls(response, []);

    expect(executable).toHaveLength(0);
    expect(skipped).toHaveLength(0);
  });
});
