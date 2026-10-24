import type { AIMessage } from '@langchain/core/messages';
import type { ToolCallRecord } from './scratchpad.js';
import { hasErrorLikeToolResult } from './query-router/types.js';

/**
 * Loop-breaker helpers.
 *
 * When the model re-emits a tool call whose exact (tool, args) already produced
 * an error result, re-running it cannot make progress. Detecting these lets the
 * agent loop skip the doomed calls — and break to synthesis when only such
 * repeats remain — instead of burning the whole iteration budget on identical
 * failures.
 */

type ModelToolCall = NonNullable<AIMessage['tool_calls']>[number];

/** Deterministic, key-order-independent serialization of tool-call arguments. */
export function stableStringifyArgs(args: unknown): string {
  return JSON.stringify(sortValue(args));
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortValue);
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([key, val]) => [key, sortValue(val)]),
    );
  }
  return value;
}

/**
 * True when a prior scratchpad record has the same tool name, equivalent
 * arguments, and an error-like result — i.e. re-running this exact call would
 * just fail the same way again.
 */
export function isRepeatedFailingToolCall(call: ModelToolCall, records: ToolCallRecord[]): boolean {
  const signature = stableStringifyArgs(call.args ?? {});
  return records.some((record) =>
    record.tool === call.name
    && hasErrorLikeToolResult(record.result)
    && stableStringifyArgs(record.args ?? {}) === signature,
  );
}

/**
 * Partitions the model's tool calls into those safe to execute and those that
 * exactly repeat an earlier failed call and should be skipped.
 */
export function partitionRepeatedFailingToolCalls(
  response: AIMessage,
  records: ToolCallRecord[],
): { executable: ModelToolCall[]; skipped: ModelToolCall[] } {
  const executable: ModelToolCall[] = [];
  const skipped: ModelToolCall[] = [];

  for (const call of response.tool_calls ?? []) {
    if (isRepeatedFailingToolCall(call, records)) {
      skipped.push(call);
    } else {
      executable.push(call);
    }
  }

  return { executable, skipped };
}
