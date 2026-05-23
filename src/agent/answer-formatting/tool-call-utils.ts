import type { ToolCallRecord } from '../scratchpad.js';

export function formatDiagnosticPrice(value: unknown): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value >= 1000
    ? `$${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
    : `$${value.toFixed(2)}`;
}

export function formatDiagnosticNumber(value: unknown, digits = 3): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value.toFixed(digits);
}

export function parseToolCallData(call: ToolCallRecord): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(call.result) as { data?: unknown };
    return parsed?.data && typeof parsed.data === 'object'
      ? parsed.data as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

export function hasSuccessfulMarkovDistribution(toolCalls: ToolCallRecord[]): boolean {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    try {
      const parsed = JSON.parse(call.result) as { data?: { _tool?: string; status?: string } };
      if (parsed?.data?._tool === 'markov_distribution' && parsed.data.status === 'ok') {
        return true;
      }
    } catch {
      continue;
    }
  }
  return false;
}

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function isFinitePositiveNumber(value: unknown): value is number {
  return isFiniteNumber(value) && value > 0;
}

function getPositiveIntegerArg(args: Record<string, unknown>, key: string): number | null {
  return isFinitePositiveNumber(args[key]) ? Math.trunc(args[key]) : null;
}

export function matchesTickerAndOptionalHorizon(
  args: Record<string, unknown>,
  ticker: string | null,
  horizonKey: string,
  horizon: number | null,
): boolean {
  if (ticker) {
    const existingTicker = typeof args['ticker'] === 'string' ? args['ticker'].toUpperCase() : null;
    const expectedTicker = ticker.toUpperCase();
    const equivalentCryptoTicker = expectedTicker.endsWith('-USD')
      && existingTicker === expectedTicker.replace(/-USD$/, '');
    if (existingTicker !== expectedTicker && !equivalentCryptoTicker) return false;
  }

  if (horizon !== null && getPositiveIntegerArg(args, horizonKey) !== horizon) {
    return false;
  }

  return true;
}
