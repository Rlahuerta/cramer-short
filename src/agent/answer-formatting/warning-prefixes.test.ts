import { describe, expect, test } from 'bun:test';
import type { ToolCallRecord } from '../scratchpad.js';
import { buildDistributionWarningPrefix } from './warning-prefixes.js';

function markovCall(
  ticker: string,
  horizon: number,
  status: 'ok' | 'abstain',
  abstainReasons: string[] = [],
): ToolCallRecord {
  return {
    tool: 'markov_distribution',
    args: { ticker, horizon },
    result: JSON.stringify({
      data: {
        _tool: 'markov_distribution',
        status,
        ...(abstainReasons.length > 0 ? { abstainReasons } : {}),
      },
    }),
  };
}

describe('buildDistributionWarningPrefix markov success scoping', () => {
  test('warns when the queried ticker abstained even though another ticker succeeded', () => {
    const toolCalls = [
      markovCall('BTC', 7, 'ok'),
      markovCall('ETH', 7, 'abstain', ['insufficient regime history']),
    ];

    const warning = buildDistributionWarningPrefix(
      'What is the probability distribution for ETH price in 7 days?',
      toolCalls,
    );

    expect(warning).not.toBeNull();
    expect(warning).toContain('abstained');
  });

  test('suppresses the warning when the queried ticker succeeded', () => {
    const toolCalls = [
      markovCall('BTC', 7, 'ok'),
      markovCall('ETH', 7, 'abstain'),
    ];

    expect(
      buildDistributionWarningPrefix(
        'What is the probability distribution for BTC price in 7 days?',
        toolCalls,
      ),
    ).toBeNull();
  });
});
