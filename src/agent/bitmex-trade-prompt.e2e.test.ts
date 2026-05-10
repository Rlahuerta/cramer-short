import { describe, expect } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
import { E2E_TIMEOUT_MS, runAgentE2EWithTimeoutRetry } from '@/utils/e2e-helpers.js';
import type { ToolStartEvent } from '@/agent/types.js';

const BITMEX_TRADE_BRIEF_QUERY = [
  'BitMEX trade brief for SOLUSD and HYPEUSDT for the next 1 to 3 days.',
  'This is a live market-analysis request.',
  'Do not use any skill and do not use forecast_lab_run.',
  'Start with one bitmex_market call for SOLUSD and HYPEUSDT.',
  'Drop any market that is clearly untradeable from spread, liquidity, funding, or leverage.',
  'After that, always run markov_distribution once for SOLUSD and once for HYPEUSDT.',
  'For each market, choose the single most relevant horizon inside 1d, 2d, and 3d instead of brute-forcing every combination.',
  'If one market still has a usable edge, use forecast_arbitrator once for that market.',
  'Reject setups with wide spread, thin liquidity, low confidence, or structural-break warnings.',
  'Return a readable brief with a comparison table, the best setup or no trade, and simple entry/stop/target/leverage notes.',
  'Your final line must be exactly one of: DECISION: LONG SOLUSD, DECISION: SHORT SOLUSD, DECISION: LONG HYPEUSDT, DECISION: SHORT HYPEUSDT, DECISION: NO TRADE.',
].join(' ');

function findToolStartEvents(result: { events: unknown[] }, tool: string): ToolStartEvent[] {
  return result.events.filter((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_start' && candidate.tool === tool;
  });
}

describe('Agent E2E — BitMEX trade prompt', () => {
  e2eIt(
    'uses BitMEX market data flow without routing into forecast lab',
    async () => {
      const result = await runAgentE2EWithTimeoutRetry(BITMEX_TRADE_BRIEF_QUERY, {
        maxIterations: 6,
      });

      expect(result.toolsCalled).toContain('bitmex_market');
      expect(result.toolsCalled).not.toContain('forecast_lab_run');

      const bitmexCalls = findToolStartEvents(result, 'bitmex_market');
      expect(bitmexCalls.length).toBeGreaterThan(0);
      expect(JSON.stringify(bitmexCalls[0]?.args)).toContain('SOLUSD');
      expect(JSON.stringify(bitmexCalls[0]?.args)).toContain('HYPEUSDT');

      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );
});
