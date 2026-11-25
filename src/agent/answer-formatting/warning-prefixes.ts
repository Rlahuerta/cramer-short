import type { ToolCallRecord } from '../scratchpad.js';
import {
  detectBtcShortHorizonDisagreement,
  hasSuccessfulMarkovDistributionForQuery,
  isDistributionQuery,
} from '../query-router.js';
import { parseToolCallData } from './tool-call-utils.js';

export function buildDistributionWarningPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!isDistributionQuery(query)) return null;
  if (hasSuccessfulMarkovDistributionForQuery(query, toolCalls)) return null;

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    const data = parseToolCallData(call);
    if (!data) continue;

    const payload = data;
    if (payload['_tool'] !== 'markov_distribution' || payload['status'] !== 'abstain') continue;

    const abstainReasons = Array.isArray(payload['abstainReasons'])
      ? payload['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
      : [];

    return [
      '## Warning: no calibrated Markov terminal distribution was available',
      '',
      'The Markov distribution workflow abstained for this query. Any fallback analysis below must be treated as non-distribution context unless it comes directly from a validated canonical Markov payload.',
      ...(abstainReasons.length > 0
        ? ['', 'Key abstain reasons:', ...abstainReasons.map((reason) => `- ${reason}`)]
        : []),
      '',
      '---',
      '',
    ].join('\n');
  }

  return [
    '## Warning: no validated Markov distribution was produced',
    '',
    'Cramer-Short did not produce a successful non-abstaining `markov_distribution` result for this distribution query. Any answer below should be read as fallback analysis, not a calibrated probability distribution.',
    '',
    '---',
    '',
  ].join('\n');
}

export function buildForecastDisagreementPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!detectBtcShortHorizonDisagreement(query, toolCalls)) return null;

  return [
    '## Warning: BTC short-horizon signals are mixed',
    '',
    'Markov and Polymarket are pointing in different directions for this BTC short-horizon forecast. Read any directional takeaway below as mixed evidence with moderated confidence, not a high-conviction signal.',
    '',
    '---',
    '',
  ].join('\n');
}
