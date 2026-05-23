import type { ToolCallRecord } from '../scratchpad.js';
import { isDistributionQuery } from '../query-router.js';
import {
  formatDiagnosticNumber,
  formatDiagnosticPrice,
  hasSuccessfulMarkovDistribution,
} from './tool-call-utils.js';

export function buildAbstainingMarkovAnswer(toolCalls: ToolCallRecord[]): string | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(call.result);
    } catch {
      continue;
    }

    const data = parsed && typeof parsed === 'object' ? (parsed as { data?: unknown }).data : null;
    if (!data || typeof data !== 'object') continue;

    const payload = data as Record<string, unknown>;
    if (payload['_tool'] !== 'markov_distribution' || payload['status'] !== 'abstain') continue;

    const canonical = payload['canonical'] && typeof payload['canonical'] === 'object'
      ? payload['canonical'] as Record<string, unknown>
      : {};
    const diagnostics = canonical['diagnostics'] && typeof canonical['diagnostics'] === 'object'
      ? canonical['diagnostics'] as Record<string, unknown>
      : {};
    const abstainReasons = Array.isArray(payload['abstainReasons'])
      ? payload['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
      : [];

    const ticker = typeof canonical['ticker'] === 'string' ? canonical['ticker'] : 'This asset';
    const horizon = typeof canonical['horizon'] === 'number' ? canonical['horizon'] : null;
    const currentPrice = formatDiagnosticPrice(canonical['currentPrice']);
    const trustedAnchors = typeof diagnostics['trustedAnchors'] === 'number' ? diagnostics['trustedAnchors'] : null;
    const totalAnchors = typeof diagnostics['totalAnchors'] === 'number' ? diagnostics['totalAnchors'] : null;
    const anchorQuality = typeof diagnostics['anchorQuality'] === 'string' ? diagnostics['anchorQuality'] : null;
    const outOfSampleR2 = formatDiagnosticNumber(diagnostics['outOfSampleR2']);
    const structuralBreakDetected = diagnostics['structuralBreakDetected'] === true;
    const structuralBreakDivergence = formatDiagnosticNumber(diagnostics['structuralBreakDivergence']);
    const predictionConfidence = formatDiagnosticNumber(diagnostics['predictionConfidence'], 2);

    const lines = [
      `## ${ticker} ${horizon !== null ? `${horizon}-Day ` : ''}Probability Distribution: Model Abstained`,
      '',
      'A calibrated probability distribution is not available. `markov_distribution` explicitly abstained, so no replacement scenario probabilities are shown.',
      '',
      '## Why it abstained',
      ...(abstainReasons.length > 0
        ? abstainReasons.map((reason) => `- ${reason}`)
        : ['- The tool marked this run as diagnostics-only.']),
      '',
      '## Diagnostics',
      '| Metric | Value |',
      '|--------|-------|',
      ...(currentPrice ? [`| Current price | ${currentPrice} |`] : []),
      `| Trusted anchors | ${trustedAnchors !== null && totalAnchors !== null ? `${trustedAnchors} / ${totalAnchors}` : 'N/A'} |`,
      `| Anchor quality | ${anchorQuality ?? 'N/A'} |`,
      `| Out-of-sample R² | ${outOfSampleR2 ?? 'N/A'} |`,
      `| Structural break | ${structuralBreakDetected ? `Yes${structuralBreakDivergence ? ` (${structuralBreakDivergence})` : ''}` : 'No'} |`,
      `| Prediction confidence | ${predictionConfidence ?? 'N/A'} |`,
      '',
      '## Interpretation',
      'Use later prediction-market searches only as diagnostics about missing or low-quality anchors. They do not justify a replacement calibrated distribution for this horizon.',
      '',
      '## Next best option',
      '- Wait for horizon-matched terminal threshold markets to list, or',
      '- Use `polymarket_forecast` for a point estimate with a confidence interval instead of a full distribution.',
    ];

    return lines.join('\n');
  }

  return null;
}

export function buildUnavailableDistributionAnswer(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!isDistributionQuery(query)) return null;
  if (hasSuccessfulMarkovDistribution(toolCalls)) return null;

  return [
    '## Probability Distribution Unavailable',
    '',
    'A calibrated price-distribution answer is not available for this query. No successful non-abstaining `markov_distribution` result was produced, so I am not emitting substitute distribution buckets or historical-volatility scenario percentages.',
    '',
    '## Why this answer stops here',
    '- A price-distribution request needs horizon-matched terminal threshold anchors or a validated canonical Markov distribution.',
    '- Neither condition was met in this run, so any bucketed probability table would be speculative rather than grounded.',
    '',
    '## Safe next options',
    '- Wait for terminal threshold markets that match the requested horizon, or',
    '- Use `polymarket_forecast` for a point estimate with a confidence interval instead of a full distribution.',
  ].join('\n');
}
