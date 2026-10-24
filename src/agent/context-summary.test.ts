import { describe, expect, it } from 'bun:test';
import { buildContextSummaryText, MAX_CONTEXT_SUMMARY_CHARS } from './context-summary.js';

describe('buildContextSummaryText — merged summary bound', () => {
  it('keeps the merged summary bounded across overflow cycles', () => {
    let summary: string | null = null;
    for (let i = 0; i < 40; i += 1) {
      summary = buildContextSummaryText(
        [{ toolName: 'get_financials', args: { ticker: 'AAPL' }, snippet: `cycle ${i} ${'x'.repeat(1500)}` }],
        summary,
      );
    }

    expect(summary!.length).toBeLessThanOrEqual(MAX_CONTEXT_SUMMARY_CHARS);
    expect(summary).toContain('cycle 39');
    expect(summary).not.toContain('cycle 0 ');
  });
});
