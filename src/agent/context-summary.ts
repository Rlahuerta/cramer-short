// ============================================================================
// Context summary helpers (exported for unit tests)
// ============================================================================

/**
 * Numeric fact patterns extracted from tool results before they are cleared
 * from context. Each pattern captures a distinct type of financial data.
 */
export const FACT_PATTERNS: ReadonlyArray<RegExp> = [
  /\$[\d,]+(?:\.\d{1,2})?(?:\s*[BMK](?:illion)?)?/gi,  // prices / market caps
  /[-+]?\d+(?:\.\d+)?%/g,                                // percentages
  /\b(?:IC|ICIR|RankIC)\s*[:=]\s*[-+]?\d+\.\d+/gi,     // factor IC values
  /\bP\/E\s*[:=]?\s*\d+(?:\.\d+)?x?/gi,                 // P/E ratios
  /\bEV\/EBITDA\s*[:=]?\s*\d+(?:\.\d+)?x?/gi,           // EV/EBITDA
  /\bP\/[SB]\s*[:=]?\s*\d+(?:\.\d+)?x?/gi,              // P/S, P/B
  /\b(?:probability|chance|likely)\s+[:=]?\s*\d+(?:\.\d+)?%/gi, // probabilities
  /\bWACC\s*[:=]\s*\d+(?:\.\d+)?%/gi,                   // WACC
  /\bROIC?\s*[:=]\s*\d+(?:\.\d+)?%/gi,                  // ROIC
];

/**
 * Extract up to `maxFacts` unique key numeric facts from a text snippet.
 * Returns them as a compact comma-separated string, or '' when none found.
 */
export function extractKeyFacts(text: string, maxFacts = 10): string {
  const seen = new Set<string>();
  const facts: string[] = [];
  for (const re of FACT_PATTERNS) {
    const pattern = new RegExp(re.source, re.flags);
    for (const m of text.matchAll(pattern)) {
      const key = m[0].toLowerCase().replace(/\s+/g, ' ').trim();
      if (!seen.has(key) && facts.length < maxFacts) {
        seen.add(key);
        facts.push(m[0].trim());
      }
    }
  }
  return facts.join(', ');
}
/** Maps raw JSON field names found in financial tool results to compact labels. */
const METRIC_KEY_MAP: Readonly<Record<string, string>> = {
  revenue: 'rev',
  total_revenue: 'rev',
  net_income: 'NI',
  earnings_per_share: 'EPS',
  eps: 'EPS',
  pe_ratio: 'PE',
  price_to_earnings_ratio: 'PE',
  ev_to_ebitda: 'EV/EBITDA',
  enterprise_value_over_ebitda: 'EV/EBITDA',
  market_cap: 'mktcap',
  market_capitalization: 'mktcap',
  gross_margin: 'GM%',
  operating_margin: 'OpM%',
  price_to_book: 'P/B',
  return_on_equity: 'ROE%',
  return_on_assets: 'ROA%',
  debt_to_equity: 'D/E',
};

/**
 * Parse key financial metrics from a JSON-like tool result snippet.
 * Returns compact `label=value` strings (up to 6) for ticker table rows.
 */
export function extractTickerMetrics(text: string): string[] {
  const metrics: string[] = [];
  const seen = new Set<string>();
  const kvPattern = /"([\w_]+)":\s*"?([^",\n\]}{]+)"?/g;
  for (const m of text.matchAll(kvPattern)) {
    const label = METRIC_KEY_MAP[m[1]!.toLowerCase()];
    if (label) {
      const val = m[2]!.trim().replace(/,$/, '');
      const entry = `${label}=${val}`;
      if (!seen.has(entry) && metrics.length < 6) {
        seen.add(entry);
        metrics.push(entry);
      }
    }
  }
  return metrics;
}

/**
 * Build a merged context summary string from tool results about to be cleared.
 *
 * - Prefixes each line with the tool's ticker/query arg when present so the
 *   LLM retains the ticker→value association (e.g. `get_financials(ticker=NVDA): …`).
 * - Appends a compact ticker→metric table when financial key/value pairs are found.
 * - Snippet length is 400 chars (up from the previous 200) for richer context.
 * - When `existingSummary` is provided the new facts are merged into it instead
 *   of appending a separate entry, preventing 3+ summary blocks stacking up.
 *
 * Returns null when there is nothing to summarise.
 */
export function buildContextSummaryText(
  toSummarise: Array<{ toolName: string; args: Record<string, unknown>; snippet: string }>,
  existingSummary: string | null,
): string | null {
  if (toSummarise.length === 0) return null;

  const lines: string[] = [];
  const tickerRows = new Map<string, string[]>();

  for (const { toolName, args, snippet } of toSummarise) {
    const ticker = typeof args['ticker'] === 'string' ? args['ticker'].toUpperCase() : null;
    const queryArg = typeof args['query'] === 'string' ? args['query'] : null;

    const argsStr = Object.entries(args).map(([k, v]) => `${k}=${v}`).join(', ');
    const condensed = snippet.replace(/\s+/g, ' ').trim().slice(0, 400);
    const keyFacts = extractKeyFacts(snippet);
    const factsNote = keyFacts ? ` [KEY FACTS: ${keyFacts}]` : '';

    // Prefix with ticker/query so the LLM knows which asset the data belongs to.
    const callLabel = ticker
      ? `${toolName}(ticker=${ticker})`
      : queryArg
        ? `${toolName}(query=${queryArg})`
        : `${toolName}(${argsStr})`;
    lines.push(`- ${callLabel}: ${condensed}…${factsNote}`);

    if (ticker) {
      const metrics = extractTickerMetrics(snippet);
      if (metrics.length > 0 && !tickerRows.has(ticker)) {
        tickerRows.set(ticker, metrics);
      }
    }
  }

  let newSummary = `The following ${toSummarise.length} earlier tool result(s) were condensed to save context:\n${lines.join('\n')}`;

  if (tickerRows.size > 0) {
    const tableLines = [...tickerRows.entries()].map(([t, m]) => `${t}: ${m.join(', ')}`);
    newSummary += `\n\nKey metrics by ticker:\n${tableLines.join('\n')}`;
  }

  // Merge into the existing summary rather than appending a second block.
  if (existingSummary) {
    return `${existingSummary}\n\n---\n${newSummary}`;
  }
  return newSummary;
}
