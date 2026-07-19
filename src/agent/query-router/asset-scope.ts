import { extractTickers } from '../../memory/ticker-extractor.js';

/**
 * Conservative multi-asset detector for the forecast-tool ticker normalizer.
 *
 * Returns true when a query references more than one distinct tradable asset,
 * so the normalizer can avoid clobbering a legitimate comparison (e.g.
 * "compare BTC and ETH"). Tokens that appear only inside an exclusion clause
 * ("no GOLD, ETH, SOL context") are ignored, and an asset and its ETF proxy
 * (GOLD/GLD, SILVER/SLV, OIL/USO) count as the same asset.
 */

// Strips exclusion clauses up to the next sentence boundary so excluded assets
// ("no GOLD, ETH…") don't inflate the distinct-asset count.
const EXCLUSION_CLAUSE_RE =
  /\b(?:no|not|without|avoid(?:ing)?|exclude|excluding|ignore|ignoring|skip(?:ping)?|except)\b[^.!?\n]*/gi;

const ASSET_KEYWORD_ALIASES: ReadonlyArray<readonly [RegExp, string]> = [
  [/\b(?:bitcoin|btc)\b/i, 'BTC'],
  [/\b(?:ethereum|eth)\b/i, 'ETH'],
  [/\b(?:solana|sol)\b/i, 'SOL'],
  [/\b(?:gold|gld|xauusd)\b/i, 'GOLD'],
  [/\b(?:silver|slv|xagusd)\b/i, 'SILVER'],
  [/\b(?:oil|crude|wti|uso|brent)\b/i, 'OIL'],
];

const TICKER_ROOT_ALIASES: Record<string, string> = {
  BITCOIN: 'BTC',
  ETHEREUM: 'ETH',
  SOLANA: 'SOL',
  GLD: 'GOLD',
  IAU: 'GOLD',
  XAUUSD: 'GOLD',
  SLV: 'SILVER',
  XAGUSD: 'SILVER',
  USO: 'OIL',
  CRUDE: 'OIL',
  WTI: 'OIL',
};

function canonicalizeAssetRoot(token: string): string {
  const upper = token.toUpperCase().replace(/-USD$/, '').replace(/(?:USDT|USDC|USD)$/, '');
  return TICKER_ROOT_ALIASES[upper] ?? upper;
}

export function collectDistinctAssetRoots(query: string): Set<string> {
  const stripped = query.replace(EXCLUSION_CLAUSE_RE, ' ');
  const roots = new Set<string>();

  for (const [pattern, canonical] of ASSET_KEYWORD_ALIASES) {
    if (pattern.test(stripped)) roots.add(canonical);
  }

  for (const ticker of extractTickers(stripped)) {
    roots.add(canonicalizeAssetRoot(ticker));
  }

  return roots;
}

export function referencesMultipleDistinctAssets(query: string): boolean {
  return collectDistinctAssetRoots(query).size > 1;
}
