export type ResolvedAssetClass = 'commodity_gold' | 'commodity_silver' | 'gold_miner' | 'ticker';

export interface ResolvedAssetIntent {
  rawQuery: string;
  rawTicker: string | null;
  resolvedTicker: string | null;
  assetClass: ResolvedAssetClass | null;
  displayName: string | null;
  proxyLabel: string | null;
  needsClarification: boolean;
}

export interface ResolvedTickerSearchIdentity {
  rawTicker: string;
  canonicalTicker: string;
  searchQuery: string;
  canonicalNames: string[];
  strictQuestionMatch: boolean;
}

const BARRICK_CONTEXT_RE = /\bbarrick\b|\bgold\s+(?:stock|equity|shares|company|earnings|revenue|miner|mining)\b|\$gold\b/i;
const GOLD_COMMODITY_RE = /\bgold\b|\bxauusd\b/i;
const SILVER_COMMODITY_RE = /\bsilver\b|\bxagusd\b/i;
const GOLD_PROXY_TICKERS = new Set(['GLD', 'IAU', 'SGOL', 'XAUUSD']);
const SILVER_PROXY_TICKERS = new Set(['SLV', 'SIVR', 'XAGUSD', 'SILVER']);

function normalizeExplicitTicker(explicitTicker?: string | null): string | null {
  const value = explicitTicker?.trim().toUpperCase();
  return value && value.length > 0 ? value : null;
}

export function resolveTickerSearchIdentity(ticker: string): ResolvedTickerSearchIdentity {
  const normalized = normalizeExplicitTicker(ticker) ?? ticker.trim().toUpperCase();
  const bareTicker = normalized.replace(/-USD$/, '');

  if (GOLD_PROXY_TICKERS.has(bareTicker)) {
    return {
      rawTicker: normalized,
      canonicalTicker: 'GLD',
      searchQuery: 'gold',
      canonicalNames: ['gold', 'gld'],
      strictQuestionMatch: false,
    };
  }

  if (SILVER_PROXY_TICKERS.has(bareTicker)) {
    return {
      rawTicker: normalized,
      canonicalTicker: 'SLV',
      searchQuery: 'silver',
      canonicalNames: ['silver', 'slv'],
      strictQuestionMatch: false,
    };
  }

  if (bareTicker === 'GOLD') {
    return {
      rawTicker: normalized,
      canonicalTicker: 'GOLD',
      searchQuery: 'Barrick Gold',
      canonicalNames: ['barrick gold', 'barrick'],
      strictQuestionMatch: true,
    };
  }

  return {
    rawTicker: normalized,
    canonicalTicker: bareTicker,
    searchQuery: bareTicker,
    canonicalNames: [bareTicker.toLowerCase()],
    strictQuestionMatch: false,
  };
}

export function resolveAssetIntent(query: string, explicitTicker?: string | null): ResolvedAssetIntent {
  const normalizedTicker = normalizeExplicitTicker(explicitTicker);

  if (normalizedTicker === 'GLD') {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'GLD',
      assetClass: 'commodity_gold',
      displayName: 'Gold (GLD proxy)',
      proxyLabel: 'GLD',
      needsClarification: false,
    };
  }

  if (normalizedTicker === 'SLV') {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'SLV',
      assetClass: 'commodity_silver',
      displayName: 'Silver (SLV proxy)',
      proxyLabel: 'SLV',
      needsClarification: false,
    };
  }

  const hasBarrickContext = BARRICK_CONTEXT_RE.test(query);
  if (hasBarrickContext) {
    if (normalizedTicker && normalizedTicker !== 'GOLD') {
      return {
        rawQuery: query,
        rawTicker: normalizedTicker,
        resolvedTicker: normalizedTicker,
        assetClass: 'ticker',
        displayName: normalizedTicker,
        proxyLabel: null,
        needsClarification: false,
      };
    }

    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'GOLD',
      assetClass: 'gold_miner',
      displayName: 'Barrick Gold',
      proxyLabel: null,
      needsClarification: false,
    };
  }

  if ((normalizedTicker === 'GOLD' && !BARRICK_CONTEXT_RE.test(query)) || GOLD_COMMODITY_RE.test(query)) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'GLD',
      assetClass: 'commodity_gold',
      displayName: 'Gold (GLD proxy)',
      proxyLabel: 'GLD',
      needsClarification: false,
    };
  }

  if (normalizedTicker === 'SILVER' || normalizedTicker === 'XAGUSD' || SILVER_COMMODITY_RE.test(query)) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'SLV',
      assetClass: 'commodity_silver',
      displayName: 'Silver (SLV proxy)',
      proxyLabel: 'SLV',
      needsClarification: false,
    };
  }

  if (normalizedTicker) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: normalizedTicker,
      assetClass: 'ticker',
      displayName: normalizedTicker,
      proxyLabel: null,
      needsClarification: false,
    };
  }

  return {
    rawQuery: query,
    rawTicker: null,
    resolvedTicker: null,
    assetClass: null,
    displayName: null,
    proxyLabel: null,
    needsClarification: false,
  };
}

export function assertAssetConsistency(
  intent: ResolvedAssetIntent,
  toolName: string,
  ticker: string,
): void {
  const normalized = ticker.trim().toUpperCase();

  if (intent.assetClass === 'commodity_gold' && normalized === 'GOLD' && toolName !== 'get_stock_tickers') {
    throw new Error('Commodity gold intent cannot use Barrick GOLD ticker directly; use GLD proxy instead.');
  }

  if (intent.assetClass === 'gold_miner' && normalized === 'GLD') {
    throw new Error('Barrick gold equity intent cannot use GLD commodity proxy.');
  }

  if (intent.assetClass === 'commodity_silver' && normalized === 'SILVER' && toolName !== 'get_stock_tickers') {
    throw new Error('Commodity silver intent cannot use SILVER pseudo-ticker directly; use SLV proxy instead.');
  }
}
