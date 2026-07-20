export interface NormalizedPolymarketOutcome {
  label: string;
  probability: number;
  tokenId?: string;
}

export interface NormalizedPolymarketMarket {
  marketId: string;
  question: string;
  outcomes: {
    yes?: NormalizedPolymarketOutcome;
    no?: NormalizedPolymarketOutcome;
  };
  primaryYesTokenId?: string;
  endDate: string | null;
  volume24h: number;
  liquidity: number;
  ageDays: number | undefined;
  active: boolean;
  closed: boolean;
  enableOrderBook?: boolean;
  description?: string;
}

export interface GammaDateFilter {
  end_date_min: string;
  end_date_max: string;
}

export interface PolymarketSearchInput {
  query: string;
  limit: number;
  tagSlugs?: string[];
  endDateFilter?: GammaDateFilter;
  pageSize?: number;
  maxPages?: number;
}

export interface PolymarketAnchorSearchInput extends PolymarketSearchInput {
  ticker?: string;
  horizonDays?: number;
  enrichMicrostructure?: boolean;
}

export interface PolymarketSearchProvenance {
  source: 'gamma-http' | 'sdk' | 'proxy';
  pagesRead: number;
  tagSlugsTried: string[];
  usedEndDateFilter: boolean;
}

export interface PolymarketSearchResult {
  markets: NormalizedPolymarketMarket[];
  warnings: string[];
  provenance: PolymarketSearchProvenance;
}

export interface PolymarketReadDriver {
  searchEvents(input: PolymarketSearchInput): Promise<PolymarketSearchResult>;
  fetchSpread(tokenId: string): Promise<number | null>;
  fetchPriceHistory(
    tokenId: string,
    interval: '1h' | '6h' | '1d',
  ): Promise<Array<{ tSec: number; p: number }>>;
}
