import type {
  PolymarketAnchorSearchInput,
  PolymarketReadDriver,
  PolymarketSearchInput,
  PolymarketSearchResult,
} from './polymarket-types.js';

export interface PolymarketGateway {
  searchMarkets(input: PolymarketSearchInput): Promise<PolymarketSearchResult>;
  searchAnchorMarkets(input: PolymarketAnchorSearchInput): Promise<PolymarketSearchResult>;
}

export function createPolymarketGateway(options: {
  readDriver: PolymarketReadDriver;
}): PolymarketGateway {
  const { readDriver } = options;

  return {
    async searchMarkets(input) {
      return readDriver.searchEvents(input);
    },
    async searchAnchorMarkets(input) {
      return readDriver.searchEvents(input);
    },
  };
}
