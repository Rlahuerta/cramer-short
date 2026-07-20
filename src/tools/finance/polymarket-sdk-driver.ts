import { questionMatchesQuery } from './polymarket-gamma-client.js';
import type {
  NormalizedPolymarketMarket,
  PolymarketReadDriver,
  PolymarketSearchInput,
  PolymarketSearchResult,
} from './polymarket-types.js';

interface SdkOutcome {
  price?: number;
  tokenId?: string;
}

interface SdkMarket {
  id?: string;
  conditionId?: string;
  question: string;
  endDateIso?: string;
  volume24hr?: number;
  liquidityNum?: number;
  active?: boolean;
  closed?: boolean;
  outcomes?: {
    yes?: SdkOutcome;
    no?: SdkOutcome;
  };
}

interface SdkMarketPage {
  items: SdkMarket[];
}

interface SdkLikeClient {
  listMarkets(input?: Record<string, unknown>): AsyncIterable<SdkMarketPage>;
}

function normalizeSdkMarket(market: SdkMarket): NormalizedPolymarketMarket {
  return {
    marketId: market.conditionId ?? market.id ?? market.question,
    question: market.question,
    outcomes: {
      ...(market.outcomes?.yes
        ? {
            yes: {
              label: 'Yes',
              probability: market.outcomes.yes.price ?? 0,
              ...(market.outcomes.yes.tokenId
                ? { tokenId: market.outcomes.yes.tokenId }
                : {}),
            },
          }
        : {}),
      ...(market.outcomes?.no
        ? {
            no: {
              label: 'No',
              probability: market.outcomes.no.price ?? 0,
              ...(market.outcomes.no.tokenId
                ? { tokenId: market.outcomes.no.tokenId }
                : {}),
            },
          }
        : {}),
    },
    ...(market.outcomes?.yes?.tokenId ? { primaryYesTokenId: market.outcomes.yes.tokenId } : {}),
    endDate: market.endDateIso ?? null,
    volume24h: market.volume24hr ?? 0,
    liquidity: market.liquidityNum ?? 0,
    ageDays: undefined,
    active: market.active ?? true,
    closed: market.closed ?? false,
  };
}

export function createSdkReadDriver(client: SdkLikeClient): PolymarketReadDriver {
  return {
    async searchEvents(input: PolymarketSearchInput): Promise<PolymarketSearchResult> {
      const markets: NormalizedPolymarketMarket[] = [];
      let pagesRead = 0;

      for await (const page of client.listMarkets({ closed: false, pageSize: input.limit })) {
        pagesRead += 1;
        for (const item of page.items) {
          if (!questionMatchesQuery(item.question, input.query)) continue;
          markets.push(normalizeSdkMarket(item));
          if (markets.length >= input.limit) break;
        }
        if (markets.length >= input.limit) break;
      }

      return {
        markets,
        warnings: [],
        provenance: {
          source: 'sdk',
          pagesRead,
          tagSlugsTried: [],
          usedEndDateFilter: input.endDateFilter !== undefined,
        },
      };
    },
    async fetchSpread(_tokenId: string) {
      return null;
    },
    async fetchPriceHistory(_tokenId: string, _interval: '1h' | '6h' | '1d') {
      return [];
    },
  };
}
