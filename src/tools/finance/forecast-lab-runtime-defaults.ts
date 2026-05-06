import { AsyncLocalStorage } from 'node:async_hooks';
import { resolveTickerSearchIdentity } from './asset-resolver.js';

export type ForecastLabRuntimeAssetScope = 'shared' | 'btc' | 'gold';

export type ForecastLabRuntimeScalarValue = boolean | number;

type ForecastLabRuntimeDefaultsRecord = Record<string, ForecastLabRuntimeScalarValue>;

const runtimeAssetScopeStorage = new AsyncLocalStorage<ForecastLabRuntimeAssetScope | undefined>();

export function resolveForecastLabRuntimeAssetScopeForTicker(
  ticker: string,
): ForecastLabRuntimeAssetScope | undefined {
  const normalizedTicker = ticker.trim().toUpperCase();
  if (normalizedTicker === 'BTC' || normalizedTicker === 'BTC-USD') {
    return 'btc';
  }

  return resolveTickerSearchIdentity(ticker).canonicalTicker === 'GLD' ? 'gold' : 'shared';
}

function getRuntimeScopeResolutionOrder(
  assetScope: ForecastLabRuntimeAssetScope | undefined,
): readonly ForecastLabRuntimeAssetScope[] {
  if (!assetScope) {
    return [];
  }

  if (assetScope === 'gold') {
    return ['shared', 'gold'];
  }

  return [assetScope];
}

export function withForecastLabRuntimeAssetScope<T>(
  assetScope: ForecastLabRuntimeAssetScope | undefined,
  callback: () => T,
): T {
  if (!assetScope) {
    return callback();
  }

  return runtimeAssetScopeStorage.run(assetScope, callback);
}

export function getForecastLabRuntimeAssetScope(): ForecastLabRuntimeAssetScope | undefined {
  return runtimeAssetScopeStorage.getStore();
}

export function createForecastLabAssetScopedRuntimeDefaults<
  TDefaults extends ForecastLabRuntimeDefaultsRecord,
>(shippedDefaults: TDefaults) {
  const activeOverrides = new Map<ForecastLabRuntimeAssetScope, Partial<TDefaults>>();

  return {
    resolve(
      assetScope: ForecastLabRuntimeAssetScope | undefined = getForecastLabRuntimeAssetScope(),
      explicitOverrides?: Partial<TDefaults>,
    ): TDefaults {
      const layeredOverrides = getRuntimeScopeResolutionOrder(assetScope).reduce<Partial<TDefaults>>(
        (resolved, scope) => ({
          ...resolved,
          ...activeOverrides.get(scope),
        }),
        {},
      );

      return {
        ...shippedDefaults,
        ...layeredOverrides,
        ...explicitOverrides,
      };
    },
    get(assetScope: ForecastLabRuntimeAssetScope): Partial<TDefaults> | undefined {
      const overrides = activeOverrides.get(assetScope);
      return overrides ? { ...overrides } : undefined;
    },
    set(assetScope: ForecastLabRuntimeAssetScope, overrides?: Partial<TDefaults>): void {
      if (!overrides || Object.keys(overrides).length === 0) {
        activeOverrides.delete(assetScope);
        return;
      }

      activeOverrides.set(assetScope, { ...overrides });
    },
  };
}
