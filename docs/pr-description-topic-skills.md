# Smarter Oil/Commodity Polymarket Discovery + Code Review Fixes

## Problem

When users asked for oil price forecasts, the agent resolved `OIL` -> `USO` (commodity proxy) but searched Polymarket using `"USO"` and `"commodity price"` -- phrases that don't appear in actual Polymarket oil markets (*"Will WTI hit $100?"*, *"Will OPEC cut production?"*). Additionally, the generic `commodity` signal map had no oil-specific signals (OPEC, supply, inventory), and the monolithic commodity keyword bucket allowed cross-contamination between oil and gold markets.

## Changes

### Oil Signal Discovery (`src/tools/finance/signal-extractor.ts`)
- Added dedicated `oil` signal type with 5 weighted signals: Price Level, OPEC/Supply, Geopolitical, Fed Rate Decision, US Recession
- `extractSignals()` now appends canonical commodity names (e.g. `USO` -> `oil`, `crude`, `WTI`) as extra query variants
- `resolveSignalType()` routes `commodity_oil` -> `oil` instead of generic `commodity`
- Added `oil_supply` keyword category to `SIGNAL_KEYWORDS`
- Added `USO` and `UNG` to `TICKER_TO_COMPANY_NAME` and `SECTOR_MAP`

### Impact Mapping (`src/tools/finance/impact-map.ts`)
- Added `oil_supply` impact category with sector-specific deltas for energy, airline, consumer, materials

### Tool Description (`src/tools/finance/polymarket-forecast.ts`)
- Updated `POLYMARKET_FORECAST_DESCRIPTION` to explain commodity-specific search behavior (e.g. searches both ETF ticker and underlying commodity names)

### Code Review Fixes (`markov-distribution.ts`, `polymarket-forecast.ts`, `cli.ts`)
- **Issue 1**: `DATE_ANCHORED_TRADE_PATTERN` regex now matches ISO dates (`2025-05-01`) and day-first formats (`1 May 2025`) -- previously silently dropped these markets
- **Issue 2**: BTC 14d horizon now uses retry queries on empty initial results (was asymmetric: only long-horizon crypto got retries)
- **Issue 3**: Hoisted `marketReader` closure outside the hot market loop in `polymarket-forecast.ts`
- **Issue 4**: Simplified empty `if (persistenceSnapshot) {}` block
- **Issue 5**: Removed dead `return` before `process.exit(0)` in `cli.ts`

### Tests
- Added 8 oil signal routing tests (weights, search phrases, query variants, category order)
- Added 3 date-anchor regex tests (ISO, slashes, day-first)
- Added `normalizeHistoricalPriceTicker` tests (OIL/CRUDE/WTICOUSD -> USO)

## Verification

```bash
bun run typecheck              # clean
bun test src                   # 4,249 pass, 0 fail, 59 skip
```

## Files Changed

| File | What |
|------|------|
| `src/tools/finance/signal-extractor.ts` | Oil signal type, canonical name variants, `USO`/`UNG` mappings |
| `src/tools/finance/impact-map.ts` | `oil_supply` impact category |
| `src/tools/finance/polymarket-forecast.ts` | Tool description, hoisted closure, empty if cleanup |
| `src/tools/finance/markov-distribution.ts` | Regex fix, BTC 14d retry fix |
| `src/tools/finance/signal-extractor.test.ts` | Oil routing tests |
| `src/tools/finance/markov-distribution.test.ts` | Date anchor + normalizeHistoricalPriceTicker tests |
| `src/cli.ts` | Stale comment cleanup, dead code removal |
