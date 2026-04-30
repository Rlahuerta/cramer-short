import { afterEach, describe, expect, it } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  appendArbiterReplayBundle,
  appendReplayCacheBundle,
  appendReplayCachePolymarketCapture,
  appendReplayCacheWhaleCapture,
  appendRawPolymarketReplayRow,
  appendRawWhaleReplayRow,
  createArbiterReplayBundleFromArbitratorInput,
  createRawPolymarketReplayRow,
  createRawWhaleReplayRowFromToolResult,
  freezePolymarketReplayBlock,
  getArbiterReplayCachePaths,
  normalizeWhaleReplayRow,
  parseArbiterReplayBundleLine,
  parseRawPolymarketReplayLine,
  parseRawWhaleReplayLine,
  readArbiterReplayBundles,
  readRawPolymarketReplayRows,
  readRawWhaleReplayRows,
  toForecastArbiterInput,
  type ArbiterReplayBundle,
  type RawPolymarketReplayRow,
  type RawWhaleReplayRow,
} from './arbiter-replay.js';

const tempDirs: string[] = [];

function makeTempFile(name: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'arbiter-replay-'));
  tempDirs.push(dir);
  return join(dir, name);
}

const validBundle: ArbiterReplayBundle = {
  capturedAt: '2026-04-30T12:00:00.000Z',
  ticker: 'BTC',
  horizonDays: 7,
  currentPrice: 68000,
  leverage: 3,
  markov: {
    forecast_return: 0.032,
    p_up: 0.61,
    confidence: 0.58,
    structural_break: false,
    flat_probability: 0.18,
    ci_low: 65000,
    ci_high: 71500,
    summary: 'Markov prior is mildly bullish.',
  },
  polymarket: {
    querySet: ['bitcoin price', 'btc price'],
    selectedMarketIds: ['pm-1'],
    selectedMarkets: [
      {
        marketId: 'pm-1',
        assetId: 'asset-yes-1',
        question: 'Will Bitcoin be above $70,000 on May 7?',
        probability: 0.54,
        volume24h: 250000,
        endDate: '2026-05-07T00:00:00.000Z',
        semantics: 'terminal',
        extractedPriceLevels: [70000],
        relevanceScore: 0.92,
        bid: 0.53,
        ask: 0.55,
      },
    ],
    summary: 'Selected terminal BTC threshold market.',
    confidence: 0.63,
    forecastReturn: 0.014,
    qualityScore: 72,
    qualityGrade: 'B',
    semanticParserVersion: 'v1',
    warnings: ['history is thin'],
  },
  whale: {
    source: 'whale-alert',
    direction: 'long',
    confidence: 0.66,
    summary: 'Large BTC inflows into accumulation wallets.',
    observationWindowStart: '2026-04-30T10:00:00.000Z',
    observationWindowEnd: '2026-04-30T12:00:00.000Z',
    txCount: 2,
    notionalUsd: 12000000,
    txHashes: ['0xabc', '0xdef'],
  },
  warnings: ['bundle built from capture job'],
};

afterEach(() => {
  for (const dir of tempDirs.splice(0, tempDirs.length)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe('arbiter replay raw row parsers', () => {
  it('parses valid raw Polymarket replay rows', () => {
    const row: RawPolymarketReplayRow = {
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      horizonDays: 7,
      currentPrice: 68000,
      querySet: ['bitcoin price'],
      selectedMarketIds: ['pm-1'],
      candidates: [
        {
          marketId: 'pm-1',
          assetId: 'asset-yes-1',
          question: 'Will Bitcoin be above $70,000 on May 7?',
          probability: 0.54,
          volume24h: 250000,
          createdAt: '2026-04-29T00:00:00.000Z',
          endDate: '2026-05-07T00:00:00.000Z',
        },
      ],
      warnings: ['thin history'],
    };

    expect(parseRawPolymarketReplayLine(JSON.stringify(row))).toEqual(row);
  });

  it('parses valid raw whale replay rows', () => {
    const row: RawWhaleReplayRow = {
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      source: 'whale-alert',
      observationWindowStart: '2026-04-30T10:00:00.000Z',
      observationWindowEnd: '2026-04-30T12:00:00.000Z',
      transactions: [
        {
          hash: '0xabc',
          timestamp: '2026-04-30T11:30:00.000Z',
          symbol: 'BTC',
          valueUsd: 5000000,
          fromOwner: 'unknown wallet',
          toOwner: 'accumulation wallet',
        },
      ],
      warnings: ['single-source capture'],
    };

    expect(parseRawWhaleReplayLine(JSON.stringify(row))).toEqual(row);
  });
});

describe('ArbiterReplayBundle persistence', () => {
  it('round-trips raw rows and bundles through JSONL helpers', () => {
    const polymarketPath = makeTempFile('polymarket.jsonl');
    const whalePath = makeTempFile('whale.jsonl');
    const bundlePath = makeTempFile('bundle.jsonl');

    appendRawPolymarketReplayRow({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      horizonDays: 7,
      currentPrice: 68000,
      querySet: ['bitcoin price'],
      selectedMarketIds: ['pm-1'],
      candidates: [
        {
          marketId: 'pm-1',
          assetId: 'asset-yes-1',
          question: 'Will Bitcoin be above $70,000 on May 7?',
          probability: 0.54,
          volume24h: 250000,
        },
      ],
    }, polymarketPath);
    appendRawWhaleReplayRow({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      source: 'whale-alert',
      observationWindowStart: '2026-04-30T10:00:00.000Z',
      observationWindowEnd: '2026-04-30T12:00:00.000Z',
      transactions: [
        {
          hash: '0xabc',
          timestamp: '2026-04-30T11:30:00.000Z',
        },
      ],
    }, whalePath);
    appendArbiterReplayBundle(validBundle, bundlePath);

    expect(readRawPolymarketReplayRows(polymarketPath)).toHaveLength(1);
    expect(readRawWhaleReplayRows(whalePath)).toHaveLength(1);
    expect(readArbiterReplayBundles(bundlePath)).toEqual([validBundle]);
  });

  it('rejects malformed bundles that omit required replay fields', () => {
    const malformed = {
      ...validBundle,
      polymarket: {
        ...validBundle.polymarket,
        selectedMarkets: [
          {
            marketId: 'pm-1',
            question: 'Will Bitcoin be above $70,000 on May 7?',
            probability: 0.54,
            volume24h: 250000,
            endDate: '2026-05-07T00:00:00.000Z',
            semantics: 'terminal',
            extractedPriceLevels: [70000],
          },
        ],
      },
    };

    expect(parseArbiterReplayBundleLine(JSON.stringify(malformed))).toBeNull();
    expect(parseArbiterReplayBundleLine('{"capturedAt":"not-a-date"}')).toBeNull();
  });

  it('parses semantic label arrays on replay bundles', () => {
    const labeledBundle: ArbiterReplayBundle = {
      ...validBundle,
      labels: {
        forecast: {
          realizedPrice: 70500,
          realizedReturn: (70500 - 68000) / 68000,
          actualBinary: 1,
          labeledAt: '2026-05-08T12:00:00.000Z',
        },
        semantic: [
          {
            marketId: 'pm-1',
            semantics: 'terminal',
            outcome: 'yes',
            labeledAt: '2026-05-08T12:00:00.000Z',
          },
        ],
      },
    };

    expect(parseArbiterReplayBundleLine(JSON.stringify(labeledBundle))).toEqual(labeledBundle);
  });

  it('appends auto-captured replay records into the local cache layout', () => {
    const dir = mkdtempSync(join(tmpdir(), 'arbiter-replay-cache-'));
    tempDirs.push(dir);
    const paths = getArbiterReplayCachePaths({
      bundlePath: join(dir, 'bundles.jsonl'),
      polymarketRawPath: join(dir, 'polymarket-raw.jsonl'),
      whaleRawPath: join(dir, 'whale-raw.jsonl'),
    });

    appendReplayCachePolymarketCapture({
      rawRow: {
        capturedAt: '2026-04-30T12:00:00.000Z',
        ticker: 'BTC',
        horizonDays: 7,
        currentPrice: 68000,
        querySet: ['bitcoin price'],
        selectedMarketIds: ['pm-1'],
        candidates: [
          {
            marketId: 'pm-1',
            assetId: 'asset-yes-1',
            question: 'Will Bitcoin be above $70,000 on May 7?',
            probability: 0.54,
            volume24h: 250000,
          },
        ],
      },
      polymarket: validBundle.polymarket!,
    }, paths);
    appendReplayCacheWhaleCapture({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      source: 'whale-alert',
      observationWindowStart: '2026-04-30T10:00:00.000Z',
      observationWindowEnd: '2026-04-30T12:00:00.000Z',
      transactions: [
        {
          hash: '0xabc',
          timestamp: '2026-04-30T11:30:00.000Z',
        },
      ],
    }, paths);
    appendReplayCacheBundle(validBundle, paths);

    expect(readRawPolymarketReplayRows(paths.polymarketRawPath)).toHaveLength(1);
    expect(readRawWhaleReplayRows(paths.whaleRawPath)).toHaveLength(1);
    expect(readArbiterReplayBundles(paths.bundlePath)).toHaveLength(2);
  });
});

describe('Phase 2 Polymarket replay helpers', () => {
  it('builds a raw Polymarket replay row from selected decision-time markets', () => {
    const row = createRawPolymarketReplayRow({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      horizonDays: 7,
      currentPrice: 68000,
      querySet: ['bitcoin price', 'bitcoin price', 'btc'],
      selectedMarkets: [
        {
          marketId: 'pm-1',
          assetId: 'asset-yes-1',
          question: 'Will Bitcoin be above $70,000 on May 7?',
          probability: 0.54,
          volume24h: 250000,
          endDate: '2026-05-07T00:00:00.000Z',
          active: true,
          closed: false,
          enableOrderBook: true,
        },
      ],
      warnings: ['thin history'],
    });

    expect(row).toEqual({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      horizonDays: 7,
      currentPrice: 68000,
      querySet: ['bitcoin price', 'btc'],
      selectedMarketIds: ['pm-1'],
      candidates: [
        {
          marketId: 'pm-1',
          assetId: 'asset-yes-1',
          question: 'Will Bitcoin be above $70,000 on May 7?',
          probability: 0.54,
          volume24h: 250000,
          endDate: '2026-05-07T00:00:00.000Z',
          active: true,
          closed: false,
          enableOrderBook: true,
        },
      ],
      warnings: ['thin history'],
    });
  });

  it('freezes Polymarket semantics and extracted price levels at capture time', () => {
    const block = freezePolymarketReplayBlock({
      querySet: ['bitcoin price', 'btc'],
      selectedMarkets: [
        {
          marketId: 'pm-1',
          assetId: 'asset-yes-1',
          question: 'Will Bitcoin be above $70,000 on May 7?',
          probability: 0.54,
          volume24h: 250000,
          endDate: '2026-05-07T00:00:00.000Z',
          relevanceScore: 0.91,
        },
      ],
      warnings: ['thin history'],
      summary: 'Selected terminal BTC threshold market.',
      confidence: 0.63,
      forecastReturn: 0.014,
      qualityScore: 72,
      qualityGrade: 'B',
    });

    expect(block.selectedMarketIds).toEqual(['pm-1']);
    expect(block.selectedMarkets).toEqual([
      {
        marketId: 'pm-1',
        assetId: 'asset-yes-1',
        question: 'Will Bitcoin be above $70,000 on May 7?',
        probability: 0.54,
        volume24h: 250000,
        endDate: '2026-05-07T00:00:00.000Z',
        semantics: 'terminal',
        extractedPriceLevels: [70000],
        relevanceScore: 0.91,
      },
    ]);
    expect(block.semanticParserVersion).toBe('forecast-arbitrator:classifyPolymarketQuestion');
    expect(block.warnings).toEqual(['thin history']);
  });

  it('drops markets without CLOB token ids from the frozen replay block and keeps an explicit warning', () => {
    const block = freezePolymarketReplayBlock({
      querySet: ['bitcoin price'],
      selectedMarkets: [
        {
          marketId: 'pm-1',
          question: 'Will Bitcoin be above $70,000 on May 7?',
          probability: 0.54,
          volume24h: 250000,
          endDate: '2026-05-07T00:00:00.000Z',
        },
      ],
    });

    expect(block.selectedMarketIds).toEqual([]);
    expect(block.selectedMarkets).toEqual([]);
    expect(block.warnings).toEqual(['Missing CLOB token id for Polymarket market pm-1']);
  });
});

describe('Phase 3 whale replay normalization', () => {
  it('normalizes exchange outflows into a long whale signal', () => {
    const whale = normalizeWhaleReplayRow({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      source: 'whale-alert',
      observationWindowStart: '2026-04-30T10:00:00.000Z',
      observationWindowEnd: '2026-04-30T12:00:00.000Z',
      transactions: [
        {
          hash: '0xabc',
          timestamp: '2026-04-30T11:00:00.000Z',
          symbol: 'BTC',
          valueUsd: 8_000_000,
          fromOwner: 'Binance exchange',
          toOwner: 'accumulation wallet',
        },
      ],
    });

    expect(whale).toMatchObject({
      source: 'whale-alert',
      direction: 'long',
      observationWindowStart: '2026-04-30T10:00:00.000Z',
      observationWindowEnd: '2026-04-30T12:00:00.000Z',
      txCount: 1,
      notionalUsd: 8_000_000,
      txHashes: ['0xabc'],
    });
    expect(whale.confidence).toBeGreaterThan(0.7);
    expect(whale.summary).toContain('bullish');
  });

  it('normalizes exchange inflows into a short whale signal', () => {
    const whale = normalizeWhaleReplayRow({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      source: 'whale-alert',
      observationWindowStart: '2026-04-30T10:00:00.000Z',
      observationWindowEnd: '2026-04-30T12:00:00.000Z',
      transactions: [
        {
          hash: '0xdef',
          timestamp: '2026-04-30T11:15:00.000Z',
          symbol: 'BTC',
          valueUsd: 6_500_000,
          fromOwner: 'unknown wallet',
          toOwner: 'Coinbase exchange',
        },
      ],
    });

    expect(whale.direction).toBe('short');
    expect(whale.confidence).toBeGreaterThan(0.7);
    expect(whale.summary).toContain('bearish');
  });

  it('collapses ambiguous owner-neutral whale flow to a neutral summary', () => {
    const whale = normalizeWhaleReplayRow({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      source: 'whale-alert',
      observationWindowStart: '2026-04-30T10:00:00.000Z',
      observationWindowEnd: '2026-04-30T12:00:00.000Z',
      transactions: [
        {
          hash: '0xghi',
          timestamp: '2026-04-30T11:20:00.000Z',
          symbol: 'BTC',
          valueUsd: 3_000_000,
          fromOwner: 'unknown wallet',
          toOwner: 'unknown wallet',
        },
      ],
    });

    expect(whale.direction).toBe('neutral');
    expect(whale.confidence).toBe(0.35);
    expect(whale.summary).toContain('neutral');
  });

  it('builds raw whale replay rows from whale tool outputs', () => {
    const rawRow = createRawWhaleReplayRowFromToolResult({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      whale: {
        source: 'blockchain.info-mempool',
        recent_large_transactions: [
          {
            hash: '0xabc',
            time: 1_777_548_000,
            usd_value: 5000000,
          },
        ],
      },
    });

    expect(rawRow).toEqual({
      capturedAt: '2026-04-30T12:00:00.000Z',
      ticker: 'BTC',
      source: 'blockchain.info-mempool',
      observationWindowStart: '2026-04-30T11:20:00.000Z',
      observationWindowEnd: '2026-04-30T11:20:00.000Z',
      transactions: [
        {
          hash: '0xabc',
          timestamp: '2026-04-30T11:20:00.000Z',
          symbol: 'BTC',
          valueUsd: 5000000,
        },
      ],
    });
  });
});

describe('arbiter input capture helpers', () => {
  it('builds replay bundles directly from arbiter tool input payloads', () => {
    const bundle = createArbiterReplayBundleFromArbitratorInput({
      capturedAt: '2026-04-30T12:00:00.000Z',
      input: {
        ticker: 'btc',
        horizon_days: 7,
        current_price: 68000,
        leverage: 3,
        markov: validBundle.markov,
        polymarket: {
          forecast_return: 0.014,
          confidence: 0.63,
          quality_score: 72,
          quality_grade: 'B',
          querySet: ['bitcoin price'],
          markets: [
            {
              marketId: 'pm-1',
              assetId: 'asset-yes-1',
              question: 'Will Bitcoin be above $70,000 on May 7?',
              probability: 0.54,
              semantics: 'terminal',
              price: 70000,
              volume24h: 250000,
              endDate: '2026-05-07T00:00:00.000Z',
            },
          ],
          summary: 'Selected terminal BTC threshold market.',
        },
        whale: {
          direction: 'long',
          confidence: 0.66,
          summary: 'Large BTC inflows into accumulation wallets.',
          source: 'whale-alert',
          observationWindowStart: '2026-04-30T10:00:00.000Z',
          observationWindowEnd: '2026-04-30T12:00:00.000Z',
          txCount: 2,
          notionalUsd: 12000000,
          txHashes: ['0xabc', '0xdef'],
        },
      },
    });

    expect(bundle.ticker).toBe('BTC');
    expect(bundle.polymarket?.selectedMarketIds).toEqual(['pm-1']);
    expect(bundle.whale).toMatchObject({
      source: 'whale-alert',
      direction: 'long',
      txCount: 2,
    });
  });
});

describe('toForecastArbiterInput', () => {
  it('maps bundles into deterministic arbiter inputs', () => {
    const first = toForecastArbiterInput(validBundle);
    const second = toForecastArbiterInput(validBundle);

    expect(first).toEqual(second);
    expect(first).toEqual({
      ticker: 'BTC',
      horizon_days: 7,
      current_price: 68000,
      leverage: 3,
      markov: validBundle.markov,
      polymarket: {
        forecast_return: 0.014,
        confidence: 0.63,
        quality_score: 72,
        quality_grade: 'B',
        summary: 'Selected terminal BTC threshold market.',
        markets: [
          {
            question: 'Will Bitcoin be above $70,000 on May 7?',
            probability: 0.54,
            semantics: 'terminal',
            price: 70000,
          },
        ],
      },
      whale: {
        direction: 'long',
        confidence: 0.66,
        summary: 'Large BTC inflows into accumulation wallets.',
      },
    });
  });
});
