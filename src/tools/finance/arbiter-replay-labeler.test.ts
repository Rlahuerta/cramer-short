import { describe, expect, it } from 'bun:test';
import type { ArbiterReplayBundle } from './arbiter-replay.js';
import { evaluateReplayLabelEligibility, labelReplayBundle } from './arbiter-replay-labeler.js';

function makeBundle(
  market: NonNullable<ArbiterReplayBundle['polymarket']>['selectedMarkets'][number],
): ArbiterReplayBundle {
  return {
    capturedAt: '2026-05-01T00:00:00.000Z',
    ticker: 'BTC',
    horizonDays: 7,
    currentPrice: 68000,
    polymarket: {
      querySet: ['bitcoin price'],
      selectedMarketIds: [market.marketId],
      selectedMarkets: [market],
      warnings: [],
    },
    warnings: [],
  };
}

describe('arbiter replay labeler', () => {
  it('labels terminal markets from the settlement price at the forecast horizon', () => {
    const bundle = makeBundle({
      marketId: 'pm-terminal',
      assetId: 'asset-yes-1',
      question: 'Will Bitcoin be above $70,000 on May 8?',
      probability: 0.54,
      volume24h: 250000,
      endDate: '2026-05-08T00:00:00.000Z',
      semantics: 'terminal',
      extractedPriceLevels: [70000],
    });

    const labeled = labelReplayBundle(bundle, {
      points: [
        { at: '2026-05-01T00:00:00.000Z', price: 68000 },
        { at: '2026-05-04T00:00:00.000Z', price: 69500 },
        { at: '2026-05-08T00:00:00.000Z', price: 71000 },
      ],
    }, '2026-05-08T12:00:00.000Z');

    expect(labeled.labels?.forecast).toEqual({
      realizedPrice: 71000,
      realizedReturn: (71000 - 68000) / 68000,
      actualBinary: 1,
      labeledAt: '2026-05-08T12:00:00.000Z',
    });
    expect(labeled.labels?.semantic).toEqual([
      {
        marketId: 'pm-terminal',
        semantics: 'terminal',
        outcome: 'yes',
        labeledAt: '2026-05-08T12:00:00.000Z',
      },
    ]);
  });

  it('labels barrier-touch markets from the realized path, not just the final settlement', () => {
    const bundle = makeBundle({
      marketId: 'pm-barrier',
      assetId: 'asset-yes-2',
      question: 'Will Bitcoin reach $70,000 by May 8?',
      probability: 0.48,
      volume24h: 180000,
      endDate: '2026-05-08T00:00:00.000Z',
      semantics: 'barrier_touch',
      extractedPriceLevels: [70000],
    });

    const labeled = labelReplayBundle(bundle, {
      points: [
        { at: '2026-05-01T00:00:00.000Z', price: 68000 },
        { at: '2026-05-03T00:00:00.000Z', price: 70250 },
        { at: '2026-05-08T00:00:00.000Z', price: 69000 },
      ],
    }, '2026-05-08T12:00:00.000Z');

    expect(labeled.labels?.semantic).toEqual([
      {
        marketId: 'pm-barrier',
        semantics: 'barrier_touch',
        outcome: 'yes',
        labeledAt: '2026-05-08T12:00:00.000Z',
      },
    ]);
  });

  it('labels range markets from the final settlement against the stored bounds', () => {
    const bundle = makeBundle({
      marketId: 'pm-range',
      assetId: 'asset-yes-3',
      question: 'Will Bitcoin be between $65,000 and $69,000 on May 8?',
      probability: 0.31,
      volume24h: 140000,
      endDate: '2026-05-08T00:00:00.000Z',
      semantics: 'range',
      extractedPriceLevels: [65000, 69000],
    });

    const labeled = labelReplayBundle(bundle, {
      points: [
        { at: '2026-05-01T00:00:00.000Z', price: 68000 },
        { at: '2026-05-08T00:00:00.000Z', price: 70000 },
      ],
    }, '2026-05-08T12:00:00.000Z');

    expect(labeled.labels?.semantic).toEqual([
      {
        marketId: 'pm-range',
        semantics: 'range',
        outcome: 'no',
        labeledAt: '2026-05-08T12:00:00.000Z',
      },
    ]);
  });

  it('does not attach labels before the forecast horizon and market expiry are available', () => {
    const bundle = makeBundle({
      marketId: 'pm-terminal-pending',
      assetId: 'asset-yes-4',
      question: 'Will Bitcoin be above $70,000 on May 8?',
      probability: 0.54,
      volume24h: 250000,
      endDate: '2026-05-08T00:00:00.000Z',
      semantics: 'terminal',
      extractedPriceLevels: [70000],
    });

    const eligibility = evaluateReplayLabelEligibility(bundle, {
      points: [
        { at: '2026-05-01T00:00:00.000Z', price: 68000 },
        { at: '2026-05-04T00:00:00.000Z', price: 69500 },
      ],
    });
    const labeled = labelReplayBundle(bundle, {
      points: [
        { at: '2026-05-01T00:00:00.000Z', price: 68000 },
        { at: '2026-05-04T00:00:00.000Z', price: 69500 },
      ],
    }, '2026-05-04T12:00:00.000Z');

    expect(eligibility.ready).toBe(false);
    expect(eligibility.pendingReasons).toContain('Forecast horizon has not elapsed in the supplied price history.');
    expect(labeled.labels).toBeUndefined();
  });
});
