import { describe, test, expect } from 'bun:test';
import {
  lookupReplaySnapshot,
  filterReplayMarketsByTime,
  filterReplayMarketsByQuality,
  walkForwardWithReplay,
  type ReplaySnapshot,
  type ReplayMarket
} from './replay.js';

describe('Replay Module', () => {
  describe('lookupReplaySnapshot', () => {
    test('returns undefined for empty snapshots', () => {
      expect(lookupReplaySnapshot('2024-01-01', [])).toBeUndefined();
      expect(lookupReplaySnapshot('2024-01-01')).toBeUndefined();
    });

    test('returns exact match', () => {
      const snaps: ReplaySnapshot[] = [
        { date: '2024-01-01', markets: [] },
        { date: '2024-01-05', markets: [] }
      ];
      expect(lookupReplaySnapshot('2024-01-01', snaps)?.date).toBe('2024-01-01');
      expect(lookupReplaySnapshot('2024-01-05', snaps)?.date).toBe('2024-01-05');
    });

    test('returns latest prior date when no exact match', () => {
      const snaps: ReplaySnapshot[] = [
        { date: '2024-01-01', markets: [] },
        { date: '2024-01-05', markets: [] },
        { date: '2024-01-10', markets: [] }
      ];
      expect(lookupReplaySnapshot('2024-01-04', snaps)?.date).toBe('2024-01-01');
      expect(lookupReplaySnapshot('2024-01-06', snaps)?.date).toBe('2024-01-05');
      expect(lookupReplaySnapshot('2024-01-11', snaps)?.date).toBe('2024-01-10');
    });

    test('returns undefined if all snapshots are in the future', () => {
      const snaps: ReplaySnapshot[] = [
        { date: '2024-01-05', markets: [] }
      ];
      expect(lookupReplaySnapshot('2024-01-01', snaps)).toBeUndefined();
    });
  });

  describe('filterReplayMarketsByTime', () => {
    test('keeps markets without createdAt', () => {
      const markets: ReplayMarket[] = [
        { question: 'Will BTC hit 100k?', probability: 0.5 }
      ];
      const filtered = filterReplayMarketsByTime(markets, 1000);
      expect(filtered).toHaveLength(1);
    });

    test('rejects markets created after reference time', () => {
      const markets: ReplayMarket[] = [
        { question: 'Q1', probability: 0.5, createdAt: 1000 },
        { question: 'Q2', probability: 0.5, createdAt: 2000 },
        { question: 'Q3', probability: 0.5, createdAt: '1970-01-01T00:00:03.000Z' } // 3000 ms
      ];
      
      const filtered1 = filterReplayMarketsByTime(markets, 1500);
      expect(filtered1).toHaveLength(1);
      expect(filtered1[0].question).toBe('Q1');

      const filtered2 = filterReplayMarketsByTime(markets, 2500);
      expect(filtered2).toHaveLength(2);
      expect(filtered2.map(m => m.question)).toEqual(['Q1', 'Q2']);
      
      const filtered3 = filterReplayMarketsByTime(markets, 500);
      expect(filtered3).toHaveLength(0);
    });

    test('rejects closed or non-orderbook markets', () => {
      const markets: ReplayMarket[] = [
        { question: 'Tradable', probability: 0.5, active: true, closed: false, enableOrderBook: true },
        { question: 'Closed', probability: 0.5, active: true, closed: true, enableOrderBook: true },
        { question: 'Disabled book', probability: 0.5, active: true, closed: false, enableOrderBook: false },
      ];

      const filtered = filterReplayMarketsByTime(markets, 1000);
      expect(filtered).toHaveLength(1);
      expect(filtered[0].question).toBe('Tradable');
    });
  });

  describe('filterReplayMarketsByQuality', () => {
    const snapshots: ReplaySnapshot[] = [
      {
        date: '2024-06-01',
        markets: [
          {
            question: 'Will Bitcoin be above $70000 on June 19?',
            probability: 0.60,
            volume: 120000,
            endDate: '2024-06-19T00:00:00Z'
          },
          {
            question: 'Will Bitcoin be above $80000 on August 30?',
            probability: 0.20,
            volume: 120000,
            endDate: '2024-08-30T00:00:00Z'
          }
        ]
      },
      {
        date: '2024-06-05',
        markets: [
          {
            question: 'Will Bitcoin be above $70000 on June 19?',
            probability: 0.63,
            volume: 125000,
            endDate: '2024-06-19T00:00:00Z'
          },
          {
            question: 'Will Bitcoin be above $75000 on June 30?',
            probability: 0.48,
            volume: 500,
            endDate: '2024-06-30T00:00:00Z'
          },
          {
            question: 'Will Bitcoin be above $80000 on August 30?',
            probability: 0.18,
            volume: 125000,
            endDate: '2024-08-30T00:00:00Z'
          },
          {
            question: 'Will Bitcoin be above $90000 on June 30?',
            probability: 0.95,
            volume: 125000,
            endDate: '2024-06-30T00:00:00Z'
          }
        ]
      }
    ];

    test('filters by volume, persistence, and horizon alignment', () => {
      const filtered = filterReplayMarketsByQuality(
        snapshots[1].markets,
        '2024-06-05',
        snapshots,
        Date.parse('2024-06-05T23:59:59Z'),
        14,
        {
          minVolume: 1000,
          requirePersistence: true,
          maxProbabilityShock: 0.1,
          requireHorizonAlignment: true,
        },
      );

      expect(filtered.map(m => m.question)).toEqual([
        'Will Bitcoin be above $70000 on June 19?'
      ]);
    });

    test('allows markets through unchanged when no quality filters are provided', () => {
      const filtered = filterReplayMarketsByQuality(
        snapshots[1].markets,
        '2024-06-05',
        snapshots,
        Date.parse('2024-06-05T23:59:59Z'),
        14,
      );

      expect(filtered).toHaveLength(snapshots[1].markets.length);
    });

    test('requires a truly earlier snapshot when a later step reuses an older snapshot', async () => {
      const replaySnapshots: ReplaySnapshot[] = [
        {
          date: '2024-06-01',
          markets: [
            {
              question: 'Will Bitcoin be above $70000 on June 19?',
              probability: 0.60,
              volume: 120000,
              endDate: '2024-06-19T00:00:00Z'
            }
          ]
        }
      ];

      const result = await walkForwardWithReplay({
        ticker: 'BTC-USD',
        prices: [100, 101, 102, 103],
        dates: ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04'],
        horizon: 1,
        warmup: 1,
        stride: 1,
        replaySnapshots,
        replayQualityFilters: {
          minVolume: 1000,
          requirePersistence: true,
          maxProbabilityShock: 0.1,
          requireHorizonAlignment: true,
        },
      });

      expect(result.errors).toHaveLength(0);
      expect(result.steps).toHaveLength(2);
      for (const step of result.steps) {
        expect(step.decisionSource).not.toContain('replay-anchor');
        expect(step.trustedAnchors).toBe(0);
      }
    });

    test('forwards btcReturnThresholdMultiplier through replay path', async () => {
      const prices = [
        100,
        101.2,
        102.412,
        103.640944,
        104.884635328,
        106.143250951936,
        107.416970, 
        108.706, 
        107.61994,
        106.5437406,
        105.478303194,
        104.42352016206,
      ];
      const dates = [
        '2024-06-01',
        '2024-06-02',
        '2024-06-03',
        '2024-06-04',
        '2024-06-05',
        '2024-06-06',
        '2024-06-07',
        '2024-06-08',
        '2024-06-09',
        '2024-06-10',
        '2024-06-11',
        '2024-06-12',
      ];

      const defaultReplay = await walkForwardWithReplay({
        ticker: 'BTC-USD',
        prices,
        dates,
        horizon: 1,
        warmup: 8,
        stride: 1,
        replaySnapshots: [{ date: '2024-06-09', markets: [] }],
      });

      const widenedThresholdReplay = await walkForwardWithReplay({
        ticker: 'BTC-USD',
        prices,
        dates,
        horizon: 1,
        warmup: 8,
        stride: 1,
        replaySnapshots: [{ date: '2024-06-09', markets: [] }],
        btcReturnThresholdMultiplier: 1.0,
      });

      expect(defaultReplay.errors).toHaveLength(0);
      expect(widenedThresholdReplay.errors).toHaveLength(0);
      expect(defaultReplay.steps).toHaveLength(widenedThresholdReplay.steps.length);
      expect(defaultReplay.steps.length).toBeGreaterThan(0);

      const changedSteps = widenedThresholdReplay.steps.filter((step, index) => {
        const baseline = defaultReplay.steps[index];
        if (!baseline) return false;
        return (
          Math.abs(step.predictedProb - baseline.predictedProb) > 1e-9 ||
          step.regime !== baseline.regime ||
          step.recommendation !== baseline.recommendation
        );
      }).length;

      expect(changedSteps).toBeGreaterThan(0);
    });

    test('forwards startStateMixture through replay path', async () => {
      const prices = [
        100,
        101.2,
        102.412,
        103.640944,
        104.884635328,
        106.143250951936,
        107.416970,
        108.706,
        107.61994,
        106.5437406,
        105.478303194,
        104.42352016206,
      ];
      const dates = [
        '2024-06-01',
        '2024-06-02',
        '2024-06-03',
        '2024-06-04',
        '2024-06-05',
        '2024-06-06',
        '2024-06-07',
        '2024-06-08',
        '2024-06-09',
        '2024-06-10',
        '2024-06-11',
        '2024-06-12',
      ];

      const promotedDefaultReplay = await walkForwardWithReplay({
        ticker: 'BTC-USD',
        prices,
        dates,
        horizon: 1,
        warmup: 8,
        stride: 1,
        replaySnapshots: [{ date: '2024-06-09', markets: [] }],
      });

      const legacyControlReplay = await walkForwardWithReplay({
        ticker: 'BTC-USD',
        prices,
        dates,
        horizon: 1,
        warmup: 8,
        stride: 1,
        replaySnapshots: [{ date: '2024-06-09', markets: [] }],
        startStateMixture: false,
      });

      const explicitPromotedReplay = await walkForwardWithReplay({
        ticker: 'BTC-USD',
        prices,
        dates,
        horizon: 1,
        warmup: 8,
        stride: 1,
        replaySnapshots: [{ date: '2024-06-09', markets: [] }],
        startStateMixture: true,
      });

      expect(promotedDefaultReplay.errors).toHaveLength(0);
      expect(legacyControlReplay.errors).toHaveLength(0);
      expect(explicitPromotedReplay.errors).toHaveLength(0);
      expect(promotedDefaultReplay.steps).toHaveLength(legacyControlReplay.steps.length);
      expect(promotedDefaultReplay.steps).toHaveLength(explicitPromotedReplay.steps.length);
      expect(promotedDefaultReplay.steps.length).toBeGreaterThan(0);

      const changedSteps = promotedDefaultReplay.steps.filter((step, index) => {
        const legacy = legacyControlReplay.steps[index];
        const explicit = explicitPromotedReplay.steps[index];
        if (!legacy || !explicit) return false;
        return (
          Math.abs(step.predictedProb - legacy.predictedProb) > 1e-9 ||
          step.regime !== legacy.regime ||
          step.recommendation !== legacy.recommendation
        ) && (
          Math.abs(step.predictedProb - explicit.predictedProb) <= 1e-9 &&
          step.regime === explicit.regime &&
          step.recommendation === explicit.recommendation
        );
      }).length;

      expect(changedSteps).toBeGreaterThan(0);
    });
  });
});
