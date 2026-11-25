import { describe, expect, it } from 'bun:test';
import { anchorsAtFromHistory, sampleDailyCloses, type AnchorHistoryMarket } from './polymarket-anchor-history.js';

const ts = (iso: string) => Date.parse(iso) / 1000;

describe('sampleDailyCloses', () => {
  it('takes each UTC day\'s last price at or before 23:59:59Z, from the first traded day to the day before resolution', () => {
    const history = [
      { t: ts('2024-07-01T10:00:00Z'), p: 0.40 },
      { t: ts('2024-07-01T22:00:00Z'), p: 0.45 },
      { t: ts('2024-07-03T05:00:00Z'), p: 0.60 },
      { t: ts('2024-07-05T15:00:00Z'), p: 0.99 },
    ];
    expect(sampleDailyCloses(history, '2024-07-05T16:00:00Z')).toEqual({
      '2024-07-01': 0.45,
      '2024-07-02': 0.45, // no trade that day: the last close carries over
      '2024-07-03': 0.60,
      '2024-07-04': 0.60,
    });
  });

  it('is empty without history', () => {
    expect(sampleDailyCloses([], '2024-07-05T16:00:00Z')).toEqual({});
  });
});

describe('anchorsAtFromHistory', () => {
  const market: AnchorHistoryMarket = {
    question: 'Bitcoin above $61,000 on July 5?',
    createdAt: '2024-06-28T15:00:00Z',
    endDate: '2024-07-05T16:00:00Z',
    volume: 72_724,
    daily: { '2024-06-30': 0.47, '2024-07-01': 0.40 },
  };
  const dates = ['2024-06-29', '2024-06-30', '2024-07-01', '2024-07-06'];
  const anchorsAt = anchorsAtFromHistory(dates, [market]);

  it('shows a market at an origin with its probability that day, observed at the day\'s close', () => {
    expect(anchorsAt(1)).toEqual({
      referenceTimeMs: Date.parse('2024-06-30T23:59:59Z'),
      markets: [{
        question: market.question, probability: 0.47, volume: 72_724,
        createdAt: market.createdAt, endDate: market.endDate,
      }],
    });
    expect(anchorsAt(2).markets[0].probability).toBe(0.40);
  });

  it('hides it on days without a price and after it resolved', () => {
    expect(anchorsAt(0).markets).toEqual([]);
    expect(anchorsAt(3).markets).toEqual([]);
  });
});
