import { describe, expect, it } from 'bun:test';
import { isFinishedHistoryItem, type HistoryItemStatus } from './types.js';

describe('isFinishedHistoryItem', () => {
  it.each([
    ['processing', false],
    ['complete', true],
    ['interrupted', true],
    ['error', true],
  ] as Array<[HistoryItemStatus, boolean]>)('%s → %p', (status, finished) => {
    expect(isFinishedHistoryItem({ id: '1', query: 'q', events: [], answer: '', status })).toBe(finished);
  });
});
