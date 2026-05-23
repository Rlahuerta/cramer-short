/**
 * Tests for relevantMessagesByQuery LRU eviction in InMemoryChatHistory.
 * selectRelevantMessages() calls callLlm — mocked here to avoid real API calls.
 */

import { beforeEach, describe, it, expect, mock } from 'bun:test';

let relevanceCallCount = 0;

mock.module('../model/llm.js', () => ({
  callLlm: async () => {
    relevanceCallCount++;
    return { response: { message_ids: [] } };
  },
  DEFAULT_MODEL: 'gpt-4o',
}));

const { InMemoryChatHistory, RELEVANCE_CACHE_MAX_SIZE } = await import('./in-memory-chat-history.js');

beforeEach(() => {
  relevanceCallCount = 0;
});

describe('InMemoryChatHistory — relevantMessagesByQuery eviction', () => {
  it('keeps relevance cache bounded and evicts the least recently used query', async () => {
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'base', answer: 'answer', summary: 'summary' });

    for (let i = 0; i < RELEVANCE_CACHE_MAX_SIZE; i++) {
      await h.selectRelevantMessages(`q-${i}`);
    }
    expect(relevanceCallCount).toBe(RELEVANCE_CACHE_MAX_SIZE);

    await h.selectRelevantMessages('q-0');
    expect(relevanceCallCount).toBe(RELEVANCE_CACHE_MAX_SIZE);

    await h.selectRelevantMessages('q-overflow');
    expect(relevanceCallCount).toBe(RELEVANCE_CACHE_MAX_SIZE + 1);

    await h.selectRelevantMessages('q-0');
    expect(relevanceCallCount).toBe(RELEVANCE_CACHE_MAX_SIZE + 1);

    await h.selectRelevantMessages('q-1');
    expect(relevanceCallCount).toBe(RELEVANCE_CACHE_MAX_SIZE + 2);
  });
});
