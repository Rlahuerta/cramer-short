/**
 * Tests for relevantMessagesByQuery LRU eviction in InMemoryChatHistory.
 * selectRelevantMessages() calls callLlm — mocked here to avoid real API calls.
 */

import { afterAll, beforeEach, describe, it, expect, spyOn } from 'bun:test';
import * as llm from '../model/llm.js';

let relevanceCallCount = 0;

// spyOn (restored afterAll) rather than mock.module, which would leak into later test files.
const callLlmSpy = spyOn(llm, 'callLlm').mockImplementation((async () => {
  relevanceCallCount++;
  return { response: { message_ids: [] } };
}) as unknown as typeof llm.callLlm);
afterAll(() => callLlmSpy.mockRestore());

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
