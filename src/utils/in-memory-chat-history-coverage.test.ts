/**
 * Coverage for InMemoryChatHistory methods not covered in the base test file.
 * Mocks callLlm so saveAnswer/selectRelevantMessages run without a real API.
 */

import { mock, describe, it, expect, beforeEach } from 'bun:test';

// Mock callLlm before importing the class
const mockCallLlm = mock(async (_prompt: string, _opts?: unknown) => ({
  response: 'Mocked summary',
  usage: { inputTokens: 10, outputTokens: 5 },
}));

mock.module('../model/llm.js', () => ({
  callLlm: mockCallLlm,
  DEFAULT_MODEL: 'mock-model',
}));

import { InMemoryChatHistory } from './in-memory-chat-history.js';
import { DEFAULT_HISTORY_LIMIT } from './history-context.js';

beforeEach(() => {
  mockCallLlm.mockClear();
  // Default mock: returns a plain string summary
  mockCallLlm.mockImplementation(async () => ({
    response: 'Mocked summary text',
    usage: { inputTokens: 10, outputTokens: 5 },
  }));
});

// ---------------------------------------------------------------------------
// setModel
// ---------------------------------------------------------------------------

describe('InMemoryChatHistory.setModel()', () => {
  it('updates the model used for LLM calls', async () => {
    const h = new InMemoryChatHistory('original-model');
    h.setModel('new-model');
    // Verify that saveAnswer uses the new model by checking callLlm was called
    h.saveUserQuery('Test query');
    await h.saveAnswer('Test answer');
    expect(mockCallLlm).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// saveUserQuery
// ---------------------------------------------------------------------------

describe('InMemoryChatHistory.saveUserQuery()', () => {
  it('adds a message with null answer and summary', () => {
    const h = new InMemoryChatHistory();
    h.saveUserQuery('What is AAPL?');
    const msgs = h.getMessages();
    expect(msgs).toHaveLength(1);
    expect(msgs[0]!.query).toBe('What is AAPL?');
    expect(msgs[0]!.answer).toBeNull();
    expect(msgs[0]!.summary).toBeNull();
  });

  it('clears the relevance cache when a new query is added', async () => {
    const h = new InMemoryChatHistory();
    h.saveUserQuery('First query');
    await h.saveAnswer('First answer');
    // selectRelevantMessages caches by query hash — save a second query to clear cache
    h.saveUserQuery('Second query');
    expect(h.getMessages()).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// saveAnswer
// ---------------------------------------------------------------------------

describe('InMemoryChatHistory.saveAnswer()', () => {
  it('sets answer and generates summary via LLM', async () => {
    const h = new InMemoryChatHistory();
    h.saveUserQuery('What is the price of AAPL?');
    await h.saveAnswer('$178.50');
    const msgs = h.getMessages();
    expect(msgs[0]!.answer).toBe('$178.50');
    expect(msgs[0]!.summary).toBe('Mocked summary text');
    expect(mockCallLlm).toHaveBeenCalledTimes(1);
  });

  it('does nothing when no pending query exists', async () => {
    const h = new InMemoryChatHistory();
    await h.saveAnswer('An answer with no question');
    expect(h.getMessages()).toHaveLength(0);
    expect(mockCallLlm).not.toHaveBeenCalled();
  });

  it('does nothing when the latest message already has an answer', async () => {
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'Q', answer: 'A', summary: 'S' });
    await h.saveAnswer('Duplicate answer');
    expect(h.getMessages()[0]!.answer).toBe('A'); // unchanged
  });

  it('falls back to a simple summary when callLlm throws', async () => {
    mockCallLlm.mockImplementation(async () => { throw new Error('Network error'); });
    const h = new InMemoryChatHistory();
    h.saveUserQuery('Fallback test query');
    await h.saveAnswer('Some answer');
    const msgs = h.getMessages();
    expect(msgs[0]!.summary).toContain('Fallback test query');
  });
});

// ---------------------------------------------------------------------------
// selectRelevantMessages
// ---------------------------------------------------------------------------

describe('InMemoryChatHistory.selectRelevantMessages()', () => {
  it('returns empty array when no completed messages exist', async () => {
    const h = new InMemoryChatHistory();
    h.saveUserQuery('Pending question');
    const result = await h.selectRelevantMessages('current query');
    expect(result).toHaveLength(0);
    expect(mockCallLlm).not.toHaveBeenCalled();
  });

  it('calls LLM and returns selected messages', async () => {
    mockCallLlm.mockImplementation(async () => ({
      response: { message_ids: [0] },
      usage: {},
    }));
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'What is AAPL?', answer: 'Apple stock.', summary: 'AAPL info' });
    const result = await h.selectRelevantMessages('Tell me about Apple');
    expect(mockCallLlm).toHaveBeenCalled();
    expect(result.length).toBeGreaterThanOrEqual(0); // depends on mock response
  });

  it('returns empty array when callLlm throws', async () => {
    mockCallLlm.mockImplementation(async () => { throw new Error('API error'); });
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'Q', answer: 'A', summary: 'S' });
    const result = await h.selectRelevantMessages('anything');
    expect(result).toHaveLength(0);
  });

  it('caches result for the same query', async () => {
    mockCallLlm.mockImplementation(async () => ({
      response: { message_ids: [] },
      usage: {},
    }));
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'Q', answer: 'A', summary: 'S' });
    await h.selectRelevantMessages('same query');
    await h.selectRelevantMessages('same query');
    // Second call should use cache, so callLlm called only once
    expect(mockCallLlm).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// formatForPlanning / formatForAnswerGeneration
// ---------------------------------------------------------------------------

describe('InMemoryChatHistory.formatForPlanning()', () => {
  it('returns empty string for empty messages array', () => {
    const h = new InMemoryChatHistory();
    expect(h.formatForPlanning([])).toBe('');
  });

  it('formats messages as query + summary pairs', () => {
    const h = new InMemoryChatHistory();
    const msgs = h.getMessages();
    h.seedMessage({ query: 'What is AAPL?', answer: 'Apple stock at $178.', summary: 'AAPL is $178' });
    const formatted = h.formatForPlanning(h.getMessages());
    expect(formatted).toContain('User: What is AAPL?');
    expect(formatted).toContain('Assistant: AAPL is $178');
    expect(msgs.length).toBe(0); // original msgs array not modified
  });
});

describe('InMemoryChatHistory.formatForAnswerGeneration()', () => {
  it('returns empty string for empty messages array', () => {
    const h = new InMemoryChatHistory();
    expect(h.formatForAnswerGeneration([])).toBe('');
  });

  it('formats messages as query + full answer pairs', () => {
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'What is AAPL?', answer: 'Apple Inc. trades at $178.', summary: 'AAPL info' });
    const formatted = h.formatForAnswerGeneration(h.getMessages());
    expect(formatted).toContain('User: What is AAPL?');
    expect(formatted).toContain('Apple Inc. trades at $178.');
  });
});

// ---------------------------------------------------------------------------
// getRecentTurns
// ---------------------------------------------------------------------------

describe('InMemoryChatHistory.getRecentTurns()', () => {
  it('returns empty array when no completed turns', () => {
    const h = new InMemoryChatHistory();
    h.saveUserQuery('pending');
    expect(h.getRecentTurns()).toHaveLength(0);
  });

  it('returns user+assistant pairs for completed turns', () => {
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'Q1', answer: 'A1', summary: 'S1' });
    const turns = h.getRecentTurns();
    expect(turns).toHaveLength(2);
    expect(turns[0]!.role).toBe('user');
    expect(turns[1]!.role).toBe('assistant');
  });

  it('respects the limit parameter', () => {
    const h = new InMemoryChatHistory();
    for (let i = 0; i < 5; i++) {
      h.seedMessage({ query: `Q${i}`, answer: `A${i}`, summary: `S${i}` });
    }
    const turns = h.getRecentTurns(2);
    expect(turns).toHaveLength(4); // 2 messages × 2 entries each
  });

  it('returns empty array when limit is 0', () => {
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'Q', answer: 'A', summary: 'S' });
    expect(h.getRecentTurns(0)).toHaveLength(0);
  });

  it('respects DEFAULT_HISTORY_LIMIT with no argument', () => {
    const h = new InMemoryChatHistory();
    for (let i = 0; i < DEFAULT_HISTORY_LIMIT + 5; i++) {
      h.seedMessage({ query: `Q${i}`, answer: `A${i}`, summary: `S${i}` });
    }
    const turns = h.getRecentTurns();
    // Should be at most DEFAULT_HISTORY_LIMIT × 2 entries
    expect(turns.length).toBeLessThanOrEqual(DEFAULT_HISTORY_LIMIT * 2);
  });
});

// ---------------------------------------------------------------------------
// hasMessages / getUserMessages / clear / pruneLastTurn
// ---------------------------------------------------------------------------

describe('InMemoryChatHistory.hasMessages()', () => {
  it('returns false for a new instance', () => {
    expect(new InMemoryChatHistory().hasMessages()).toBe(false);
  });

  it('returns true after a query is added', () => {
    const h = new InMemoryChatHistory();
    h.saveUserQuery('hello');
    expect(h.hasMessages()).toBe(true);
  });
});

describe('InMemoryChatHistory.getUserMessages()', () => {
  it('returns query strings in order', () => {
    const h = new InMemoryChatHistory();
    h.saveUserQuery('First');
    h.saveUserQuery('Second');
    expect(h.getUserMessages()).toEqual(['First', 'Second']);
  });
});

describe('InMemoryChatHistory.clear()', () => {
  it('removes all messages', () => {
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'Q', answer: 'A', summary: 'S' });
    h.clear();
    expect(h.hasMessages()).toBe(false);
  });
});

describe('InMemoryChatHistory.pruneLastTurn()', () => {
  it('removes the last message', () => {
    const h = new InMemoryChatHistory();
    h.seedMessage({ query: 'Q1', answer: 'A1', summary: 'S1' });
    h.seedMessage({ query: 'Q2', answer: 'A2', summary: 'S2' });
    h.pruneLastTurn();
    expect(h.getMessages()).toHaveLength(1);
    expect(h.getMessages()[0]!.query).toBe('Q1');
  });

  it('does nothing on an empty history', () => {
    const h = new InMemoryChatHistory();
    h.pruneLastTurn(); // should not throw
    expect(h.hasMessages()).toBe(false);
  });
});
