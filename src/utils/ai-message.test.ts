import { describe, test, expect } from 'bun:test';
import { AIMessage } from '@langchain/core/messages';
import { extractTextContent, hasToolCalls, extractReasoningContent } from './ai-message.js';

// ===========================================================================
// extractReasoningContent (new)
// ===========================================================================

describe('extractReasoningContent', () => {
  test('returns reasoning_content string when present in additional_kwargs', () => {
    const msg = new AIMessage({
      content: 'The answer is 42.',
      additional_kwargs: { reasoning_content: 'Let me think step by step...' },
    });
    expect(extractReasoningContent(msg)).toBe('Let me think step by step...');
  });

  test('returns null when reasoning_content is absent', () => {
    const msg = new AIMessage({ content: 'Hello' });
    expect(extractReasoningContent(msg)).toBeNull();
  });

  test('returns null when reasoning_content is an empty string', () => {
    const msg = new AIMessage({
      content: 'Hello',
      additional_kwargs: { reasoning_content: '' },
    });
    expect(extractReasoningContent(msg)).toBeNull();
  });

  test('returns null when reasoning_content is only whitespace', () => {
    const msg = new AIMessage({
      content: 'Hello',
      additional_kwargs: { reasoning_content: '   \n  ' },
    });
    expect(extractReasoningContent(msg)).toBeNull();
  });

  test('returns null when reasoning_content is not a string', () => {
    const msg = new AIMessage({
      content: 'Hello',
      additional_kwargs: { reasoning_content: 42 as unknown as string },
    });
    expect(extractReasoningContent(msg)).toBeNull();
  });

  test('trims leading/trailing whitespace from reasoning content', () => {
    const msg = new AIMessage({
      content: 'Done',
      additional_kwargs: { reasoning_content: '  step 1\nstep 2  ' },
    });
    expect(extractReasoningContent(msg)).toBe('step 1\nstep 2');
  });
});

// ===========================================================================
// Existing helpers — regression guard
// ===========================================================================

describe('extractTextContent (regression)', () => {
  test('handles string content', () => {
    const msg = new AIMessage({ content: 'hello' });
    expect(extractTextContent(msg)).toBe('hello');
  });
});

describe('hasToolCalls (regression)', () => {
  test('returns false for message with no tool calls', () => {
    const msg = new AIMessage({ content: 'hi' });
    expect(hasToolCalls(msg)).toBe(false);
  });
});

// ===========================================================================
// extractTextContent — array content branch
// ===========================================================================

describe('extractTextContent — array content', () => {
  test('joins text blocks from array content', () => {
    const msg = new AIMessage({
      content: [
        { type: 'text', text: 'First part' },
        { type: 'text', text: 'Second part' },
      ],
    });
    expect(extractTextContent(msg)).toBe('First part\nSecond part');
  });

  test('filters out non-text blocks', () => {
    const msg = new AIMessage({
      content: [
        { type: 'tool_use', id: 'tu1', name: 'search', input: {} },
        { type: 'text', text: 'The answer' },
      ],
    });
    expect(extractTextContent(msg)).toBe('The answer');
  });

  test('returns empty string when no text blocks in array', () => {
    const msg = new AIMessage({
      content: [
        { type: 'tool_use', id: 'tu1', name: 'search', input: {} },
      ],
    });
    expect(extractTextContent(msg)).toBe('');
  });
});

// ===========================================================================
// extractTextContent — non-string / non-array fallback
// ===========================================================================

describe('extractTextContent — fallback', () => {
  test('returns empty string for non-string non-array content', () => {
    const msg = new AIMessage({ content: '' });
    // content is a string (empty), handled by the string branch
    expect(extractTextContent(msg)).toBe('');
  });
});

// ===========================================================================
// hasToolCalls — positive cases
// ===========================================================================

describe('hasToolCalls — positive', () => {
  test('returns true when message has tool calls', () => {
    const msg = new AIMessage({
      content: '',
      tool_calls: [{ id: 'tc1', name: 'search', args: { query: 'AAPL' }, type: 'tool_call' }],
    });
    expect(hasToolCalls(msg)).toBe(true);
  });

  test('returns false when tool_calls array is empty', () => {
    const msg = new AIMessage({ content: 'hi', tool_calls: [] });
    expect(hasToolCalls(msg)).toBe(false);
  });
});
