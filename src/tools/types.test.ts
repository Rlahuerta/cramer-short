import { describe, it, expect } from 'bun:test';
import { formatToolResult, parseSearchResults } from './types.js';

describe('formatToolResult', () => {
  it('includes sourceUrls when urls provided', () => {
    const output = formatToolResult({ key: 'value' }, ['https://example.com']);
    const parsed = JSON.parse(output);
    expect(parsed.data).toEqual({ key: 'value' });
    expect(parsed.sourceUrls).toEqual(['https://example.com']);
  });

  it('omits sourceUrls when not provided', () => {
    const output = formatToolResult({ key: 'value' });
    const parsed = JSON.parse(output);
    expect(parsed.data).toEqual({ key: 'value' });
    expect('sourceUrls' in parsed).toBe(false);
  });

  it('omits sourceUrls when empty array provided', () => {
    const output = formatToolResult({ key: 'value' }, []);
    const parsed = JSON.parse(output);
    expect(parsed.data).toEqual({ key: 'value' });
    expect('sourceUrls' in parsed).toBe(false);
  });
});

describe('parseSearchResults', () => {
  it('parses Exa format { results: [{url}] } from JSON string', () => {
    const input = JSON.stringify({ results: [{ url: 'https://exa.ai' }, { url: 'https://other.com' }] });
    const { parsed, urls } = parseSearchResults(input);
    expect(urls).toEqual(['https://exa.ai', 'https://other.com']);
    expect((parsed as { results: unknown[] }).results).toHaveLength(2);
  });

  it('parses Tavily format [{url}] from JSON string', () => {
    const input = JSON.stringify([{ url: 'https://tavily.com' }, { url: 'https://another.com' }]);
    const { parsed, urls } = parseSearchResults(input);
    expect(urls).toEqual(['https://tavily.com', 'https://another.com']);
    expect(Array.isArray(parsed)).toBe(true);
  });

  it('returns string and empty urls for invalid JSON', () => {
    const { parsed, urls } = parseSearchResults('not valid json');
    expect(parsed).toBe('not valid json');
    expect(urls).toEqual([]);
  });

  it('handles object input with results shape', () => {
    const input = { results: [{ url: 'https://example.com' }] };
    const { parsed, urls } = parseSearchResults(input);
    expect(urls).toEqual(['https://example.com']);
    expect(parsed).toBe(input);
  });

  it('handles plain array input', () => {
    const input = [{ url: 'https://example.com' }, { url: 'https://test.com' }];
    const { parsed, urls } = parseSearchResults(input);
    expect(urls).toEqual(['https://example.com', 'https://test.com']);
    expect(Array.isArray(parsed)).toBe(true);
  });

  it('ignores results missing url fields', () => {
    const input = { results: [{ title: 'no url' }, { url: 'https://ok.com' }] };
    const { urls } = parseSearchResults(input);
    expect(urls).toEqual(['https://ok.com']);
  });
});
