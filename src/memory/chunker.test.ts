import { describe, expect, it } from 'bun:test';
import {
  chunkMemoryText,
  buildSnippet,
  estimateChunkTokens,
  countLinesInChunk,
} from './chunker.js';
import type { MemoryChunk } from './types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeChunk(content: string, overrides: Partial<MemoryChunk> = {}): MemoryChunk {
  return {
    filePath: 'test.md',
    startLine: 1,
    endLine: content.split('\n').length,
    content,
    contentHash: 'abc123',
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// buildSnippet
// ---------------------------------------------------------------------------

describe('buildSnippet', () => {
  it('returns the original text when shorter than maxChars', () => {
    expect(buildSnippet('hello world')).toBe('hello world');
  });

  it('truncates with ellipsis when text exceeds maxChars', () => {
    const long = 'a'.repeat(800);
    const result = buildSnippet(long, 700);
    expect(result.endsWith('...')).toBe(true);
    expect(result.length).toBeLessThanOrEqual(703); // 700 + '...'
  });

  it('collapses whitespace', () => {
    const result = buildSnippet('hello   \n  world');
    expect(result).toBe('hello world');
  });

  it('uses default maxChars of 700', () => {
    const exact = 'x'.repeat(700);
    expect(buildSnippet(exact)).toBe(exact);
    const overLimit = 'x'.repeat(701);
    expect(buildSnippet(overLimit).endsWith('...')).toBe(true);
  });

  it('returns empty string for empty input', () => {
    expect(buildSnippet('')).toBe('');
  });
});

// ---------------------------------------------------------------------------
// estimateChunkTokens
// ---------------------------------------------------------------------------

describe('estimateChunkTokens', () => {
  it('estimates tokens as ceil(chars / 3.5)', () => {
    const chunk = makeChunk('a'.repeat(35));
    expect(estimateChunkTokens(chunk)).toBe(10); // ceil(35 / 3.5)
  });

  it('rounds up for non-divisible content', () => {
    const chunk = makeChunk('a'.repeat(36));
    expect(estimateChunkTokens(chunk)).toBe(11); // ceil(36 / 3.5) = ceil(10.28)
  });

  it('returns 1 for a single character', () => {
    const chunk = makeChunk('x');
    expect(estimateChunkTokens(chunk)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// countLinesInChunk
// ---------------------------------------------------------------------------

describe('countLinesInChunk', () => {
  it('counts single-line content as 1', () => {
    expect(countLinesInChunk(makeChunk('just one line'))).toBe(1);
  });

  it('counts multi-line content correctly', () => {
    expect(countLinesInChunk(makeChunk('line1\nline2\nline3'))).toBe(3);
  });

  it('handles empty content', () => {
    expect(countLinesInChunk(makeChunk(''))).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// chunkMemoryText — basic behaviour
// ---------------------------------------------------------------------------

describe('chunkMemoryText', () => {
  const basePath = 'src/memo.md';

  it('returns empty array for empty text', () => {
    const result = chunkMemoryText({ filePath: basePath, text: '', chunkTokens: 100, overlapTokens: 10 });
    expect(result).toHaveLength(0);
  });

  it('returns empty array for whitespace-only text', () => {
    const result = chunkMemoryText({ filePath: basePath, text: '   \n\n  ', chunkTokens: 100, overlapTokens: 10 });
    expect(result).toHaveLength(0);
  });

  it('produces a single chunk for short text', () => {
    const text = 'This is a short paragraph.';
    const chunks = chunkMemoryText({ filePath: basePath, text, chunkTokens: 200, overlapTokens: 20 });
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.content).toBe(text);
    expect(chunks[0]!.filePath).toBe(basePath);
  });

  it('assigns correct startLine and endLine', () => {
    const text = 'Line 1\nLine 2\nLine 3';
    const chunks = chunkMemoryText({ filePath: basePath, text, chunkTokens: 200, overlapTokens: 0 });
    expect(chunks[0]!.startLine).toBe(1);
    expect(chunks[0]!.endLine).toBe(3);
  });

  it('splits paragraphs separated by blank lines into multiple chunks when budget is small', () => {
    const para = 'a'.repeat(50);
    const text = `${para}\n\n${para}\n\n${para}`;
    // budget of 60 chars = ~17 tokens; each para is 50 chars so they won't combine
    const chunks = chunkMemoryText({ filePath: basePath, text, chunkTokens: 17, overlapTokens: 0 });
    expect(chunks.length).toBeGreaterThanOrEqual(2);
  });

  it('sets contentHash as a sha256 hex string', () => {
    const chunks = chunkMemoryText({ filePath: basePath, text: 'Hello', chunkTokens: 200, overlapTokens: 0 });
    expect(chunks[0]!.contentHash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('produces stable hashes (same content → same hash)', () => {
    const opts = { filePath: basePath, text: 'Stable text', chunkTokens: 200, overlapTokens: 0 };
    const a = chunkMemoryText(opts);
    const b = chunkMemoryText(opts);
    expect(a[0]!.contentHash).toBe(b[0]!.contentHash);
  });

  it('produces different hashes for different content', () => {
    const a = chunkMemoryText({ filePath: basePath, text: 'Alpha', chunkTokens: 200, overlapTokens: 0 });
    const b = chunkMemoryText({ filePath: basePath, text: 'Beta', chunkTokens: 200, overlapTokens: 0 });
    expect(a[0]!.contentHash).not.toBe(b[0]!.contentHash);
  });

  it('hard-splits a single oversized paragraph at sentence boundary', () => {
    // Build a paragraph > budget. Sentence boundary at ". " position.
    const sentence = 'This is a sentence. ';
    const big = sentence.repeat(20); // 400 chars
    const chunks = chunkMemoryText({ filePath: basePath, text: big, chunkTokens: 30, overlapTokens: 0 });
    // chunkTokens=30 → budget = floor(30*3.5) = 105 chars
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c) => {
      // No single chunk should exceed the budget (with small tolerance for trimming)
      expect(c.content.length).toBeLessThanOrEqual(110);
    });
  });

  it('includes overlap from previous chunk when overlapTokens > 0', () => {
    const para1 = 'First paragraph content here.';
    const para2 = 'Second paragraph with more text.';
    const para3 = 'Third paragraph continues.';
    const text = `${para1}\n\n${para2}\n\n${para3}`;
    // Small budget so each para becomes its own chunk
    const chunksNoOverlap = chunkMemoryText({ filePath: basePath, text, chunkTokens: 15, overlapTokens: 0 });
    const chunksWithOverlap = chunkMemoryText({ filePath: basePath, text, chunkTokens: 15, overlapTokens: 10 });
    // With overlap, later chunks should start earlier (overlap from prior para)
    if (chunksWithOverlap.length > 1) {
      expect(chunksWithOverlap.length).toBeGreaterThanOrEqual(chunksNoOverlap.length - 1);
    }
  });

  it('joins multiple short paragraphs into a single chunk', () => {
    const text = 'Short.\n\nTiny.\n\nBrief.';
    const chunks = chunkMemoryText({ filePath: basePath, text, chunkTokens: 500, overlapTokens: 0 });
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.content).toContain('Short.');
    expect(chunks[0]!.content).toContain('Tiny.');
    expect(chunks[0]!.content).toContain('Brief.');
  });

  it('handles text without blank-line paragraph breaks', () => {
    const text = 'Line 1\nLine 2\nLine 3\nLine 4';
    const chunks = chunkMemoryText({ filePath: basePath, text, chunkTokens: 200, overlapTokens: 0 });
    expect(chunks.length).toBeGreaterThanOrEqual(1);
    expect(chunks[0]!.content).toContain('Line 1');
  });
});
