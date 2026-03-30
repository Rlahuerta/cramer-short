/**
 * Integration tests — StreamingThinkFilter unit edge cases and streamCallLlm routing.
 *
 * StreamingThinkFilter is tested as a pure unit (no network). These tests live in
 * the integration tier because they cover the streaming output path that only
 * matters when connected to a real thinking model (context for the logic).
 *
 * For live streaming with real Ollama see llm-streaming.e2e.test.ts.
 *
 * Run with:
 *   bun run test:integration
 */

import { describe, expect, it } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { StreamingThinkFilter, isThinkingModel } from '@/model/llm.js';
import { resolveProvider } from '@/providers.js';
import { getOllamaModels } from '@/utils/ollama.js';

// ---------------------------------------------------------------------------
// StreamingThinkFilter — pure unit edge cases (no network needed)
// ---------------------------------------------------------------------------

describe('StreamingThinkFilter', () => {
  it('passes through plain text unchanged', () => {
    const f = new StreamingThinkFilter();
    expect(f.process('hello world')).toBe('hello world');
    expect(f.flush()).toBe('');
  });

  it('strips a single complete <think>...</think> block', () => {
    const f = new StreamingThinkFilter();
    const out = f.process('<think>some internal reasoning</think>final answer');
    expect(out).toBe('final answer');
  });

  it('strips multiple think blocks in one chunk', () => {
    const f = new StreamingThinkFilter();
    const out = f.process('<think>step 1</think>A<think>step 2</think>B');
    expect(out).toBe('AB');
  });

  it('handles think block split across two process() calls', () => {
    const f = new StreamingThinkFilter();
    const part1 = f.process('<think>start of thi');
    const part2 = f.process('nking</think>answer');
    expect(part1 + part2).toBe('answer');
  });

  it('handles </think> tag split across chunk boundary', () => {
    const f = new StreamingThinkFilter();
    f.process('<think>reasoning</');
    const out = f.process('think>text');
    expect(out).toBe('text');
  });

  it('handles partial <think> opener at chunk boundary', () => {
    const f = new StreamingThinkFilter();
    // Chunk ends mid-tag — the partial <thi must be buffered, not emitted
    const out1 = f.process('before<thi');
    const out2 = f.process('nk>inside</think>after');
    expect(out1 + out2).toBe('beforeafter');
  });

  it('flush() returns any buffered non-thinking text', () => {
    const f = new StreamingThinkFilter();
    f.process('hel');  // partial word boundary, may be buffered
    const flushed = f.flush();
    expect(typeof flushed).toBe('string');
    // After flush, the buffer must be empty
    expect(f.flush()).toBe('');
  });

  it('flush() returns empty string when mid-thinking-block (unclosed <think>)', () => {
    const f = new StreamingThinkFilter();
    f.process('<think>this block was never closed');
    expect(f.flush()).toBe('');
  });

  it('flush() resets state — subsequent process() calls work fresh', () => {
    const f = new StreamingThinkFilter();
    f.process('<think>reasoning');  // mid-thinking-block, unclosed
    f.flush();                      // should reset this.thinking and this.buf
    // Now new input should be treated as normal text, not thinking
    const out = f.process('fresh text');
    expect(out).toBe('fresh text');
  });

  it('emits nothing while inside an open think block', () => {
    const f = new StreamingThinkFilter();
    expect(f.process('<think>')).toBe('');
    expect(f.process('lots of reasoning here')).toBe('');
    expect(f.process('more reasoning')).toBe('');
    expect(f.process('</think>')).toBe('');
    expect(f.process('answer')).toBe('answer');
  });

  it('text before and after a think block is both emitted', () => {
    const f = new StreamingThinkFilter();
    const out = f.process('prefix<think>hidden</think>suffix');
    expect(out).toBe('prefixsuffix');
  });

  it('handles empty input string without crashing', () => {
    const f = new StreamingThinkFilter();
    expect(f.process('')).toBe('');
  });

  it('handles input with no think tag as pass-through', () => {
    const f = new StreamingThinkFilter();
    const text = 'The quick brown fox jumps over the lazy dog.';
    expect(f.process(text)).toBe(text);
  });
});

// ---------------------------------------------------------------------------
// Thinking model classification — all known cloud models on this machine
// ---------------------------------------------------------------------------

describe('isThinkingModel — cloud model classification', () => {
  it('qwen3.5:397b-cloud is classified as thinking', () => {
    expect(isThinkingModel('ollama:qwen3.5:397b-cloud')).toBe(true);
  });

  it('qwen3.5:cloud is classified as thinking', () => {
    expect(isThinkingModel('ollama:qwen3.5:cloud')).toBe(true);
  });

  it('qwen3-coder-next:cloud is classified as thinking', () => {
    expect(isThinkingModel('ollama:qwen3-coder-next:cloud')).toBe(true);
  });

  it('qwen3-next:80b-cloud is classified as thinking', () => {
    expect(isThinkingModel('ollama:qwen3-next:80b-cloud')).toBe(true);
  });

  it('nemotron-3-super:cloud is classified as thinking', () => {
    expect(isThinkingModel('ollama:nemotron-3-super:cloud')).toBe(true);
  });

  it('nemotron-3-nano:30b-cloud is classified as thinking', () => {
    expect(isThinkingModel('ollama:nemotron-3-nano:30b-cloud')).toBe(true);
  });

  it('deepseek-v3.2:cloud is NOT classified as thinking', () => {
    expect(isThinkingModel('ollama:deepseek-v3.2:cloud')).toBe(false);
  });

  it('minimax-m2.7:cloud is NOT classified as thinking', () => {
    expect(isThinkingModel('ollama:minimax-m2.7:cloud')).toBe(false);
  });

  it('glm-5:cloud is NOT classified as thinking', () => {
    expect(isThinkingModel('ollama:glm-5:cloud')).toBe(false);
  });

  it('kimi-k2.5:cloud is NOT classified as thinking', () => {
    expect(isThinkingModel('ollama:kimi-k2.5:cloud')).toBe(false);
  });

  it('gpt-oss:120b-cloud is NOT classified as thinking', () => {
    expect(isThinkingModel('ollama:gpt-oss:120b-cloud')).toBe(false);
  });

  it('nomic-embed-text is NOT classified as thinking', () => {
    expect(isThinkingModel('ollama:nomic-embed-text:latest')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Provider routing edge cases
// ---------------------------------------------------------------------------

describe('resolveProvider — Ollama routing edge cases', () => {
  it('does not match partial ollama prefix (e.g. "xollama:model" → openai default)', () => {
    const p = resolveProvider('xollama:model');
    expect(p.id).not.toBe('ollama');
  });

  it('ollama prefix matching is case-sensitive (OLLAMA: → openai default)', () => {
    // modelPrefix is 'ollama:' (lowercase) — uppercase does not match
    const p = resolveProvider('OLLAMA:some-model');
    expect(p.id).not.toBe('ollama');
  });

  it('model without provider prefix falls through to its matching provider', () => {
    // deepseek has its own provider prefix, so it routes to deepseek (not openai)
    const p = resolveProvider('deepseek-r1');
    expect(p.id).toBe('deepseek');
  });

  it('completely unknown model prefix falls back to openai', () => {
    const p = resolveProvider('unknown-model-xyz');
    expect(p.id).toBe('openai');
  });
});

// ---------------------------------------------------------------------------
// Integration: model list × isThinkingModel cross-check
// ---------------------------------------------------------------------------

describe('isThinkingModel × live model list', () => {
  integrationIt('every cloud model in the live list can be classified without error', async () => {
    const models = await getOllamaModels();
    const cloudModels = models.filter((m) => m.endsWith(':cloud'));
    for (const m of cloudModels) {
      const result = isThinkingModel(`ollama:${m}`);
      expect(typeof result).toBe('boolean');
    }
  });
});
