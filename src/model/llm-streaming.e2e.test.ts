/**
 * E2E tests — live streamCallLlm with real Ollama.
 *
 * Must run in an isolated process to avoid @langchain/ollama mock contamination
 * from agent.test.ts (which permanently mocks the module for the worker).
 *
 * Run with:
 *   bun run test:e2e
 *   # or
 *   RUN_E2E=1 bun test src/model/llm-streaming.e2e.test.ts
 *
 * These tests require:
 *   - Ollama running on http://127.0.0.1:11434
 *   - nemotron-3-super:cloud (thinking model — for think-filter test)
 *   - deepseek-v3.2:cloud  (non-thinking model — for plain streaming test)
 */

import { describe, expect } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
import { streamCallLlm } from '@/model/llm.js';

const THINKING_MODEL = 'ollama:nemotron-3-super:cloud';
const PLAIN_MODEL = 'ollama:deepseek-v3.2:cloud';

// ---------------------------------------------------------------------------
// Basic streaming
// ---------------------------------------------------------------------------

describe('streamCallLlm — live Ollama streaming', () => {
  e2eIt(
    'yields multiple string chunks for a short prompt',
    async () => {
      const chunks: string[] = [];
      for await (const chunk of streamCallLlm('Reply with exactly: PONG', {
        model: PLAIN_MODEL,
        thinkOverride: false,
      })) {
        chunks.push(chunk);
      }
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks.every((c) => typeof c === 'string')).toBe(true);
    },
    120_000,
  );

  e2eIt(
    'concatenated chunks form a coherent answer',
    async () => {
      const chunks: string[] = [];
      for await (const chunk of streamCallLlm('Reply with exactly: STREAM_OK', {
        model: PLAIN_MODEL,
        thinkOverride: false,
      })) {
        chunks.push(chunk);
      }
      const full = chunks.join('');
      expect(full.length).toBeGreaterThan(0);
      expect(full.toLowerCase()).toContain('stream_ok');
    },
    120_000,
  );

  e2eIt(
    'all yielded chunks are non-empty strings',
    async () => {
      const chunks: string[] = [];
      for await (const chunk of streamCallLlm('What is 1 + 1?', {
        model: PLAIN_MODEL,
        thinkOverride: false,
      })) {
        chunks.push(chunk);
        // Every yielded chunk must be non-empty (the filter and buffer do this)
        expect(chunk.length).toBeGreaterThan(0);
      }
      expect(chunks.length).toBeGreaterThan(0);
    },
    120_000,
  );
});

// ---------------------------------------------------------------------------
// Think-tag stripping (thinking models)
// ---------------------------------------------------------------------------

describe('streamCallLlm — <think> block stripping', () => {
  e2eIt(
    'no <think> tags appear in streamed output from a thinking model',
    async () => {
      const chunks: string[] = [];
      for await (const chunk of streamCallLlm('Reply with exactly: THINK_STRIPPED', {
        model: THINKING_MODEL,
        // thinkOverride not set — will use isThinkingModel() → true for nemotron
      })) {
        chunks.push(chunk);
      }
      const full = chunks.join('');
      expect(full).not.toContain('<think>');
      expect(full).not.toContain('</think>');
    },
    120_000,
  );

  e2eIt(
    'thinking model with thinkOverride:false still produces output',
    async () => {
      const chunks: string[] = [];
      for await (const chunk of streamCallLlm('Reply with exactly: OVERRIDE_OK', {
        model: THINKING_MODEL,
        thinkOverride: false,
      })) {
        chunks.push(chunk);
      }
      const full = chunks.join('');
      expect(full.length).toBeGreaterThan(0);
    },
    120_000,
  );
});

// ---------------------------------------------------------------------------
// AbortSignal
// ---------------------------------------------------------------------------

describe('streamCallLlm — AbortSignal', () => {
  e2eIt(
    'AbortSignal stops the stream early',
    async () => {
      const ac = new AbortController();
      const chunks: string[] = [];

      // Abort after collecting a small number of chunks
      let collected = 0;
      try {
        for await (const chunk of streamCallLlm(
          'Count from 1 to 1000, one number per line.',
          { model: PLAIN_MODEL, signal: ac.signal, thinkOverride: false },
        )) {
          chunks.push(chunk);
          collected++;
          if (collected >= 3) {
            ac.abort();
          }
        }
      } catch {
        // AbortError is expected when signal fires
      }

      // We should have collected some chunks before aborting
      expect(chunks.length).toBeGreaterThan(0);
      // And we should have stopped well before the full 1000 numbers
      const full = chunks.join('');
      const lineCount = full.split('\n').filter(Boolean).length;
      expect(lineCount).toBeLessThan(50);
    },
    120_000,
  );
});
