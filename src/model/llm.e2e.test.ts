/**
 * E2E tests — Ollama cloud model live prompt execution.
 *
 * Runs real callLlm calls against Ollama. Must execute in an isolated process
 * to avoid agent.test.ts's permanent mock.module('@langchain/ollama') override.
 *
 * Prerequisites:
 *   - Ollama running at OLLAMA_BASE_URL (default http://127.0.0.1:11434)
 *   - At least one `*:cloud` model available
 *
 * Run with:
 *   bun run test:e2e
 *   # or explicitly:
 *   RUN_E2E=1 bun test src/model/llm.e2e.test.ts --timeout 120000
 */

import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { getOllamaModels } from '@/utils/ollama.js';
import { callLlm } from '@/model/llm.js';

const PREFERRED_MODEL = 'ollama:nemotron-3-super:cloud';

// ---------------------------------------------------------------------------
// Suite state — resolved once, shared across all tests
// ---------------------------------------------------------------------------

let resolvedModel: string | null = null;

beforeAll(async () => {
  if (!RUN_E2E) return;
  const models = await getOllamaModels();
  if (models.includes('nemotron-3-super:cloud')) {
    resolvedModel = PREFERRED_MODEL;
  } else {
    const cloud = models.find((m) => m.includes(':cloud'));
    resolvedModel = cloud ? `ollama:${cloud}` : null;
  }
});

// ---------------------------------------------------------------------------
// Helper — extract text from LlmResult regardless of thinking-model format
// ---------------------------------------------------------------------------

function extractText(response: Awaited<ReturnType<typeof callLlm>>['response']): string {
  if (typeof response === 'string') return response;
  const content = response.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((b): b is { type: string; text: string } => typeof b === 'object' && b !== null && b.type === 'text')
      .map((b) => b.text)
      .join('\n');
  }
  return '';
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Ollama cloud model — live callLlm', () => {
  e2eIt(
    'callLlm returns a non-empty response for a simple ping prompt',
    async () => {
      if (!resolvedModel) {
        console.log('No Ollama cloud model available — skipping');
        return;
      }

      const { response } = await callLlm(
        'Reply with exactly the word PONG and nothing else.',
        {
          model: resolvedModel,
          systemPrompt: 'You are a minimal test assistant. Follow instructions exactly.',
          timeoutMs: 60_000,
          thinkOverride: false,
        },
      );

      const text = extractText(response);
      expect(text.trim().length).toBeGreaterThan(0);
      expect(text.toUpperCase()).toContain('PONG');
    },
    90_000,
  );

  e2eIt(
    'nemotron-3-super:cloud answers a simple arithmetic question',
    async () => {
      const models = await getOllamaModels();
      if (!models.includes('nemotron-3-super:cloud')) {
        console.log('nemotron-3-super:cloud not available — skipping');
        return;
      }

      const { response } = await callLlm(
        'What is 2 + 2? Reply with only the number.',
        {
          model: PREFERRED_MODEL,
          systemPrompt: 'Answer with a single number, no explanation.',
          timeoutMs: 60_000,
        },
      );

      const text = extractText(response);
      expect(text.trim()).toMatch(/4/);
    },
    90_000,
  );

  e2eIt(
    'callLlm usage object contains token counts when available',
    async () => {
      if (!resolvedModel) {
        console.log('No Ollama cloud model available — skipping');
        return;
      }

      const { response, usage } = await callLlm('Say hello in one word.', {
        model: resolvedModel,
        thinkOverride: false,
        timeoutMs: 60_000,
      });

      const text = extractText(response);
      expect(text.trim().length).toBeGreaterThan(0);

      if (usage) {
        if (usage.inputTokens !== undefined) expect(typeof usage.inputTokens).toBe('number');
        if (usage.outputTokens !== undefined) expect(typeof usage.outputTokens).toBe('number');
      }
    },
    90_000,
  );

  e2eIt(
    'callLlm rejects with a clear [Ollama API] error for a non-existent model',
    async () => {
      await expect(
        callLlm('ping', {
          model: 'ollama:this-model-does-not-exist-xyzzy:cloud',
          timeoutMs: 30_000,
        }),
      ).rejects.toThrow(/Ollama/i);
    },
    45_000,
  );
});
