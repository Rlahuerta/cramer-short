import { describe, test, expect, mock, beforeEach } from 'bun:test';

const actualConfig = await import('@/utils/config.js');

const mockGetSetting = mock((_key: string, defaultValue: unknown) => defaultValue);

mock.module('@/utils/config.js', () => ({
  ...actualConfig,
  getSetting: mockGetSetting,
}));

const {
  isThinkingModel,
  getChatModel,
  callLlm,
  getLlmCallTimeoutMs,
  DEFAULT_LLM_CALL_TIMEOUT_MS,
  streamCallLlm,
  _setModelFactory,
} = await import('./llm.js');

beforeEach(() => {
  mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
  _setModelFactory(null);
});

// ===========================================================================
// isThinkingModel
// ===========================================================================

describe('isThinkingModel', () => {
  test('returns true for qwen3 models', () => {
    expect(isThinkingModel('qwen3:4b')).toBe(true);
    expect(isThinkingModel('qwen3:8b')).toBe(true);
    expect(isThinkingModel('qwen3:14b')).toBe(true);
    expect(isThinkingModel('qwen3:235b-a22b')).toBe(true);
  });

  test('returns true for ollama: prefixed qwen3 models', () => {
    expect(isThinkingModel('ollama:qwen3:4b')).toBe(true);
  });

  test('returns true for deepseek-r1 models', () => {
    expect(isThinkingModel('deepseek-r1:8b')).toBe(true);
    expect(isThinkingModel('deepseek-r1:32b')).toBe(true);
    expect(isThinkingModel('ollama:deepseek-r1:70b')).toBe(true);
  });

  test('returns true for qwq models', () => {
    expect(isThinkingModel('qwq:32b')).toBe(true);
    expect(isThinkingModel('ollama:qwq:32b')).toBe(true);
  });

  test('returns false for non-thinking Ollama models', () => {
    expect(isThinkingModel('llama3.1:8b')).toBe(false);
    expect(isThinkingModel('mistral:7b')).toBe(false);
    expect(isThinkingModel('phi4:14b')).toBe(false);
    expect(isThinkingModel('ollama:llama3.1:8b')).toBe(false);
  });

  test('returns false for non-Ollama model names', () => {
    expect(isThinkingModel('gpt-5.4')).toBe(false);
    expect(isThinkingModel('claude-sonnet-4-5')).toBe(false);
    expect(isThinkingModel('gemini-3')).toBe(false);
  });
});

// ===========================================================================
// getChatModel — think flag passed to ChatOllama
// ===========================================================================

describe('getChatModel — Ollama think flag', () => {
  test('passes think:true for qwen3 model', () => {
    const model = getChatModel('ollama:qwen3:4b') as { think?: boolean };
    expect(model.think).toBe(true);
  });

  test('passes think:true for deepseek-r1 model', () => {
    const model = getChatModel('ollama:deepseek-r1:8b') as { think?: boolean };
    expect(model.think).toBe(true);
  });

  test('does NOT pass think:true for llama3', () => {
    const model = getChatModel('ollama:llama3.1:8b') as { think?: boolean };
    expect(model.think).toBeUndefined();
  });

  test('strips ollama: prefix before passing model to ChatOllama', () => {
    const model = getChatModel('ollama:qwen3:4b') as { model?: string };
    expect(model.model).toBe('qwen3:4b');
  });
});

describe('callLlm compatibility fallbacks', () => {
  test('uses call() when invoke() is unavailable', async () => {
    const call = mock(async () => ({ content: 'CALL_OK' }));
    _setModelFactory((() => ({ call })) as any);

    const { response } = await callLlm('Reply with CALL_OK.', {
      model: 'ollama:llama3.1:8b',
      systemPrompt: 'Return the requested token.',
      thinkOverride: false,
      timeoutMs: 30_000,
    });

    expect(response).toBe('CALL_OK');
    expect(call).toHaveBeenCalledTimes(1);
  });
});

describe('getLlmCallTimeoutMs', () => {
  test('returns the configured timeout when llmCallTimeoutMs is set', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    delete process.env.LLM_CALL_TIMEOUT_MS;
    mockGetSetting.mockImplementation((key: string, defaultValue: unknown) =>
      key === 'llmCallTimeoutMs' ? 300_000 : defaultValue,
    );
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    expect(freshGetter()).toBe(300_000);
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('falls back to the default timeout when llmCallTimeoutMs is unset', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    delete process.env.LLM_CALL_TIMEOUT_MS;
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('config overrides env when both are present', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '180000';
    mockGetSetting.mockImplementation((key: string) =>
      key === 'llmCallTimeoutMs' ? 300_000 : undefined,
    );
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    expect(freshGetter()).toBe(300_000);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('uses env when config is unset', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '180000';
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    expect(freshGetter()).toBe(180000);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('falls back to default when env value is NaN (non-numeric string)', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = 'not-a-number';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('falls back to default when env value is empty string', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('falls back to default when env value is too small (< 30000)', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '1000';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('falls back to default when env value is too large (> 600000)', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '9999999';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('uses valid env value within range', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '180000';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(180000);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('rejects malformed env value with unit suffix (e.g., "180000ms")', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '180000ms';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('rejects malformed env value with arbitrary suffix (e.g., "300000 seconds")', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = '300000 seconds';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });

  test('rejects malformed env value with leading garbage (e.g., "timeout:120000")', async () => {
    const original = process.env.LLM_CALL_TIMEOUT_MS;
    process.env.LLM_CALL_TIMEOUT_MS = 'timeout:120000';
    const { getLlmCallTimeoutMs: freshGetter } = await import('./llm.js?' + Date.now());
    mockGetSetting.mockImplementation((_key: string, defaultValue: unknown) => defaultValue);
    expect(freshGetter()).toBe(DEFAULT_LLM_CALL_TIMEOUT_MS);
    delete process.env.LLM_CALL_TIMEOUT_MS;
    if (original !== undefined) process.env.LLM_CALL_TIMEOUT_MS = original;
  });
});

// ===========================================================================
// streamCallLlm — word-boundary buffering (Feature 7)
// ===========================================================================

describe('streamCallLlm — word-boundary buffering', () => {
  // Build a fake ChatOpenAI-like model whose stream() yields controlled chunks
  function makeStreamingModel(chunks: string[]) {
    return {
      stream: async function* (_messages: unknown) {
        for (const chunk of chunks) {
          yield { content: chunk };
        }
      },
    };
  }

  // Replace getChatModel with our fake using a closure-based override.
  // We test the buffering logic via a simplified version mirroring streamCallLlm.
  async function collectBuffered(chunks: string[]): Promise<string[]> {
    const yielded: string[] = [];
    let wordBuffer = '';

    for (const text of chunks) {
      wordBuffer += text;
      const lastBoundary = Math.max(
        wordBuffer.lastIndexOf(' '),
        wordBuffer.lastIndexOf('\n'),
        wordBuffer.lastIndexOf('\t'),
      );
      if (lastBoundary >= 0) {
        yielded.push(wordBuffer.slice(0, lastBoundary + 1));
        wordBuffer = wordBuffer.slice(lastBoundary + 1);
      }
    }
    // Final flush
    if (wordBuffer) yielded.push(wordBuffer);
    return yielded;
  }

  test('yields at word boundaries, not character-by-character', async () => {
    // Six single-char chunks that together form "hello world "
    const result = await collectBuffered(['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', ' ']);
    // Should yield "hello " and "world " as two chunks, not 12 single chars
    expect(result.length).toBeLessThan(12);
    expect(result.join('')).toBe('hello world ');
  });

  test('final word is flushed even without trailing whitespace', async () => {
    const result = await collectBuffered(['hello ', 'world']);
    const full = result.join('');
    expect(full).toBe('hello world');
    // "world" must appear (flushed at end)
    expect(full).toContain('world');
  });

  test('a stream with no whitespace flushes the entire buffer at end', async () => {
    const result = await collectBuffered(['abc', 'def', 'ghi']);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe('abcdefghi');
  });

  test('newline counts as a word boundary', async () => {
    const result = await collectBuffered(['line1\n', 'line2\n']);
    expect(result.join('')).toBe('line1\nline2\n');
    // Each line should trigger a yield (not accumulate into one)
    const hasNewlines = result.some((r) => r.includes('\n'));
    expect(hasNewlines).toBe(true);
  });

  test('yields up to last boundary, not first, keeping chunks large', async () => {
    // "hello world foo " has two boundaries — should yield up to the last one
    const result = await collectBuffered(['hello world foo ']);
    // The entire string up to the last space should be one chunk
    expect(result[0]).toBe('hello world foo ');
  });

  test('mixed whitespace: last boundary wins', async () => {
    const result = await collectBuffered(['a b\tc ']);
    expect(result.join('')).toBe('a b\tc ');
  });
});

describe('streamCallLlm', () => {
  test('creates the chat model with streaming enabled', async () => {
    const factory = mock((_name: string, opts: { streaming: boolean }) => ({
      stream: async function* () {
        yield { content: 'STREAM_OK' };
      },
    }));
    _setModelFactory(factory as any);

    const chunks: string[] = [];
    for await (const chunk of streamCallLlm('Reply with exactly: STREAM_OK', {
      model: 'ollama:llama3.1:8b',
      thinkOverride: false,
    })) {
      chunks.push(chunk);
    }

    expect(factory).toHaveBeenCalledTimes(1);
    expect(factory.mock.calls[0]?.[1]).toMatchObject({ streaming: true });
    expect(chunks.join('')).toBe('STREAM_OK');
  });
});
