import { describe, it, expect, beforeEach, afterEach, spyOn, mock, setSystemTime } from 'bun:test';
import { MemoryManager } from '../../memory/index.js';
import { memoryGetTool } from './memory-get.js';
import { memorySearchTool } from './memory-search.js';
import { memoryUpdateTool } from './memory-update.js';
import { recallFinancialContextTool } from './financial-recall.js';
import { storeFinancialInsightTool } from './financial-store.js';

/** formatToolResult wraps data as { data: ... } — unwrap for assertions */
function parseResult(result: string): unknown {
  return (JSON.parse(result) as { data: unknown }).data;
}

const FIXED_NOW = new Date('2026-01-15T12:00:00.000Z');
const FIXED_TODAY = FIXED_NOW.toISOString().slice(0, 10);
const FIXED_TIMESTAMP = FIXED_NOW.getTime();

// ---------------------------------------------------------------------------
// Shared spy lifecycle
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let managerSpy: ReturnType<typeof spyOn<any, any>> | undefined;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const managerSpies: Array<ReturnType<typeof spyOn<any, any>>> = [];

beforeEach(() => {
  setSystemTime(FIXED_NOW);
});

afterEach(() => {
  for (const spy of managerSpies.splice(0)) {
    spy.mockRestore();
  }
  managerSpy = undefined;
  setSystemTime();
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function setupManager(partial: Record<string, unknown>): void {
  const spy = spyOn(MemoryManager, 'get').mockResolvedValue(partial as never);
  managerSpy = spy;
  managerSpies.push(spy);
}

type FinancialInsightFixture = {
  id: number;
  ticker: string;
  content: string;
  tags: string[];
  updatedAt: number;
};

function financialInsight(overrides: Partial<FinancialInsightFixture> = {}): FinancialInsightFixture {
  return {
    id: 1,
    ticker: 'AAPL',
    content: 'test',
    tags: [],
    updatedAt: FIXED_TIMESTAMP,
    ...overrides,
  };
}

function financialStore(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    recallByTicker: () => [],
    search: () => [],
    getRouting: () => null,
    getRelatedInsights: () => [],
    ...overrides,
  };
}

function setupFinancialStore(overrides: Record<string, unknown> = {}): void {
  setupManager({ getFinancialStore: () => financialStore(overrides) });
}

function setupStoreFinancialInsight(id = 1): {
  mockStore: ReturnType<typeof mock>;
  mockAppend: ReturnType<typeof mock>;
} {
  const mockStore = mock(async () => id);
  const mockAppend = mock(async () => {});
  setupManager({
    getFinancialStore: () => ({ storeInsight: mockStore }),
    appendMemory: mockAppend,
  });
  return { mockStore, mockAppend };
}

function firstMockCall(fn: ReturnType<typeof mock>): unknown[] {
  return fn.mock.calls[0] as unknown[];
}

function storedInsightTags(mockStore: ReturnType<typeof mock>): string[] {
  return (firstMockCall(mockStore)[0] as { tags: string[] }).tags;
}

// ---------------------------------------------------------------------------
// memory_get
// ---------------------------------------------------------------------------

describe('memory_get', () => {
  it('reads a memory file by path', async () => {
    const mockGet = mock(async () => ({ path: 'MEMORY.md', content: 'Hello\nWorld', totalLines: 2 }));
    setupManager({ get: mockGet });

    const result = parseResult(await memoryGetTool.invoke({ path: 'MEMORY.md' }));
    expect((result as { path: string }).path).toBe('MEMORY.md');
  });

  it('passes from and lines through to manager.get', async () => {
    const mockGet = mock(async () => ({ path: 'MEMORY.md', content: 'line 5', totalLines: 50 }));
    setupManager({ get: mockGet });

    await memoryGetTool.invoke({ path: 'MEMORY.md', from: 5, lines: 2 });

    expect(firstMockCall(mockGet)[0]).toEqual({ path: 'MEMORY.md', from: 5, lines: 2 });
  });

  it('reads a daily log file', async () => {
    const mockGet = mock(async () => ({ path: '2026-01-01.md', content: 'daily note', totalLines: 1 }));
    setupManager({ get: mockGet });

    await memoryGetTool.invoke({ path: '2026-01-01.md' });
    expect((firstMockCall(mockGet)[0] as { path: string }).path).toBe('2026-01-01.md');
  });
});

// ---------------------------------------------------------------------------
// memory_search
// ---------------------------------------------------------------------------

describe('memory_search', () => {
  it('returns search results when memory is available', async () => {
    const mockSearch = mock(async () => [{ score: 0.9, content: 'Found memory' }]);
    setupManager({ isAvailable: () => true, search: mockSearch });

    const result = parseResult(await memorySearchTool.invoke({ query: 'test query' })) as {
      results: Array<{ content: string }>;
    };
    expect(result.results).toHaveLength(1);
    expect(result.results[0].content).toBe('Found memory');
  });

  it('calls search with the exact query string', async () => {
    const mockSearch = mock(async () => []);
    setupManager({ isAvailable: () => true, search: mockSearch });

    await memorySearchTool.invoke({ query: 'my specific query' });
    expect(firstMockCall(mockSearch)[0]).toBe('my specific query');
  });

  it('returns disabled error when memory is unavailable', async () => {
    setupManager({
      isAvailable: () => false,
      getUnavailableReason: () => 'Memory is disabled in settings.',
    });

    const result = parseResult(await memorySearchTool.invoke({ query: 'test' })) as {
      disabled: boolean;
      error: string;
    };
    expect(result.disabled).toBe(true);
    expect(result.error).toBe('Memory is disabled in settings.');
  });

  it('uses fallback error message when getUnavailableReason returns null', async () => {
    setupManager({ isAvailable: () => false, getUnavailableReason: () => null });

    const result = parseResult(await memorySearchTool.invoke({ query: 'test' })) as {
      disabled: boolean;
      error: string;
    };
    expect(result.disabled).toBe(true);
    expect(result.error).toBe('Memory search unavailable.');
  });

  it('returns empty results array when no matches found', async () => {
    setupManager({ isAvailable: () => true, search: mock(async () => []) });

    const result = parseResult(await memorySearchTool.invoke({ query: 'obscure' })) as {
      results: unknown[];
    };
    expect(result.results).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// memory_update — append
// ---------------------------------------------------------------------------

describe('memory_update — append', () => {
  let mockAppend: ReturnType<typeof mock>;

  beforeEach(() => {
    mockAppend = mock(async () => {});
    setupManager({ appendMemory: mockAppend });
  });

  it('appends content and resolves long_term to MEMORY.md', async () => {
    const result = parseResult(
      await memoryUpdateTool.invoke({ content: 'Test content', action: 'append', file: 'long_term' }),
    ) as { success: boolean; file: string };

    expect(result.success).toBe(true);
    expect(result.file).toBe('MEMORY.md');
    expect(firstMockCall(mockAppend)).toEqual(['long_term', 'Test content']);
  });

  it('resolves daily to today YYYY-MM-DD.md', async () => {
    const result = parseResult(
      await memoryUpdateTool.invoke({ content: 'Daily note', action: 'append', file: 'daily' }),
    ) as { success: boolean; file: string };

    const today = FIXED_TODAY;
    expect(result.success).toBe(true);
    expect(result.file).toBe(`${today}.md`);
  });

  it('passes through custom filename unchanged', async () => {
    const result = parseResult(
      await memoryUpdateTool.invoke({ content: 'x', action: 'append', file: '2025-01-01.md' }),
    ) as { file: string };

    expect(result.file).toBe('2025-01-01.md');
  });

  it('returns error when content is missing', async () => {
    const result = parseResult(
      await memoryUpdateTool.invoke({ action: 'append', file: 'long_term' }),
    ) as { success: boolean; error: string };

    expect(result.success).toBe(false);
    expect(result.error).toContain('"content" is required');
  });

  it('reports correct character count in message', async () => {
    const content = 'Hello World';
    const result = parseResult(
      await memoryUpdateTool.invoke({ content, action: 'append', file: 'long_term' }),
    ) as { message: string };

    expect(result.message).toContain(`${content.length} characters`);
  });
});

// ---------------------------------------------------------------------------
// memory_update — edit
// ---------------------------------------------------------------------------

describe('memory_update — edit', () => {
  it('edits memory and returns success', async () => {
    const mockEdit = mock(async () => true);
    setupManager({ editMemory: mockEdit });

    const result = parseResult(
      await memoryUpdateTool.invoke({
        action: 'edit',
        file: 'long_term',
        old_text: 'old value',
        new_text: 'new value',
      }),
    ) as { success: boolean };

    expect(result.success).toBe(true);
    expect(firstMockCall(mockEdit)).toEqual(['long_term', 'old value', 'new value']);
  });

  it('returns error when text is not found', async () => {
    setupManager({ editMemory: mock(async () => false) });

    const result = parseResult(
      await memoryUpdateTool.invoke({
        action: 'edit',
        file: 'long_term',
        old_text: 'nonexistent',
        new_text: 'replacement',
      }),
    ) as { success: boolean; error: string };

    expect(result.success).toBe(false);
    expect(result.error).toContain('Could not find');
  });

  it('returns error when old_text or new_text is missing', async () => {
    setupManager({ editMemory: mock(async () => true) });

    const result = parseResult(
      await memoryUpdateTool.invoke({ action: 'edit', file: 'long_term' }),
    ) as { success: boolean; error: string };

    expect(result.success).toBe(false);
    expect(result.error).toContain('"old_text" and "new_text" are required');
  });
});

// ---------------------------------------------------------------------------
// memory_update — delete
// ---------------------------------------------------------------------------

describe('memory_update — delete', () => {
  it('deletes memory entry and returns success', async () => {
    const mockDelete = mock(async () => true);
    setupManager({ deleteMemory: mockDelete });

    const result = parseResult(
      await memoryUpdateTool.invoke({ action: 'delete', file: 'long_term', old_text: 'Remove me' }),
    ) as { success: boolean };

    expect(result.success).toBe(true);
    expect(firstMockCall(mockDelete)).toEqual(['long_term', 'Remove me']);
  });

  it('returns error when text is not found', async () => {
    setupManager({ deleteMemory: mock(async () => false) });

    const result = parseResult(
      await memoryUpdateTool.invoke({ action: 'delete', file: 'long_term', old_text: 'ghost text' }),
    ) as { success: boolean; error: string };

    expect(result.success).toBe(false);
    expect(result.error).toContain('Could not find');
  });

  it('returns error when old_text is missing', async () => {
    setupManager({ deleteMemory: mock(async () => true) });

    const result = parseResult(
      await memoryUpdateTool.invoke({ action: 'delete', file: 'long_term' }),
    ) as { success: boolean; error: string };

    expect(result.success).toBe(false);
    expect(result.error).toContain('"old_text" is required');
  });
});

// ---------------------------------------------------------------------------
// recall_financial_context
// ---------------------------------------------------------------------------

describe('recall_financial_context', () => {
  it('returns unavailable message when financial store is null', async () => {
    setupManager({ getFinancialStore: () => null });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toBe('Financial memory not available.');
  });

  it('returns no-context message when no insights found', async () => {
    setupFinancialStore();

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toContain('No prior financial context found for AAPL');
  });

  it('includes namespace in no-context message', async () => {
    setupFinancialStore();

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL', namespace: 'dcf' });
    expect(result).toContain('No prior financial context found for AAPL [ns:dcf]');
  });

  it('returns insights when byTicker finds results', async () => {
    const insight = financialInsight({ content: 'Great company', tags: ['analysis:thesis'] });
    setupFinancialStore({ recallByTicker: () => [insight] });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toContain('Great company');
    expect(result).toContain('Stored insights for AAPL');
    expect(result).toContain('1 found');
  });

  it('includes routing hint when routing is set', async () => {
    const insight = financialInsight({ ticker: 'VWS.CO', content: 'Wind energy' });
    setupFinancialStore({
      recallByTicker: () => [insight],
      getRouting: () => 'fmp-premium',
    });

    const result = await recallFinancialContextTool.invoke({ ticker: 'VWS.CO' });
    expect(result).toContain('**Routing:** fmp-premium');
    expect(result).toContain('skip FMP');
  });

  it('merges byTicker and byQuery results, deduplicating by id', async () => {
    const shared = financialInsight({ content: 'Shared insight' });
    const unique = financialInsight({ id: 2, content: 'Query-only insight' });
    setupFinancialStore({
      recallByTicker: () => [shared],
      search: () => [shared, unique],
    });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL', query: 'analysis' });
    expect(result).toContain('2 found');
  });

  it('includes related insights in output', async () => {
    const related = {
      relation: 'peer',
      insight: { ticker: 'MSFT', content: 'Related company insight', tags: [] },
    };
    const insight = financialInsight({ content: 'Main insight' });
    setupFinancialStore({
      recallByTicker: () => [insight],
      getRelatedInsights: () => [related],
    });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toContain('↳ peer: MSFT');
  });

  it('includes namespace in result header', async () => {
    const insight = financialInsight({ content: 'DCF analysis' });
    setupFinancialStore({ recallByTicker: () => [insight] });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL', namespace: 'dcf' });
    expect(result).toContain('[ns:dcf]');
  });

  // routingHint branches
  it('routingHint fmp-ok → FMP free tier works', async () => {
    const insight = financialInsight();
    setupFinancialStore({
      recallByTicker: () => [insight],
      getRouting: () => 'fmp-ok',
    });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toContain('FMP free tier works');
  });

  it('routingHint yahoo-ok → Yahoo Finance works', async () => {
    const insight = financialInsight();
    setupFinancialStore({
      recallByTicker: () => [insight],
      getRouting: () => 'yahoo-ok',
    });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toContain('Yahoo Finance works');
  });

  it('routingHint web-fallback → all APIs failed', async () => {
    const insight = financialInsight();
    setupFinancialStore({
      recallByTicker: () => [insight],
      getRouting: () => 'web-fallback',
    });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toContain('all APIs failed');
  });

  it('routingHint unknown value → returns routing string as-is', async () => {
    const insight = financialInsight();
    setupFinancialStore({
      recallByTicker: () => [insight],
      getRouting: () => 'custom-source',
    });

    const result = await recallFinancialContextTool.invoke({ ticker: 'AAPL' });
    expect(result).toContain('custom-source');
  });
});

// ---------------------------------------------------------------------------
// store_financial_insight
// ---------------------------------------------------------------------------

describe('store_financial_insight', () => {
  it('returns unavailable message when financial store is null', async () => {
    setupManager({ getFinancialStore: () => null });

    const result = await storeFinancialInsightTool.invoke({ ticker: 'AAPL', content: 'Great company' });
    expect(result).toBe('Financial memory not available — insight not stored.');
  });

  it('stores insight and returns confirmation with id', async () => {
    setupStoreFinancialInsight(42);

    const result = await storeFinancialInsightTool.invoke({ ticker: 'AAPL', content: 'Great company' });
    expect(result).toContain('Stored insight #42 for AAPL');
  });

  it('automatically adds ticker tag (uppercased)', async () => {
    const { mockStore } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({ ticker: 'aapl', content: 'test' });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags).toContain('ticker:AAPL');
  });

  it('does not duplicate ticker tag if already present', async () => {
    const { mockStore } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({ ticker: 'AAPL', content: 'test', tags: ['ticker:AAPL'] });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags.filter((t) => t === 'ticker:AAPL').length).toBe(1);
  });

  it('adds routing tag and writes to FINANCE.md', async () => {
    const { mockStore, mockAppend } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({ ticker: 'AAPL', content: 'test', routing: 'fmp-ok' });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags).toContain('routing:fmp-ok');
    expect(firstMockCall(mockAppend)[0]).toBe('FINANCE.md');
    expect(firstMockCall(mockAppend)[1]).toContain('AAPL');
  });

  it('does not duplicate routing tag if already in tags', async () => {
    const { mockStore } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({
      ticker: 'AAPL',
      content: 'test',
      routing: 'fmp-ok',
      tags: ['routing:fmp-ok'],
    });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags.filter((t) => t.startsWith('routing:')).length).toBe(1);
  });

  it('adds sector tag (lowercased)', async () => {
    const { mockStore } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({ ticker: 'AAPL', content: 'test', sector: 'Technology' });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags).toContain('sector:technology');
  });

  it('adds exchange tag (uppercased)', async () => {
    const { mockStore } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({ ticker: 'VWS.CO', content: 'test', exchange: 'cph' });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags).toContain('exchange:CPH');
  });

  it('adds namespace tag', async () => {
    const { mockStore } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({ ticker: 'AAPL', content: 'test', namespace: 'dcf' });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags).toContain('ns:dcf');
  });

  it('includes namespace in return string', async () => {
    setupStoreFinancialInsight();

    const result = await storeFinancialInsightTool.invoke({
      ticker: 'AAPL',
      content: 'test',
      namespace: 'dcf',
    });
    expect(result).toContain('[ns:dcf]');
  });

  it('does not write to FINANCE.md when routing is not provided', async () => {
    const { mockAppend } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({ ticker: 'AAPL', content: 'test' });
    expect(mockAppend).not.toHaveBeenCalled();
  });

  it('includes exchange in FINANCE.md entry', async () => {
    const { mockAppend } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({
      ticker: 'VWS.CO',
      content: 'Routing note',
      routing: 'fmp-premium',
      exchange: 'CPH',
    });
    expect(firstMockCall(mockAppend)[1]).toContain('(CPH)');
  });

  it('includes all passed tags alongside auto-generated ones', async () => {
    const { mockStore } = setupStoreFinancialInsight();

    await storeFinancialInsightTool.invoke({
      ticker: 'AAPL',
      content: 'test',
      tags: ['analysis:thesis', 'analysis:risk'],
    });
    const storedTags: string[] = storedInsightTags(mockStore);
    expect(storedTags).toContain('analysis:thesis');
    expect(storedTags).toContain('analysis:risk');
    expect(storedTags).toContain('ticker:AAPL');
  });
});
