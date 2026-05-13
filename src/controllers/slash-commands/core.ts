import chalk from 'chalk';
import { isThinkingModel } from '../../model/llm.js';
import { MemoryStore } from '../../memory/store.js';
import { discoverSkills } from '../../skills/registry.js';
import type { SkillMetadata } from '../../skills/types.js';
import { searchHistory } from '../../utils/chat-search.js';
import { exportSession } from '../../utils/export.js';
import type { SessionIndexEntry } from '../../utils/session-store.js';
import type { HistoryItem } from '../types.js';
import { handleConfigSlashCommand } from './config.js';

interface SlashChatLog {
  clearAll(): void;
  addQuery(query: string): void;
  resetToolGrouping(): void;
  finalizeAnswer(answer: string): void;
}

interface CoreSlashCommandOptions {
  readonly chatLog: SlashChatLog;
  readonly history: () => HistoryItem[];
  readonly currentModel: () => string;
  readonly getThinkEnabled: () => boolean | null;
  readonly setThinkEnabled: (value: boolean | null) => void;
  readonly setAgentThinkEnabled: (value: boolean | undefined) => void;
  readonly startModelSelection: () => void;
  readonly listSessions: () => Promise<SessionIndexEntry[]>;
  readonly setSessionsList: (sessions: SessionIndexEntry[]) => void;
  readonly setSessionsVisible: (visible: boolean) => void;
  readonly setSkillsList: (skills: SkillMetadata[]) => void;
  readonly setSkillsVisible: (visible: boolean) => void;
  readonly setHelpVisible: (visible: boolean) => void;
  readonly setMemoryVisible: (visible: boolean) => void;
  readonly isMemoryVisible: () => boolean;
  readonly setMemoryContent: (content: { memory: string; finance: string } | null) => void;
  readonly setError: (message: string | null) => void;
  readonly refreshError: () => void;
  readonly requestRender: () => void;
  readonly renderSelectionOverlay: () => void;
  readonly openAnswerViewer: (content: string) => void;
  readonly setStatus: (message: string) => void;
  readonly setTimeoutFn?: typeof setTimeout;
}

interface ExitCommandOptions {
  readonly history: () => HistoryItem[];
  readonly currentModel: () => string;
  readonly safeStopTui: () => void;
  readonly writeSessionDailySummary: (history: { query: string; answer: string }[], model: string) => Promise<void>;
  readonly flushSession: () => Promise<void>;
  readonly exitProcess?: (code?: number) => never;
}

function showChatAnswer(query: string, answer: string, options: Pick<CoreSlashCommandOptions, 'chatLog' | 'requestRender'>): void {
  options.chatLog.clearAll();
  options.chatLog.addQuery(query);
  options.chatLog.resetToolGrouping();
  options.chatLog.finalizeAnswer(answer);
  options.requestRender();
}

export async function handleExitCommand(query: string, options: ExitCommandOptions): Promise<boolean> {
  if (query.toLowerCase() !== 'exit' && query.toLowerCase() !== 'quit') {
    return false;
  }

  options.safeStopTui();
  await options.writeSessionDailySummary(options.history(), options.currentModel());
  await options.flushSession();
  (options.exitProcess ?? process.exit)(0);
  return true;
}

export async function handleCoreSlashCommand(query: string, options: CoreSlashCommandOptions): Promise<boolean> {
  if (query === '/help') {
    options.setHelpVisible(true);
    options.renderSelectionOverlay();
    return true;
  }

  if (query === '/full' || query === '/viewer') {
    const completedItem = [...options.history()].reverse().find((item) => item.status === 'complete' && item.answer?.trim());
    if (!completedItem?.answer?.trim()) {
      options.setError('No completed answer available to view.');
      options.refreshError();
      options.requestRender();
      return true;
    }
    options.setError(null);
    options.openAnswerViewer(completedItem.answer);
    return true;
  }

  if (query === '/think') {
    const model = options.currentModel();
    if (!isThinkingModel(model)) {
      options.setError(`${model} does not support extended thinking (supported: qwen3, deepseek-r1, qwq)`);
      options.refreshError();
      options.requestRender();
    } else {
      // Cycle: auto(on) → off → auto(on)
      // null  = auto (effective: on for thinking models)
      // false = forced off
      const wasOff = options.getThinkEnabled() === false;
      const nextThinkEnabled = wasOff ? null : false;
      options.setThinkEnabled(nextThinkEnabled);
      options.setAgentThinkEnabled(nextThinkEnabled ?? undefined);
      const label = nextThinkEnabled === false ? '🔕 Thinking OFF' : '🧠 Thinking ON (auto)';
      options.setError(null);
      // Show brief status via the intro line (reuses existing text component path)
      options.setStatus(`${model}  ${label}`);
      options.renderSelectionOverlay();
      options.requestRender();
      // Restore normal model label after 3 s
      const timeout = options.setTimeoutFn ?? setTimeout;
      timeout(() => {
        options.setStatus(model);
        options.requestRender();
      }, 3000);
    }
    return true;
  }

  if (query === '/model') {
    options.startModelSelection();
    return true;
  }

  if (query === '/sessions') {
    options.setSessionsList(await options.listSessions());
    options.setSessionsVisible(true);
    options.renderSelectionOverlay();
    options.requestRender();
    return true;
  }

  if (query === '/skills') {
    options.setSkillsList(discoverSkills());
    options.setSkillsVisible(true);
    options.renderSelectionOverlay();
    options.requestRender();
    return true;
  }

  if (query.startsWith('/export')) {
    const parts = query.split(/\s+/);
    const format = (parts[1] ?? 'markdown') as 'markdown' | 'json' | 'csv';
    const validFormats = ['markdown', 'json', 'csv'];
    if (!validFormats.includes(format)) {
      options.setError(`Invalid export format "${format}". Use: markdown, json, csv`);
      options.refreshError();
      options.requestRender();
      return true;
    }
    const exportHistory = options.history().filter((h) => h.status === 'complete');
    if (exportHistory.length === 0) {
      options.setError('No completed queries to export.');
      options.refreshError();
      options.requestRender();
      return true;
    }
    try {
      const { path } = exportSession(exportHistory, format, undefined);
      options.setError(null);
      options.setStatus(`✓ Exported to ${path}`);
      options.requestRender();
      const timeout = options.setTimeoutFn ?? setTimeout;
      timeout(() => { options.setStatus(options.currentModel()); options.requestRender(); }, 3000);
    } catch (e) {
      options.setError(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
      options.refreshError();
      options.requestRender();
    }
    return true;
  }

  if (query.startsWith('/find')) {
    const keyword = query.replace('/find', '').trim();
    if (!keyword) {
      showChatAnswer(query, 'Usage: /find <keyword> — searches chat history for matching queries and answers', options);
      return true;
    }
    const history = options.history().filter((h) => h.status === 'complete');
    if (history.length === 0) {
      showChatAnswer(query, 'No chat history in current session', options);
      return true;
    }
    const entries = history.map((h, idx) => ({ query: h.query, answer: h.answer, turn: idx + 1 }));
    const matches = searchHistory(entries, keyword);
    if (matches.length === 0) {
      showChatAnswer(query, `No matches found for "${keyword}" in current session`, options);
      return true;
    }
    const highlightKw = (text: string): string => {
      const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return text.replace(new RegExp(escaped, 'gi'), (m) => chalk.yellow(m));
    };
    const lines: string[] = [`Found ${matches.length} match${matches.length === 1 ? '' : 'es'} for "${keyword}":\n`];
    for (const m of matches) {
      const excerpt = m.answer.length > 200 ? `${m.answer.slice(0, 200)}...` : m.answer;
      lines.push(`[Turn ${m.turn}]  ${highlightKw(m.query)}`);
      lines.push(`...${highlightKw(excerpt)}...\n`);
    }
    showChatAnswer(query, lines.join('\n'), options);
    return true;
  }

  if (query === '/memory' || query === '/memory show') {
    options.setMemoryVisible(true);
    options.setMemoryContent(null); // loading
    options.renderSelectionOverlay();
    options.requestRender();
    // Load both files asynchronously then re-render.
    const memStore = new MemoryStore();
    Promise.all([
      memStore.readMemoryFile('MEMORY.md').catch(() => ''),
      memStore.readMemoryFile('FINANCE.md').catch(() => ''),
    ]).then(([memory, finance]) => {
      options.setMemoryContent({ memory: memory.trim(), finance: finance.trim() });
      if (options.isMemoryVisible()) { options.renderSelectionOverlay(); options.requestRender(); }
    });
    return true;
  }

  if (handleConfigSlashCommand(query, {
    chatLog: options.chatLog,
    currentModel: options.currentModel,
    refreshError: options.refreshError,
    requestRender: options.requestRender,
    setError: options.setError,
    setStatus: options.setStatus,
    setTimeoutFn: options.setTimeoutFn,
  })) {
    return true;
  }

  return false;
}
