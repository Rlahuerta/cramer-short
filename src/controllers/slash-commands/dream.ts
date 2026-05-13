import { runDream, shouldRunDream } from '../../memory/dream.js';
import { MemoryStore } from '../../memory/store.js';
import type { HistoryItem } from '../types.js';

interface DreamSlashCommandOptions {
  readonly history: () => HistoryItem[];
  readonly flushedItems: WeakSet<HistoryItem>;
  readonly flushItemToScrollback: (item: HistoryItem) => void;
  readonly isAgentProcessing: () => boolean;
  readonly isDreamRunning: () => boolean;
  readonly setDreamRunning: (running: boolean) => void;
  readonly currentModel: () => string;
  readonly setError: (message: string | null) => void;
  readonly refreshError: () => void;
  readonly requestRender: () => void;
  readonly setStatus: (message: string) => void;
  readonly setTimeoutFn?: typeof setTimeout;
  readonly setIntervalFn?: typeof setInterval;
  readonly clearIntervalFn?: typeof clearInterval;
}

export async function handleDreamSlashCommand(query: string, options: DreamSlashCommandOptions): Promise<boolean> {
  if (!query.startsWith('/dream')) {
    return false;
  }

  // --- /dream show — show dream status without running ---
  if (query.trim() === '/dream show' || query.trim() === '/dream status') {
    const dreamStore = new MemoryStore();
    const [meta, dailyFiles] = await Promise.all([
      dreamStore.readDreamMeta(),
      dreamStore.listDailyFiles(),
    ]);
    const lastRun = meta?.lastRunAt
      ? new Date(meta.lastRunAt).toLocaleString()
      : 'Never';
    const elapsedH = meta?.lastRunAt
      ? Math.floor((Date.now() - meta.lastRunAt) / 3_600_000)
      : null;
    const elapsedLabel = elapsedH !== null
      ? elapsedH >= 24 ? `${Math.floor(elapsedH / 24)}d ${elapsedH % 24}h ago` : `${elapsedH}h ago`
      : '—';
    const ready = shouldRunDream(meta, dailyFiles);
    const statusIcon = ready ? '✅' : '⏳';
    const statusLabel = ready ? 'Ready to consolidate' : 'Conditions not yet met';
    const needFiles = Math.max(0, 2 - dailyFiles.length);
    const needSessions = meta ? Math.max(0, 3 - (meta.sessionsSinceLastRun ?? 0)) : 3;
    const needHours = meta?.lastRunAt
      ? Math.max(0, 24 - Math.floor((Date.now() - meta.lastRunAt) / 3_600_000))
      : 0;
    const conditions: string[] = [];
    if (needFiles > 0) conditions.push(`${dailyFiles.length}/2 daily files`);
    if (needSessions > 0) conditions.push(`${meta?.sessionsSinceLastRun ?? 0}/3 sessions`);
    if (needHours > 0) conditions.push(`${needHours}h until 24h interval`);
    const condText = conditions.length > 0
      ? `\n\n_Waiting on: ${conditions.join(' · ')}_`
      : '';
    const fileList = dailyFiles.length > 0
      ? dailyFiles.map((f) => `  • ${f}`).join('\n')
      : '  _(none yet — exit Cramer-Short after conversations to generate them)_';
    const showAnswer = [
      `🌙 **Dream Status**`,
      ``,
      `${statusIcon} **${statusLabel}**${condText}`,
      ``,
      `| Field | Value |`,
      `|---|---|`,
      `| Last run | ${lastRun} ${elapsedLabel !== '—' ? `(${elapsedLabel})` : ''} |`,
      `| Total runs | ${meta?.totalRuns ?? 0} |`,
      `| Sessions since last run | ${meta?.sessionsSinceLastRun ?? 0} / 3 required |`,
      `| Daily files available | ${dailyFiles.length} / 2 required |`,
      ``,
      `**Daily files:**`,
      fileList,
      ``,
      `_Run \`/dream force\` to consolidate regardless of conditions._`,
    ].join('\n');
    options.flushItemToScrollback({
      id: `dream-show-${Date.now()}`,
      query,
      events: [],
      answer: showAnswer,
      status: 'complete',
      duration: 0,
    });
    return true;
  }

  if (options.isAgentProcessing() || options.isDreamRunning()) {
    options.setError(options.isDreamRunning()
      ? 'Dream is already running.'
      : 'Cannot run Dream while the agent is busy.');
    options.refreshError();
    options.requestRender();
    return true;
  }
  const force = query.slice('/dream'.length).trim() === 'force';
  const dreamStore = new MemoryStore();
  options.setDreamRunning(true);
  options.setStatus('🌙 Dream: consolidating memories…');
  options.requestRender();
  const dreamStart = Date.now();
  // Keep TUI alive during the long LLM consolidation call (can take 2-5 min).
  const interval = options.setIntervalFn ?? setInterval;
  const clear = options.clearIntervalFn ?? clearInterval;
  const dreamHeartbeat = interval(() => options.requestRender(), 1500);
  let dreamAnswer = '';
  try {
    const dreamResult = await runDream(dreamStore, options.currentModel(), { force });
    if (dreamResult.ran) {
      const n = dreamResult.archivedFiles.length;
      const files = dreamResult.archivedFiles.map((f) => `  • ${f}`).join('\n');
      const archiveLine = n > 0
        ? `**Archived ${n} daily file${n === 1 ? '' : 's'}:**\n${files}\n\n`
        : `_No daily session files to archive — exit Cramer-Short (ctrl+c) after conversations to generate them._\n\n`;
      dreamAnswer = `✨ **Dream complete** — memory consolidated\n\n${archiveLine}**Updated:** MEMORY.md, FINANCE.md`;
      options.setStatus(`✨ Dream: archived ${n} file${n === 1 ? '' : 's'}, memory updated`);
    } else {
      dreamAnswer = `🌙 **Dream skipped**\n\n${dreamResult.reason}\n\nUse \`/dream force\` to run regardless of conditions.`;
      options.setStatus(`🌙 Dream: ${dreamResult.reason}`);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    dreamAnswer = `❌ **Dream error**\n\n${msg}`;
    options.setStatus(`Dream error: ${msg}`);
  } finally {
    options.setDreamRunning(false);
    clear(dreamHeartbeat);
  }
  // Flush previous agent exchange to scrollback before rendering the dream answer.
  const prevItem = options.history().at(-1);
  if (prevItem && (prevItem.status === 'complete' || prevItem.status === 'interrupted') && !options.flushedItems.has(prevItem)) {
    options.flushItemToScrollback(prevItem);
    options.flushedItems.add(prevItem);
    // Small yield: let TUI settle before the second flush.
    await new Promise<void>((resolve) => setTimeout(resolve, 50));
  }
  options.flushItemToScrollback({
    id: `dream-${dreamStart}`,
    query,
    events: [],
    answer: dreamAnswer,
    status: 'complete',
    duration: Date.now() - dreamStart,
  });
  const timeout = options.setTimeoutFn ?? setTimeout;
  timeout(() => { options.setStatus(options.currentModel()); options.requestRender(); }, 5000);
  return true;
}
