import { Container, ProcessTerminal, Spacer, Text, TUI } from '@mariozechner/pi-tui';
import { AtPathAutocompleteProvider } from './components/at-path-provider.js';
import type { ApprovalDecision } from './agent/index.js';
import { DEFAULT_MAX_ITERATIONS } from './agent/index.js';
import { isThinkingModel } from './model/llm.js';
import {
  AgentRunnerController,
  fetchLivePrices,
  fetchShowData,
  InputHistoryController,
  makePriceFetcher,
  ModelSelectionController,
  SessionController,
  WatchlistController,
  writeSessionDailySummary,
  type PriceSnapshot,
  type TuiStateController,
  type WatchlistEntry,
} from './controllers/index.js';
import { MemoryStore, incrementDreamSessionCount, runDream, seedWatchlistEntries, shouldRunDream } from './memory/index.js';
import {
  ApiKeyInputComponent,
  ApprovalPromptComponent,
  ChatLogComponent,
  CustomEditor,
  DebugPanelComponent,
  FullAnswerViewerComponent,
  IntroComponent,
  WorkingIndicatorComponent,
  createApiKeyConfirmSelector,
  createModelSelector,
  createProviderSelector,
  createSessionSelector,
  createSkillSelector,
} from './components/index.js';
import { editorTheme, theme } from './theme.js';
import type { HistoryItem } from './controllers/types.js';
import {
  exportSession,
  getApiKeyNameForProvider,
  getProviderDisplayName,
  logger,
  logError,
  type SessionIndexEntry,
} from './utils/index.js';
import type { SkillMetadata } from './skills/types.js';
import {
  SLASH_COMMANDS,
  buildHelpPanel,
  buildSnapshotPanel,
  buildShowPanel,
  buildWatchlistPanel,
  createScreen,
  handleCoreSlashCommand,
  handleDreamSlashCommand,
  handleExitCommand,
  handleWatchlistSlashCommand,
} from './controllers/slash-commands/index.js';
import { flushExchangeToScrollback, renderCurrentQuery } from './controllers/cli-rendering.js';

export {
  buildHelpPanel,
  buildSnapshotPanel,
  buildShowPanel,
  buildWatchlistPanel,
  colorPct,
  createScreen,
  fmtMoney,
  fmtPct,
} from './controllers/slash-commands/panels.js';

export {
  flushExchangeToScrollback,
  summarizeToolResult,
  truncateAtWord,
} from './controllers/cli-rendering.js';
export { writeSessionDailySummary } from './controllers/session-summary.js';

/** Max agent iterations when `--deep` flag is active. */
export const DEEP_MAX_ITERATIONS = 40;

/**
 * Returns the max agent iterations to use, based on whether `--deep` is present in argv.
 * Accepts an explicit argv array so it can be unit-tested without touching process.argv.
 */
export function resolveMaxIterations(argv: string[] = process.argv): number {
  return argv.includes('--deep') ? DEEP_MAX_ITERATIONS : DEFAULT_MAX_ITERATIONS;
}

export async function runCli() {
  // --deep flag: raises max agent iterations for complex multi-skill queries
  const maxIterations = resolveMaxIterations();

  // --export [path] flag: auto-export the session as Markdown on exit.
  // undefined = flag not present (no export); null = flag present, auto-generate filename; string = explicit path.
  const exportArgIdx = process.argv.indexOf('--export');
  const exportPathArg: string | null | undefined =
    exportArgIdx === -1
      ? undefined
      : process.argv[exportArgIdx + 1] && !process.argv[exportArgIdx + 1].startsWith('-')
        ? process.argv[exportArgIdx + 1]
        : null;

  const tui = new TUI(new ProcessTerminal());
  const root = new Container();
  const chatLog = new ChatLogComponent(tui);
  const inputHistory = new InputHistoryController(() => tui.requestRender());
  let lastError: string | null = null;
  let answerViewerVisible = false;
  let answerViewerContent = '';
  let helpVisible = false;
  let sessionsVisible = false;
  let sessionsList: SessionIndexEntry[] = [];
  let resumedSessionName: string | null = null;
  let sessionStarted = false;
  let memoryVisible = false;

  // Watchlist state
  let watchlistVisible = false;
  let watchlistEntries: WatchlistEntry[] = [];
  let watchlistPrices: Map<string, PriceSnapshot> | null = null;
  let watchlistMode: 'list' | 'show' | 'snapshot' = 'list';
  let watchlistShowTicker: string | null = null;

  // Agent / Thinking state
  let thinkEnabled: boolean | null = null; // null = auto
  let dreamRunning = false;

  // null = loading; string = content (may be empty)
  let memoryContent: { memory: string; finance: string } | null = null;
  // Skills overlay state
  let skillsVisible = false;
  let skillsList: SkillMetadata[] = [];
  // Watchlist auto-refresh state
  let watchlistRefreshIntervalMs: number = (() => {
    const env = parseInt(process.env.WATCHLIST_REFRESH_INTERVAL_MS ?? '0', 10);
    return Number.isFinite(env) && env >= 1000 ? env : 0;
  })();
  let watchlistRefreshTimer: ReturnType<typeof setInterval> | null = null;
  let watchlistLastRefresh: Date | null = null;
  let tuiStopped = false;
  const safeStopTui = () => {
    if (!tuiStopped) {
      tui.stop();
      tuiStopped = true;
    }
  };
  // Tracks exchanges already flushed to scrollback.
  // Prevents double-writing when the user starts a new query.
  const flushedItems = new WeakSet<HistoryItem>();

  const sessionController = new SessionController();

  const onError = (message: string) => {
    lastError = message;
    logger.error(message);
    tui.requestRender();
  };

  const modelSelection = new ModelSelectionController(onError, () => {
    // Reset thinking override when the user switches models — the new model may
    // or may not support thinking, so auto-detect is the correct default.
    thinkEnabled = null;
    agentRunner.setThinkEnabled(undefined);
    agentRunner.setModel(modelSelection.model, modelSelection.provider);
    intro.setModel(modelSelection.model);
    renderSelectionOverlay();
    tui.requestRender();
  });

  const agentRunner = new AgentRunnerController(
    { model: modelSelection.model, modelProvider: modelSelection.provider, maxIterations },
    modelSelection.inMemoryChatHistory,
    () => {
      renderCurrentQuery(chatLog, agentRunner.history);
      workingIndicator.setState(agentRunner.workingState);
      renderSelectionOverlay();
      tui.requestRender();
    },
  );

  const intro = new IntroComponent(modelSelection.model);
  const errorText = new Text('', 0, 0);
  const workingIndicator = new WorkingIndicatorComponent(tui);
  const editor = new CustomEditor(tui, editorTheme);
  const debugPanel = new DebugPanelComponent(8, process.env.DEBUG === 'true' || process.env.DEBUG === '1');

  // Detect fd/fdfind for fuzzy @ completion; falls back to readdirSync if absent.
  const fdPath = Bun.which('fd') ?? Bun.which('fdfind') ?? null;
  editor.setAutocompleteProvider(new AtPathAutocompleteProvider(SLASH_COMMANDS, process.cwd(), fdPath));

  tui.addChild(root);

  const refreshError = () => {
    const message = lastError ?? agentRunner.error;
    errorText.setText(message ? theme.error(`Error: ${message}`) : '');
  };

  // Refresh watchlist prices and Markov confidence
  const refreshWatchlistPrices = async () => {
    if (!watchlistVisible || watchlistEntries.length === 0) return;
    
    watchlistLastRefresh = new Date();
    renderSelectionOverlay();
    tui.requestRender();

    if (watchlistMode === 'show' && watchlistShowTicker) {
      const snap = await fetchShowData(watchlistShowTicker);
      watchlistPrices = snap ? new Map([[watchlistShowTicker, snap]]) : new Map();
    } else {
      const prices = await fetchLivePrices(
        watchlistEntries.map((e) => e.ticker),
        makePriceFetcher(),
      );
      watchlistPrices = prices;
    }

    if (watchlistVisible) {
      renderSelectionOverlay();
      tui.requestRender();
    }
  };

  // Start/stop auto-refresh timer
  const setWatchlistRefresh = (intervalMs: number) => {
    if (watchlistRefreshTimer) {
      clearInterval(watchlistRefreshTimer);
      watchlistRefreshTimer = null;
    }
    watchlistRefreshIntervalMs = intervalMs;
    if (intervalMs > 0 && watchlistVisible) {
      watchlistRefreshTimer = setInterval(refreshWatchlistPrices, intervalMs);
    }
  };

  const watchlistRefreshSuffix = () => {
    if (watchlistRefreshIntervalMs <= 0 || !watchlistLastRefresh) return '';
    const secs = Math.floor((Date.now() - watchlistLastRefresh.getTime()) / 1000);
    if (secs < 5) return ' · refreshed just now';
    if (secs < 60) return ` · refreshed ${secs}s ago`;
    return ` · refreshed ${Math.floor(secs / 60)}m ago`;
  };

  const tuiState: TuiStateController = {
    cwd: () => process.cwd(),
    history: () => agentRunner.history,
    flushedItems,
    flushItemToScrollback: (item) => flushExchangeToScrollback(tui, chatLog, item),
    currentModel: () => modelSelection.model,
    setStatus: (message) => intro.setModel(message),
    setError: (message) => { lastError = message; },
    refreshError,
    requestRender: () => tui.requestRender(),
    renderSelectionOverlay: () => renderSelectionOverlay(),
    watchlist: {
      isVisible: () => watchlistVisible,
      setVisible: (visible) => { watchlistVisible = visible; },
      setEntries: (entries) => { watchlistEntries = entries; },
      setPrices: (prices) => { watchlistPrices = prices; },
      setMode: (mode) => { watchlistMode = mode; },
      setShowTicker: (ticker) => { watchlistShowTicker = ticker; },
      refreshPrices: refreshWatchlistPrices,
      getRefreshIntervalMs: () => watchlistRefreshIntervalMs,
      setRefresh: setWatchlistRefresh,
    },
    dream: {
      isAgentProcessing: () => agentRunner.isProcessing,
      isRunning: () => dreamRunning,
      setRunning: (running) => { dreamRunning = running; },
    },
  };

  const handleSubmit = async (query: string) => {
    if (answerViewerVisible) {
      answerViewerVisible = false;
      renderSelectionOverlay();
    }
    // Dismiss help overlay before processing; re-show below only if /help typed again.
    if (helpVisible) {
      helpVisible = false;
      renderSelectionOverlay();
    }
    if (watchlistVisible) {
      watchlistVisible = false;
      setWatchlistRefresh(0);
      renderSelectionOverlay();
    }
    if (sessionsVisible) {
      sessionsVisible = false;
      renderSelectionOverlay();
    }
    if (memoryVisible) {
      memoryVisible = false;
      renderSelectionOverlay();
    }
    if (skillsVisible) {
      skillsVisible = false;
      renderSelectionOverlay();
    }

    if (await handleExitCommand(query, {
      history: () => agentRunner.history,
      currentModel: () => modelSelection.model,
      safeStopTui,
      writeSessionDailySummary,
      flushSession: () => sessionController.flush(),
    })) {
      return;
    }

    if (await handleCoreSlashCommand(query, {
      chatLog,
      history: () => agentRunner.history,
      currentModel: () => modelSelection.model,
      getThinkEnabled: () => thinkEnabled,
      setThinkEnabled: (value) => { thinkEnabled = value; },
      setAgentThinkEnabled: (value) => agentRunner.setThinkEnabled(value),
      startModelSelection: () => modelSelection.startSelection(),
      listSessions: () => sessionController.listSessions(),
      setSessionsList: (sessions) => { sessionsList = sessions; },
      setSessionsVisible: (visible) => { sessionsVisible = visible; },
      setSkillsList: (skills) => { skillsList = skills; },
      setSkillsVisible: (visible) => { skillsVisible = visible; },
      setHelpVisible: (visible) => { helpVisible = visible; },
      setMemoryVisible: (visible) => { memoryVisible = visible; },
      isMemoryVisible: () => memoryVisible,
      setMemoryContent: (content) => { memoryContent = content; },
      setError: (message) => { lastError = message; },
      refreshError,
      requestRender: () => tui.requestRender(),
      renderSelectionOverlay,
      openAnswerViewer,
      setStatus: (message) => intro.setModel(message),
    })) {
      return;
    }

    const watchlistResult = await handleWatchlistSlashCommand(query, tuiState);
    if (watchlistResult.handled) {
      return;
    }
    query = watchlistResult.query ?? query;

    if (await handleDreamSlashCommand(query, tuiState)) {
      return;
    }

    if (modelSelection.isInSelectionFlow() || sessionsVisible || skillsVisible || watchlistVisible || agentRunner.pendingApproval || agentRunner.isProcessing) {
      return;
    }

    // Flush the PREVIOUS completed exchange to scrollback before starting the
    // new query. If already flushed, skip the scrollback write and just clear
    // the compact TUI view.
    const prevItem = agentRunner.history.at(-1);
    if (prevItem && (prevItem.status === 'complete' || prevItem.status === 'interrupted')) {
      if (flushedItems.has(prevItem)) {
        chatLog.clearAll(); // compact view is still in chatLog; clear it
      } else {
        flushExchangeToScrollback(tui, chatLog, prevItem);
        flushedItems.add(prevItem);
      }
    }

    await inputHistory.saveMessage(query);
    inputHistory.resetNavigation();

    // Auto-save: start a new session on the first query of each run.
    if (!sessionStarted) {
      sessionStarted = true;
      void sessionController.startSession(query);
    }

    try {
      agentRunner.setWatchlistEntries(new WatchlistController(process.cwd()).list());
    } catch {
      agentRunner.setWatchlistEntries([]);
    }

    const result = await agentRunner.runQuery(query);
    if (result?.answer) {
      await inputHistory.updateAgentResponse(result.answer);
    }

    // Persist the updated history after each completed exchange.
    sessionController.autosave(agentRunner.history, modelSelection.inMemoryChatHistory);

    // Update running token counter in the compact status bar.
    const totalTokens = agentRunner.history.reduce(
      (sum, item) => sum + (item.tokenUsage?.totalTokens ?? 0),
      0,
    );
    intro.setTokenCount(totalTokens);

    refreshError();
    tui.requestRender();
  };

  editor.onSubmit = (text) => {
    const value = text.trim();
    // Empty Enter while a modal overlay is open → close it (same as Esc).
    if (!value) {
      if (answerViewerVisible) {
        answerViewerVisible = false;
        renderSelectionOverlay();
        tui.requestRender();
      } else if (memoryVisible) {
        memoryVisible = false;
        renderSelectionOverlay();
        tui.requestRender();
      } else if (watchlistVisible) {
        watchlistVisible = false;
        setWatchlistRefresh(0);
        renderSelectionOverlay();
        tui.requestRender();
      } else if (helpVisible) {
        helpVisible = false;
        renderSelectionOverlay();
        tui.requestRender();
      } else if (skillsVisible) {
        skillsVisible = false;
        renderSelectionOverlay();
        tui.requestRender();
      }
      return;
    }
    editor.setText('');
    editor.addToHistory(value);
    void handleSubmit(value);
  };

  editor.onEscape = () => {
    if (answerViewerVisible) {
      answerViewerVisible = false;
      renderSelectionOverlay();
      return;
    }
    if (memoryVisible) {
      memoryVisible = false;
      renderSelectionOverlay();
      return;
    }
    if (skillsVisible) {
      skillsVisible = false;
      renderSelectionOverlay();
      return;
    }
    if (helpVisible) {
      helpVisible = false;
      renderSelectionOverlay();
      return;
    }
    if (watchlistVisible) {
      watchlistVisible = false;
      setWatchlistRefresh(0);
      renderSelectionOverlay();
      return;
    }
    if (sessionsVisible) {
      sessionsVisible = false;
      renderSelectionOverlay();
      return;
    }
    if (modelSelection.isInSelectionFlow()) {
      modelSelection.cancelSelection();
      return;
    }
    if (agentRunner.isProcessing || agentRunner.pendingApproval) {
      agentRunner.cancelExecution();
      return;
    }
  };

  editor.onCtrlC = () => {
    if (modelSelection.isInSelectionFlow()) {
      modelSelection.cancelSelection();
      return;
    }
    if (agentRunner.isProcessing || agentRunner.pendingApproval) {
      agentRunner.cancelExecution();
      return;
    }
    safeStopTui();
    // Auto-export session if --export flag was provided
    if (exportPathArg !== undefined) {
      const completedHistory = agentRunner.history.filter((h) => h.status === 'complete');
      if (completedHistory.length > 0) {
        try {
          const { path } = exportSession(completedHistory, 'markdown', undefined, exportPathArg ?? undefined);
          process.stdout.write(`\nExported session to ${path}\n`);
        } catch (e) {
          process.stdout.write(`\nExport failed: ${e instanceof Error ? e.message : String(e)}\n`);
        }
      }
    }
    // Write an end-of-session daily summary so Dream has material to consolidate,
    // then flush the session autosave, then exit.
    void writeSessionDailySummary(agentRunner.history, modelSelection.model)
      .finally(() => sessionController.flush().finally(() => process.exit(0)));
  };

  const renderMainView = () => {
    root.clear();
    // Collapse the 15-line ASCII intro to a single header line once the user
    // has started a conversation, freeing vertical space for the chat log.
    intro.setCompact(agentRunner.history.length > 0);
    // Sync think state into the status bar (auto = on for thinking-capable models).
    intro.setThinkState(thinkEnabled !== false && isThinkingModel(modelSelection.model));
    root.addChild(intro);
    root.addChild(chatLog);
    if (lastError ?? agentRunner.error) {
      root.addChild(errorText);
    }
    if (agentRunner.workingState.status !== 'idle') {
      root.addChild(workingIndicator);
    }
    // Hint footer: keyboard shortcuts when idle, cancel hint while running.
    const hintLine = agentRunner.isProcessing
      ? theme.muted('  esc · cancel query')
      : theme.muted('  ↑↓ history  ·  /help  /skills  /model  /watchlist  /memory  /dream  ·  ctrl+c exit');
    root.addChild(new Text(hintLine, 0, 0));
    root.addChild(editor);
    root.addChild(debugPanel);
    tui.setFocus(editor);
  };

  const openAnswerViewer = (content: string) => {
    answerViewerContent = content;
    answerViewerVisible = true;
    renderSelectionOverlay();
    tui.requestRender();
  };

  const renderScreenView = (
    title: string,
    description: string,
    body: any,
    footer?: string,
    focusTarget?: any,
  ) => {
    root.clear();
    root.addChild(createScreen(title, description, body, footer));
    if (focusTarget) {
      tui.setFocus(focusTarget);
    }
  };

  /**
   * Renders a watchlist panel (list / show / snapshot) with the editor visible
   * at the bottom.  Including the editor in the component tree means:
   *   - The cursor is visible so the user knows the TUI is interactive.
   *   - Keyboard input shows in the editor (the user can type the next command).
   *   - Esc (and Enter on an empty line) both close the panel as expected.
   *   - The hint line tells the user which commands make sense next.
   */
  const renderWatchlistView = (
    title: string,
    description: string,
    panel: Container,
    hint: string,
  ) => {
    root.clear();
    root.addChild(createScreen(title, description, panel));
    root.addChild(new Text(theme.muted(hint), 0, 0));
    root.addChild(editor);
    root.addChild(debugPanel);
    tui.setFocus(editor);
  };

  const renderSelectionOverlay = () => {
    const state = modelSelection.state;

    if (answerViewerVisible) {
      const body = new FullAnswerViewerComponent(tui, answerViewerContent, () => {
        answerViewerVisible = false;
        renderSelectionOverlay();
        tui.requestRender();
      });
      renderScreenView(
        '⬡ Cramer-Short — Full Answer',
        '',
        body,
        'Esc to close · ↑↓/j/k scroll · Ctrl+U/D page · g/G top/bottom',
        body,
      );
      return;
    }

    if (helpVisible) {
      renderScreenView(
        '⬡ Cramer-Short — Help',
        '',
        buildHelpPanel(),
        'Esc to close · type a question to close and ask',
        editor,
      );
      return;
    }

    if (sessionsVisible) {
      const selector = createSessionSelector(sessionsList, async (id) => {
        sessionsVisible = false;
        if (id) {
          const loaded = await sessionController.loadSession(id);
          if (loaded) {
            // Flush any current exchange to scrollback before overwriting history.
            const prevItem = agentRunner.history.at(-1);
            if (prevItem && (prevItem.status === 'complete' || prevItem.status === 'interrupted')) {
              flushExchangeToScrollback(tui, chatLog, prevItem);
            } else {
              chatLog.clearAll();
            }

            // Restore LLM context — seed InMemoryChatHistory from the compact layer.
            modelSelection.inMemoryChatHistory.seedFromLlmMessages(
              loaded.llmMessages,
              loaded.priorSummary,
            );

            // Restore display history.
            agentRunner.loadHistory(loaded.history);

            resumedSessionName = loaded.name;
            sessionStarted = true;
            // Adopt the loaded session as current so auto-saves append to it.
            void sessionController.startSessionFromLoaded(loaded);

            intro.setModel(`${modelSelection.model}  ↩ ${loaded.name}`);
            setTimeout(() => {
              intro.setModel(modelSelection.model);
              tui.requestRender();
            }, 4000);
          }
        }
        renderSelectionOverlay();
        tui.requestRender();
      });
      renderScreenView(
        '⬡ Cramer-Short — Sessions',
        'Select a past conversation to resume',
        selector,
        'Enter to resume · ↑↓ navigate · Esc to close',
        selector,
      );
      return;
    }

    if (skillsVisible) {
      const selector = createSkillSelector(skillsList, (name) => {
        skillsVisible = false;
        if (name) {
          editor.setText(`Use the ${name} skill for `);
        }
        renderSelectionOverlay();
        tui.requestRender();
      });
      renderScreenView(
        '⬡ Cramer-Short — Skills',
        'Select a skill to use — press Enter to pre-fill the prompt',
        selector,
        'Enter to use · ↑↓ navigate · Esc to close',
        selector,
      );
      return;
    }

    if (memoryVisible) {
      const panel = new Container();
      if (memoryContent === null) {
        panel.addChild(new Text(theme.muted('  ⏳ Loading memory files…'), 0, 0));
      } else {
        // MEMORY.md section
        panel.addChild(new Text(theme.bold(theme.primary('MEMORY.md')), 0, 0));
        panel.addChild(new Spacer(1));
        if (memoryContent.memory) {
          for (const line of memoryContent.memory.split('\n')) {
            panel.addChild(new Text(`  ${line}`, 0, 0));
          }
        } else {
          panel.addChild(new Text(theme.muted('  (empty)'), 0, 0));
        }
        panel.addChild(new Spacer(1));
        // FINANCE.md section
        panel.addChild(new Text(theme.bold(theme.primary('FINANCE.md')), 0, 0));
        panel.addChild(new Spacer(1));
        if (memoryContent.finance) {
          for (const line of memoryContent.finance.split('\n')) {
            panel.addChild(new Text(`  ${line}`, 0, 0));
          }
        } else {
          panel.addChild(new Text(theme.muted('  (empty)'), 0, 0));
        }
      }
      renderScreenView(
        '⬡ Cramer-Short — Memory',
        'Consolidated long-term memory',
        panel,
        'Esc to close · /dream [force] to consolidate · /dream shows merge conditions',
        editor,
      );
      return;
    }

    if (watchlistVisible) {
      let title: string;
      let subtitle: string;
      let panel: Container;
      let footer: string;

      if (watchlistMode === 'show' && watchlistShowTicker) {
        const snap = watchlistPrices?.get(watchlistShowTicker) ?? null;
        title    = `⬡ Cramer-Short — ${watchlistShowTicker}`;
        subtitle = watchlistPrices === null ? 'Loading…' : (snap ? 'Quick snapshot' : 'Price unavailable');
        panel    = watchlistPrices === null
          ? (() => { const c = new Container(); c.addChild(new Text(theme.muted('  ⏳ Fetching data…'), 0, 0)); return c; })()
          : (snap ? buildShowPanel(watchlistShowTicker, snap) : (() => {
              const c = new Container();
              c.addChild(new Text(theme.error(`  No price data available for ${watchlistShowTicker}`), 0, 0));
              return c;
            })());
        footer   = `Esc to close · /watchlist list · /watchlist snapshot${watchlistRefreshSuffix()}`;
      } else if (watchlistMode === 'snapshot') {
        title    = '⬡ Cramer-Short — Portfolio Snapshot';
        subtitle = watchlistEntries.length === 0 ? 'No positions tracked' : `${watchlistEntries.length} ticker${watchlistEntries.length === 1 ? '' : 's'}`;
        panel    = buildSnapshotPanel(watchlistEntries, watchlistPrices);
        footer   = `Esc to close · /watchlist list · /watchlist show TICKER${watchlistRefreshSuffix()}`;
      } else {
        const loading = watchlistPrices === null;
        title    = '⬡ Cramer-Short — Watchlist';
        subtitle = watchlistEntries.length === 0
          ? 'No positions tracked'
          : loading
            ? `${watchlistEntries.length} position${watchlistEntries.length === 1 ? '' : 's'} — loading prices…`
            : `${watchlistEntries.length} position${watchlistEntries.length === 1 ? '' : 's'}`;
        panel    = buildWatchlistPanel(watchlistEntries, watchlistPrices);
        footer   = `Esc to close · /watchlist show TICKER · /watchlist snapshot${watchlistRefreshSuffix()}`;
      }

      renderWatchlistView(title, subtitle, panel, footer);
      return;
    }

    if (state.appState === 'idle' && !agentRunner.pendingApproval) {
      refreshError();
      renderMainView();
      return;
    }

    if (agentRunner.pendingApproval) {
      const prompt = new ApprovalPromptComponent(
        agentRunner.pendingApproval.tool,
        agentRunner.pendingApproval.args,
      );
      prompt.onSelect = (decision: ApprovalDecision) => {
        agentRunner.respondToApproval(decision);
      };
      renderScreenView('', '', prompt, undefined, prompt.selector);
      return;
    }

    if (state.appState === 'provider_select') {
      const selector = createProviderSelector(modelSelection.provider, (providerId) => {
        void modelSelection.handleProviderSelect(providerId);
      });
      renderScreenView(
        'Select provider',
        'Switch between LLM providers. Applies to this session and future sessions.',
        selector,
        'Enter to confirm · esc to exit',
        selector,
      );
      return;
    }

    if (state.appState === 'model_select' && state.pendingProvider) {
      const selector = createModelSelector(
        state.pendingModels,
        modelSelection.provider === state.pendingProvider ? modelSelection.model : undefined,
        (modelId) => modelSelection.handleModelSelect(modelId),
        state.pendingProvider,
      );
      renderScreenView(
        `Select model for ${getProviderDisplayName(state.pendingProvider)}`,
        '',
        selector,
        'Enter to confirm · esc to go back',
        selector,
      );
      return;
    }

    if (state.appState === 'model_input' && state.pendingProvider) {
      const input = new ApiKeyInputComponent();
      input.onSubmit = (value) => modelSelection.handleModelInputSubmit(value);
      input.onCancel = () => modelSelection.handleModelInputSubmit(null);
      renderScreenView(
        `Enter model name for ${getProviderDisplayName(state.pendingProvider)}`,
        'Type or paste the model name from openrouter.ai/models',
        input,
        'Examples: anthropic/claude-3.5-sonnet, openai/gpt-4-turbo, meta-llama/llama-3-70b\nEnter to confirm · esc to go back',
        input,
      );
      return;
    }

    if (state.appState === 'api_key_confirm' && state.pendingProvider) {
      const selector = createApiKeyConfirmSelector((wantsToSet) =>
        modelSelection.handleApiKeyConfirm(wantsToSet),
      );
      renderScreenView(
        'Set API Key',
        `Would you like to set your ${getProviderDisplayName(state.pendingProvider)} API key?`,
        selector,
        'Enter to confirm · esc to decline',
        selector,
      );
      return;
    }

    if (state.appState === 'api_key_input' && state.pendingProvider) {
      const input = new ApiKeyInputComponent(true);
      input.onSubmit = (apiKey) => modelSelection.handleApiKeySubmit(apiKey);
      input.onCancel = () => modelSelection.handleApiKeySubmit(null);
      const apiKeyName = getApiKeyNameForProvider(state.pendingProvider) ?? '';
      renderScreenView(
        `Enter ${getProviderDisplayName(state.pendingProvider)} API Key`,
        apiKeyName ? `(${apiKeyName})` : '',
        input,
        'Enter to confirm · Esc to cancel',
        input,
      );
    }
  };

  await inputHistory.init();
  for (const msg of inputHistory.getMessages().reverse()) {
    editor.addToHistory(msg);
  }
  renderSelectionOverlay();
  refreshError();

  // Suppress third-party console output (e.g. @langchain/tavily logs raw Response objects
  // on HTTP errors) from bleeding into the TUI's stdout rendering.  Redirect to the
  // structured error log instead so the output is never lost but never corrupts the UI.
  const _origConsoleLog = console.log;
  const _origConsoleWarn = console.warn;
  const _origConsoleError = console.error;
  const suppressToLog = (level: 'warn' | 'error', ...args: unknown[]) => {
    const msg = args.map(a => (a instanceof Error ? a.message : String(a))).join(' ');
    logError({ type: `console-${level}`, message: msg, context: 'tui-suppressed' });
  };
  console.log = (...args: unknown[]) => suppressToLog('warn', ...args);
  console.warn = (...args: unknown[]) => suppressToLog('warn', ...args);
  console.error = (...args: unknown[]) => suppressToLog('error', ...args);
  process.once('exit', () => {
    // Safety net: ensure terminal is restored on any exit path.
    // Without this, raw mode, Kitty protocol, bracketed paste, and
    // cursor visibility can be left in a broken state, making the
    // parent shell unusable (no echo, escape sequences leak as text).
    try {
      if (process.stdin.setRawMode && process.stdin.isRaw) {
        process.stdin.setRawMode(false);
      }
    } catch { /* best-effort */ }
    try {
      process.stdout.write('\x1b[?25h');   // show cursor
      process.stdout.write('\x1b[?2004l'); // disable bracketed paste
      process.stdout.write('\x1b[<u');     // disable Kitty keyboard protocol
    } catch {
      // stdout may already be closed (EPIPE/EBADF); best-effort.
    }

    console.log = _origConsoleLog;
    console.warn = _origConsoleWarn;
    console.error = _origConsoleError;
  });

  tui.start();

  // Seed existing watchlist tickers into financial memory at startup.
  // Ensures recall_financial_context() returns a result for tracked tickers
  // even before any LLM analysis has run for them.
  void (async () => {
    try {
      const existingEntries = new WatchlistController(process.cwd()).list();
      if (existingEntries.length > 0) {
        await seedWatchlistEntries(existingEntries);
      }
    } catch {
      // Non-critical.
    }
  })();

  // Auto-trigger Dream consolidation on startup if conditions are met.
  // Increments the session counter unconditionally, then runs consolidation
  // in the background without blocking the TUI or the user's first query.
  // The 400ms defer ensures the TUI is fully painted before Dream starts,
  // so the user sees a responsive interface even on first launch.
  void (async () => {
    await new Promise<void>((r) => setTimeout(r, 400));
    const dreamStore = new MemoryStore();
    try {
      await incrementDreamSessionCount(dreamStore);
      const [dreamMeta, dreamDailyFiles] = await Promise.all([
        dreamStore.readDreamMeta(),
        dreamStore.listDailyFiles(),
      ]);
      if (!dreamRunning && shouldRunDream(dreamMeta, dreamDailyFiles)) {
        dreamRunning = true;
        intro.setModel('🌙 Dream running…');
        tui.requestRender();
        const result = await runDream(dreamStore, modelSelection.model);
        if (result.ran) {
          const n = result.archivedFiles.length;
          intro.setModel(`✨ Dream: archived ${n} file${n === 1 ? '' : 's'}`);
          tui.requestRender();
          setTimeout(() => { intro.setModel(modelSelection.model); tui.requestRender(); }, 4000);
        } else {
          intro.setModel(modelSelection.model);
          tui.requestRender();
        }
      }
    } catch {
      // Non-fatal — Dream failure must never crash the TUI.
    } finally {
      dreamRunning = false;
    }
  })();
  await new Promise<void>((resolve) => {
    const finish = () => resolve();
    process.once('exit', finish);
    process.once('SIGINT', finish);
    process.once('SIGTERM', finish);
  });

  // Restore terminal state before disposing resources.
  safeStopTui();
  workingIndicator.dispose();
  debugPanel.dispose();
}
