import { Spacer, Text } from '@mariozechner/pi-tui';
import type { TUI } from '@mariozechner/pi-tui';
import type { ReasoningEvent, ToolEndEvent, ToolErrorEvent, ToolStartEvent } from '../agent/types.js';
import type { ChatLogComponent } from '../components/index.js';
import type { AgentRunnerController } from './agent-runner.js';
import type { HistoryItem } from './types.js';
import { theme } from '../theme.js';
import { countRenderedTuiMarkdownLines, truncateTuiMarkdownTail } from '../utils/markdown-table.js';
import { formatExchangeForScrollback } from '../utils/scrollback.js';

export function truncateAtWord(str: string, maxLength: number): string {
  if (str.length <= maxLength) {
    return str;
  }
  const lastSpace = str.lastIndexOf(' ', maxLength);
  if (lastSpace > maxLength * 0.5) {
    return `${str.slice(0, lastSpace)}...`;
  }
  return `${str.slice(0, maxLength)}...`;
}

export function summarizeToolResult(tool: string, args: Record<string, unknown>, result: string): string {
  if (tool === 'skill') {
    const skillName = args.skill as string;
    return `Loaded ${skillName} skill`;
  }
  try {
    const parsed = JSON.parse(result);
    if (parsed.data) {
      if (Array.isArray(parsed.data)) {
        return `Received ${parsed.data.length} items`;
      }
      if (typeof parsed.data === 'object') {
        const keys = Object.keys(parsed.data).filter((key) => !key.startsWith('_'));
        if (tool === 'get_financials' || tool === 'get_market_data' || tool === 'stock_screener') {
          return keys.length === 1 ? 'Called 1 data source' : `Called ${keys.length} data sources`;
        }
        if (tool === 'web_search') {
          return 'Did 1 search';
        }
        return `Received ${keys.length} fields`;
      }
    }
  } catch {
    return truncateAtWord(result, 50);
  }
  return 'Received data';
}

/**
 * While a query is running, cap the number of visible events to keep the TUI
 * within the terminal viewport.  Without this limit, a long-running agent with
 * many tool calls would push the earliest events above the top of the screen
 * with no way for the user to scroll back (the TUI snaps to the bottom on
 * every re-render).  After completion, all events are shown.
 */
const MAX_RUNNING_EVENTS = 30;

function getRenderedAnswerWidth(): number {
  return Math.max(10, process.stdout.columns ?? 80);
}

/**
 * Render only the most recent history item (currently executing or just completed).
 * Completed exchanges are flushed to the terminal scrollback buffer so the TUI
 * stays lean — only the active query lives in the TUI viewport.
 */
export function renderCurrentQuery(chatLog: ChatLogComponent, history: AgentRunnerController['history']) {
  chatLog.clearAll();
  const item = history[history.length - 1];
  if (!item) return;

  chatLog.addQuery(item.query);
  chatLog.resetToolGrouping();

  if (item.status === 'interrupted') {
    chatLog.addInterrupted();
  }

  // Dynamic event cap: keep total TUI rendering ≤ termRows+2 so the query header
  // (at absolute line index 2) never scrolls into the terminal's native scrollback.
  // If it did, flushExchangeToScrollback() cannot clear it — causing the prompt to
  // appear twice ("duplicate prompt" bug).
  //
  // Fixed overhead accounts for: intro(1)+spacer(1)+query(1)+hidden-events(1)+
  // hint(1)+editor(3)+working-indicator(1)+safety-margin(1) = ~10 lines.
  const termRows = process.stdout.rows ?? 40;
  const maxContentLines = Math.max(8, termRows - 8); // budget for events + answer
  const isRunning = item.status !== 'complete' && item.status !== 'interrupted';
  const effectiveMaxEvents = Math.min(
    MAX_RUNNING_EVENTS,
    Math.max(2, Math.floor((maxContentLines - 5) / 2)),
  );
  const allEvents = item.events;
  const hiddenCount = isRunning && allEvents.length > effectiveMaxEvents
    ? allEvents.length - effectiveMaxEvents
    : 0;
  const visibleEvents = hiddenCount > 0 ? allEvents.slice(-effectiveMaxEvents) : allEvents;

  if (hiddenCount > 0) {
    chatLog.addChild(new Text(theme.muted(`  … ${hiddenCount} earlier events`), 0, 0));
  }

  for (const display of visibleEvents) {
    const event = display.event;
    if (event.type === 'thinking') {
      const message = event.message.trim();
      if (message) {
        const preview = message.length > 120 ? `${message.slice(0, 120)}…` : message;
        chatLog.addChild(new Text(theme.muted(`  💭 ${preview}`), 0, 0));
      }
      continue;
    }

    if (event.type === 'reasoning') {
      const reasoning = (event as ReasoningEvent).content.trim();
      if (reasoning) {
        const preview = reasoning.length > 300 ? `${reasoning.slice(0, 300)}...` : reasoning;
        chatLog.addChild(new Spacer(1));
        chatLog.addChild(new Text(theme.muted(`💭 Reasoning (${reasoning.length} chars)`), 0, 0));
        chatLog.addChild(new Text(theme.muted(preview), 0, 0));
      }
      continue;
    }

    if (event.type === 'tool_start') {
      const toolStart = event as ToolStartEvent;
      const component = chatLog.startTool(display.id, toolStart.tool, toolStart.args);
      if (display.completed && display.endEvent?.type === 'tool_end') {
        const done = display.endEvent as ToolEndEvent;
        component.setComplete(
          summarizeToolResult(done.tool, toolStart.args, done.result),
          done.duration,
        );
      } else if (display.completed && display.endEvent?.type === 'tool_error') {
        const toolError = display.endEvent as ToolErrorEvent;
        component.setError(toolError.error);
      } else if (display.progressMessage) {
        component.setActive(display.progressMessage);
      }
      continue;
    }

    if (event.type === 'tool_approval') {
      const approval = chatLog.startTool(display.id, event.tool, event.args);
      approval.setApproval(event.approved);
      continue;
    }

    if (event.type === 'tool_denied') {
      const denied = chatLog.startTool(display.id, event.tool, event.args);
      const path = (event.args.path as string) ?? '';
      denied.setDenied(path, event.tool);
      continue;
    }

    if (event.type === 'tool_limit') {
      continue;
    }

    if (event.type === 'context_cleared') {
      chatLog.addContextCleared(event.clearedCount, event.keptCount);
    }
  }

  if (item.answer) {
    const isStreaming = item.status === 'processing';

    if (isStreaming) {
      // During streaming: show only the tail so the TUI never overflows the viewport.
      // Overflow would push early lines into the terminal's native scrollback,
      // making it impossible for flushExchangeToScrollback() to clear them — causing
      // the answer to appear twice (partial live view + full flushed version).
      const answerBudget = Math.max(3, maxContentLines - visibleEvents.length * 2);
      const renderedAnswerLines = countRenderedTuiMarkdownLines(item.answer, getRenderedAnswerWidth());
      if (renderedAnswerLines > answerBudget) {
        const tail = truncateTuiMarkdownTail(item.answer, answerBudget, getRenderedAnswerWidth());
        chatLog.finalizeAnswer(tail.text);
      } else {
        chatLog.finalizeAnswer(item.answer);
      }
    } else {
      // Completed answer: show the full answer. The user needs to read it.
      // Overflow into terminal scrollback is acceptable — the flush-on-next-query
      // mechanism handles cleanup when the user starts a new query.
      chatLog.finalizeAnswer(item.answer);
    }
  }
  if (item.status === 'complete') {
    chatLog.addPerformanceStats(item.duration ?? 0, item.tokenUsage, item.tokensPerSecond);
  }
}

/**
 * Flush a completed exchange to the terminal's native scrollback buffer.
 *
 * How it works:
 *  1. Capture how many lines the TUI is currently rendering (before stop).
 *  2. Stop the TUI — this positions the hardware cursor at the end of all
 *     rendered content and briefly disables raw mode.
 *  3. Move the cursor up to the top of the TUI viewport and clear to the end
 *     of the screen. This removes the live-processing trail from the viewport
 *     but does NOT erase lines already pushed into the scrollback buffer.
 *  4. Write the formatted exchange — it lands in the terminal's scroll buffer.
 *  5. Clear the TUI component tree (chatLog) so the next render starts fresh.
 *  6. Restart the TUI — re-enables raw mode, resets rendering state, re-renders
 *     the now-empty chatLog + editor from the current cursor position.
 */
export function flushExchangeToScrollback(
  tui: TUI,
  chatLog: ChatLogComponent,
  item: HistoryItem,
): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tuiInternal = tui as any;
  // Snapshot BEFORE stop() moves the cursor.
  const prevLineCount: number = tuiInternal.previousLines?.length ?? 0;

  // Stop TUI: moves cursor to end of rendered content (+\r\n), disables raw mode.
  tui.stop();

  // Move cursor back to the top of the TUI's rendered area and clear to end of
  // screen so the live processing trail doesn't appear above the clean output.
  if (prevLineCount > 0) {
    // After stop(), cursor is near the bottom of the terminal viewport.
    // IMPORTANT: clamp to terminal height — if prevLineCount > terminal rows the TUI
    // was using internal viewport scrolling, and moving up more than the viewport
    // height would overshoot into the scrollback buffer, causing \x1b[J to wipe
    // previously-committed exchange content.
    const termRows = process.stdout.rows ?? 24;
    const moveUp = Math.min(prevLineCount + 1, termRows);
    process.stdout.write(`\x1b[${moveUp}A`); // move up (clamped to viewport)
    process.stdout.write('\x1b[J');          // clear to end of screen
  }

  // Write the formatted exchange — this is what lands in the scroll buffer.
  process.stdout.write(formatExchangeForScrollback(item));

  // Clear TUI component state.
  chatLog.clearAll();

  // Restart TUI: re-enable raw mode, hide cursor.
  //
  // IMPORTANT: Do NOT call tui.requestRender(true) here.
  // requestRender(true) sets previousWidth = -1 which causes the next render to
  // take the "width changed" code path → fullRender(clear=true) → \x1b[3J which
  // CLEARS THE ENTIRE SCROLLBACK BUFFER, erasing the exchange we just wrote above.
  //
  // Instead, manually reset only the cursor-tracking fields and leave previousWidth
  // at its current value. With previousLines=[] and widthChanged=false, the render
  // takes the "first render" path → fullRender(clear=false) → writes UI content at
  // the current cursor position WITHOUT touching the scrollback buffer.
  tuiInternal.previousLines = [];
  tuiInternal.cursorRow = 0;
  tuiInternal.hardwareCursorRow = 0;
  tuiInternal.maxLinesRendered = 0;
  tuiInternal.previousViewportTop = 0;
  // Do NOT reset previousWidth — keeps widthChanged=false → no \x1b[3J scrollback wipe.
  tui.start();
  tui.requestRender(); // non-force: uses manual state above, hits "first render" path
}
