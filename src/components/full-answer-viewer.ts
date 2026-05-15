import { Key, Markdown, matchesKey, type TUI } from '@mariozechner/pi-tui';
import { markdownTheme, theme } from '../theme.js';
import { formatResponseTui, renderTuiMarkdownLines } from '../utils/ui/markdown-table.js';

interface FullAnswerViewerOptions {
  viewportRows?: number;
}

function getViewportHeight(viewportRows?: number): number {
  const rows = viewportRows ?? process.stdout.rows ?? 40;
  return Math.max(8, rows - 8);
}

function getVisibleContentHeight(viewportRows?: number): number {
  return Math.max(3, getViewportHeight(viewportRows) - 2);
}

/**
 * Read-only scrollable markdown viewer for long completed answers.
 * Keeps markdown syntax intact so pi-tui can do width-aware rendering,
 * while adding simple keyboard-driven vertical scrolling.
 */
export class FullAnswerViewerComponent {
  private readonly body: Markdown;
  private readonly onClose: () => void;
  private readonly tui: TUI;
  private readonly viewportRows?: number;
  private content = '';
  private scrollOffset = 0;
  private cachedWidth = -1;
  private cachedLines: string[] = [];

  constructor(
    tui: TUI,
    content: string,
    onClose: () => void,
    options: FullAnswerViewerOptions = {},
  ) {
    this.tui = tui;
    this.onClose = onClose;
    this.viewportRows = options.viewportRows;
    this.body = new Markdown('', 0, 0, markdownTheme, { color: (line) => line });
    this.setContent(content);
  }

  setContent(content: string) {
    const normalized = formatResponseTui(content).replace(/^\n+/, '');
    this.content = normalized;
    this.body.setText(normalized);
    this.cachedWidth = -1;
    this.cachedLines = [];
    this.scrollOffset = 0;
  }

  handleInput(data: string): void {
    if (matchesKey(data, Key.escape)) {
      this.onClose();
      return;
    }

    const maxOffset = Math.max(0, this.getRenderedLines().length - getVisibleContentHeight(this.viewportRows));
    const prevOffset = this.scrollOffset;

    if (matchesKey(data, Key.down) || data === 'j') {
      this.scrollOffset = Math.min(maxOffset, this.scrollOffset + 1);
    } else if (matchesKey(data, Key.up) || data === 'k') {
      this.scrollOffset = Math.max(0, this.scrollOffset - 1);
    } else if (matchesKey(data, Key.ctrl('d')) || data === '\u001b[6~') {
      this.scrollOffset = Math.min(maxOffset, this.scrollOffset + Math.max(3, Math.floor(getVisibleContentHeight(this.viewportRows) / 2)));
    } else if (matchesKey(data, Key.ctrl('u')) || data === '\u001b[5~') {
      this.scrollOffset = Math.max(0, this.scrollOffset - Math.max(3, Math.floor(getVisibleContentHeight(this.viewportRows) / 2)));
    } else if (matchesKey(data, Key.home) || data === 'g') {
      this.scrollOffset = 0;
    } else if (matchesKey(data, Key.end) || data === 'G') {
      this.scrollOffset = maxOffset;
    }

    if (this.scrollOffset !== prevOffset) {
      this.tui.requestRender();
    }
  }

  render(width: number): string[] {
    const allLines = this.getRenderedLines(width);
    const visibleHeight = getVisibleContentHeight(this.viewportRows);
    const maxOffset = Math.max(0, allLines.length - visibleHeight);
    this.scrollOffset = Math.min(this.scrollOffset, maxOffset);

    const topHidden = this.scrollOffset;
    const bottomHidden = Math.max(0, allLines.length - (this.scrollOffset + visibleHeight));
    const slice = allLines.slice(this.scrollOffset, this.scrollOffset + visibleHeight);

    const lines: string[] = [];
    if (topHidden > 0) {
      lines.push(theme.muted(`↑ ${topHidden} line${topHidden === 1 ? '' : 's'} above`));
    }
    lines.push(...slice);
    if (bottomHidden > 0) {
      lines.push(theme.muted(`↓ ${bottomHidden} line${bottomHidden === 1 ? '' : 's'} below`));
    }
    return lines;
  }

  invalidate(): void {
    this.body.invalidate();
    this.cachedWidth = -1;
    this.cachedLines = [];
  }

  private getRenderedLines(width?: number): string[] {
    const effectiveWidth = width ?? this.cachedWidth;
    if (effectiveWidth > 0 && this.cachedWidth === effectiveWidth && this.cachedLines.length > 0) {
      return this.cachedLines;
    }
    const rendered = renderTuiMarkdownLines(this.content, Math.max(10, effectiveWidth > 0 ? effectiveWidth : 80));
    if (effectiveWidth > 0) {
      this.cachedWidth = effectiveWidth;
      this.cachedLines = rendered;
    }
    return rendered;
  }
}
