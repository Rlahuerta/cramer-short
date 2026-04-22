/**
 * Markdown table parsing and box-drawing rendering utilities.
 * 
 * Converts markdown tables to properly-aligned Unicode box-drawing tables.
 * Also handles bold text formatting.
 */

import { Markdown } from '@mariozechner/pi-tui';
import chalk from 'chalk';
import { markdownTheme } from '../theme.js';

// Box-drawing characters
const BOX = {
  topLeft: '┌',
  topRight: '┐',
  bottomLeft: '└',
  bottomRight: '┘',
  horizontal: '─',
  vertical: '│',
  topT: '┬',
  bottomT: '┴',
  leftT: '├',
  rightT: '┤',
  cross: '┼',
};

/**
 * Strip markdown formatting markers (**bold**, *italic*, _italic_) from a string
 * to get its visible display width.
 *
 * `transformBold` later converts **...** to invisible ANSI escape sequences, so
 * column widths and padding MUST be computed against the stripped content, not
 * the raw markdown string.  Otherwise bold-formatted cells (e.g. **HON**) get
 * measured as 7 chars but display as 3 — causing misaligned data rows.
 */
function stripMarkdownFormatting(text: string): string {
  return text
    .replace(/\*\*([^*]+)\*\*/g, '$1')   // **bold** → content
    .replace(/\*([^*]+)\*/g, '$1')        // *italic* → content
    .replace(/_([^_\s][^_]*)_/g, '$1');   // _italic_ → content
}

/**
 * Visible width of a cell value for layout purposes (strips markdown markers).
 */
function cellDisplayWidth(text: string): number {
  return stripMarkdownFormatting(text).length;
}

/**
 * Pad a cell to a target visible width, accounting for markdown formatting
 * markers that will later be converted to invisible ANSI escape sequences.
 * Uses visible width (not raw .length) for correct alignment after conversion.
 */
function padCellAware(value: string, targetWidth: number, rightAlign: boolean): string {
  const visWidth = cellDisplayWidth(value);
  const padding = Math.max(0, targetWidth - visWidth);
  if (rightAlign) {
    return ' '.repeat(padding) + value;
  }
  return value + ' '.repeat(padding);
}

/**
 * Check if a string looks like a number (for right-alignment).
 */
function isNumeric(value: string): boolean {
  const trimmed = stripMarkdownFormatting(value).trim();
  // Match numbers with optional $, %, B/M/K suffixes
  return /^[$]?[-+]?[\d,]+\.?\d*[%BMK]?$/.test(trimmed);
}

/**
 * Parse a markdown table into headers and rows.
 */
export function parseMarkdownTable(tableText: string): { headers: string[]; rows: string[][] } | null {
  const lines = tableText.trim().split('\n').map(line => line.trim());
  
  if (lines.length < 2) return null;
  
  // Parse header line
  const headerLine = lines[0];
  if (!headerLine.includes('|')) return null;
  
  const headers = headerLine
    .split('|')
    .map(cell => cell.trim())
    .filter((_, i, arr) => i > 0 && i < arr.length - 1 || arr.length === 1);
  
  // Handle edge case where there's no leading/trailing pipe
  if (headers.length === 0) {
    const rawHeaders = headerLine.split('|').map(cell => cell.trim());
    if (rawHeaders.length > 0) {
      headers.push(...rawHeaders);
    }
  }
  
  if (headers.length === 0) return null;
  
  // Check for separator line (---|---|---)
  const separatorLine = lines[1];
  if (!separatorLine || !/^[\s|:-]+$/.test(separatorLine)) return null;
  
  // Parse data rows
  const rows: string[][] = [];
  for (let i = 2; i < lines.length; i++) {
    const line = lines[i];
    if (!line.includes('|')) continue;
    
    const cells = line
      .split('|')
      .map(cell => cell.trim());
    
    // Remove empty first/last cells from pipes at start/end
    if (cells[0] === '') cells.shift();
    if (cells[cells.length - 1] === '') cells.pop();
    
    if (cells.length > 0) {
      rows.push(cells);
    }
  }
  
  return { headers, rows };
}

/**
 * Render a parsed table as a Unicode box-drawing table.
 */
export function renderBoxTable(headers: string[], rows: string[][]): string {
  // Calculate column widths using visible display width (strips **bold** etc. markers
  // that will later be converted to invisible ANSI codes by transformBold).
  const colWidths: number[] = headers.map(h => cellDisplayWidth(h));
  
  for (const row of rows) {
    for (let i = 0; i < row.length; i++) {
      if (i < colWidths.length) {
        colWidths[i] = Math.max(colWidths[i], cellDisplayWidth(row[i]));
      }
    }
  }
  
  // Determine alignment for each column (right for numeric, left for text)
  const alignRight: boolean[] = headers.map((_, colIndex) => {
    // Check if most values in this column are numeric
    let numericCount = 0;
    for (const row of rows) {
      if (row[colIndex] && isNumeric(row[colIndex])) {
        numericCount++;
      }
    }
    return numericCount > rows.length / 2;
  });
  
  // Build the table
  const lines: string[] = [];
  
  // Top border
  const topBorder = BOX.topLeft + 
    colWidths.map(w => BOX.horizontal.repeat(w + 2)).join(BOX.topT) + 
    BOX.topRight;
  lines.push(topBorder);
  
  // Header row — padCellAware accounts for markdown markers in header text
  const headerRow = BOX.vertical + 
    headers.map((h, i) => ` ${padCellAware(h, colWidths[i], false)} `).join(BOX.vertical) + 
    BOX.vertical;
  lines.push(headerRow);
  
  // Header separator
  const headerSep = BOX.leftT + 
    colWidths.map(w => BOX.horizontal.repeat(w + 2)).join(BOX.cross) + 
    BOX.rightT;
  lines.push(headerSep);
  
  // Data rows — padCellAware handles bold/italic markers in cell values
  for (const row of rows) {
    const dataRow = BOX.vertical + 
      colWidths.map((w, i) => {
        const value = row[i] || '';
        return ` ${padCellAware(value, w, alignRight[i])} `;
      }).join(BOX.vertical) + 
      BOX.vertical;
    lines.push(dataRow);
  }
  
  // Bottom border
  const bottomBorder = BOX.bottomLeft + 
    colWidths.map(w => BOX.horizontal.repeat(w + 2)).join(BOX.bottomT) + 
    BOX.bottomRight;
  lines.push(bottomBorder);
  
  return lines.join('\n');
}

/**
 * Find and transform all markdown tables in content to box-drawing tables.
 */
export function transformMarkdownTables(content: string): string {
  // Normalize line endings: convert \r\n to \n, then trim trailing whitespace from each line
  const normalized = content
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map(line => line.trimEnd())
    .join('\n');
  
  // Regex to match markdown tables:
  // - Starts with a line containing pipes
  // - Followed by a separator line (---|---|---)
  // - Followed by zero or more data rows with pipes
  // IMPORTANT: Use [ \t] instead of \s in separator to avoid matching newlines
  const tableRegex = /^(\|[^\n]+\|\n\|[-:| \t]+\|(?:\n\|[^\n]+\|)*)/gm;
  
  // Also match tables without leading/trailing pipes on each line
  const tableRegex2 = /^([^\n|]*\|[^\n]+\n[-:| \t]+(?:\n[^\n|]*\|[^\n]+)*)/gm;
  
  let result = normalized;
  
  // Process tables with pipes at start/end
  result = result.replace(tableRegex, (match) => {
    const parsed = parseMarkdownTable(match);
    if (parsed && parsed.headers.length > 0 && parsed.rows.length > 0) {
      return renderBoxTable(parsed.headers, parsed.rows);
    }
    return match;
  });
  
  // Process tables that might not have leading pipes
  result = result.replace(tableRegex2, (match) => {
    // Skip if already transformed (contains box-drawing chars)
    if (match.includes(BOX.topLeft)) return match;
    
    const parsed = parseMarkdownTable(match);
    if (parsed && parsed.headers.length > 0 && parsed.rows.length > 0) {
      return renderBoxTable(parsed.headers, parsed.rows);
    }
    return match;
  });
  
  return result;
}

/**
 * Transform markdown bold (**text**) to ANSI bold.
 */
export function transformBold(content: string): string {
  return content.replace(/\*\*([^*]+)\*\*/g, (_, text) => chalk.bold(text));
}

/**
 * Transform markdown headers (# / ## / ###) to styled terminal output.
 * - `#`  → bold yellow with surrounding newlines
 * - `##` → bold
 * - `###` → bold dim
 */
export function transformHeaders(content: string): string {
  return content.replace(/^(#{1,3})\s+(.+)$/gm, (_, hashes, text) => {
    if (hashes === '#') {
      return '\n' + chalk.bold(chalk.yellow(text)) + '\n';
    } else if (hashes === '##') {
      return chalk.bold(text);
    } else {
      return chalk.bold(chalk.dim(text));
    }
  });
}

/**
 * Transform markdown italic (*text* and _text_) to ANSI italic.
 * Must run after transformBold so **bold** is already replaced and only
 * single-asterisk patterns remain.
 */
export function transformItalic(content: string): string {
  // *text* — single asterisk not preceded/followed by another *
  let result = content.replace(/(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)/g, (_, text) => chalk.italic(text));
  // _text_ — not adjacent to word characters (avoids snake_case)
  result = result.replace(/(?<!\w)_([^_\s][^_\n]*)_(?!\w)/g, (_, text) => chalk.italic(text));
  return result;
}

/**
 * Transform inline code (`code`) to cyan terminal output.
 */
export function transformInlineCode(content: string): string {
  return content.replace(/`([^`\n]+)`/g, (_, code) => chalk.cyan(code));
}

/**
 * Transform markdown list items to styled terminal output.
 * - Unordered (`- item` or `* item`) → `• item`
 * - Ordered (`1. item`) → number kept, item styled
 */
export function transformLists(content: string): string {
  return content
    .replace(/^[ \t]*[-*]\s+(.+)$/gm, (_, item) => chalk.white(`• ${item}`))
    .replace(/^[ \t]*(\d+)\.\s+(.+)$/gm, (_, num, item) => `${num}. ${chalk.white(item)}`);
}

/**
 * Transform bare URLs to cyan underlined terminal output.
 * Does not process URLs already inside markdown link syntax [text](url).
 */
export function transformURLs(content: string): string {
  return content.replace(/(?<!\()(https?:\/\/[^\s)\]]+)/g, (url) => chalk.cyan.underline(url));
}

/**
 * Apply all pre-render formatting to response content.
 * Processing order matters to avoid conflicts:
 * 1. Tables (box-drawing must happen on raw markdown)
 * 2. Headers
 * 3. Bold (**text**)
 * 4. Italic (*text*, _text_) — after bold so ** is already consumed
 * 5. Inline code (`code`)
 * 6. Lists (- / * / 1.)
 * 7. URLs
 */
export function formatResponse(content: string): string {
  let result = content;
  result = transformMarkdownTables(result);
  result = transformHeaders(result);
  result = transformBold(result);
  result = transformItalic(result);
  result = transformInlineCode(result);
  result = transformLists(result);
  result = transformURLs(result);
  return result;
}

/**
 * Strip terminal control sequences from content before sending it through the
 * live TUI markdown renderer. This preserves visible text while removing ANSI,
 * OSC, C1, and other non-printing control bytes that could manipulate the
 * terminal when replayed interactively.
 */
function stripTerminalControlSequences(content: string): string {
  return content
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\u001B\][\s\S]*?(?:\u0007|\u001B\\)/g, '')
    .replace(/(?:\u001B\[|\u009B)[0-?]*[ -/]*[@-~]/g, '')
    .replace(/\u001B[@-_]/g, '')
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001A\u001C-\u001F\u007F\u0080-\u009A\u009C-\u009F]/g, '')
    .replace(/\u001B/g, '');
}

/**
 * TUI-optimized formatting that preserves raw markdown tables and markdown
 * syntax for the final renderer, while stripping terminal control sequences so
 * untrusted content cannot manipulate the live TUI.
 */
export function formatResponseTui(content: string): string {
  return stripTerminalControlSequences(content);
}

function normalizeTuiMarkdownContent(content: string): string {
  return formatResponseTui(content).replace(/^\n+/, '');
}

export function renderTuiMarkdownLines(content: string, width: number): string[] {
  const markdown = new Markdown('', 0, 0, markdownTheme, { color: (line) => line });
  markdown.setText(normalizeTuiMarkdownContent(content));
  return markdown.render(Math.max(10, width));
}

export function countRenderedTuiMarkdownLines(content: string, width: number): number {
  return renderTuiMarkdownLines(content, width).length;
}

export function truncateTuiMarkdownTail(
  content: string,
  maxRenderedLines: number,
  width: number,
): { text: string; truncated: boolean; renderedLineCount: number } {
  const fullRenderedLineCount = countRenderedTuiMarkdownLines(content, width);
  if (fullRenderedLineCount <= maxRenderedLines) {
    return { text: content, truncated: false, renderedLineCount: fullRenderedLineCount };
  }

  const rawLines = content.split('\n');
  const contentBudget = Math.max(1, maxRenderedLines - 1);
  let low = 0;
  let high = rawLines.length - 1;
  let bestStart = rawLines.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const candidate = rawLines.slice(mid).join('\n');
    if (countRenderedTuiMarkdownLines(candidate, width) <= contentBudget) {
      bestStart = mid;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }

  const fallbackCandidate = rawLines.slice(bestStart).join('\n');
  if (countRenderedTuiMarkdownLines(fallbackCandidate, width) > contentBudget) {
    const normalized = normalizeTuiMarkdownContent(content);
    let charLow = 0;
    let charHigh = normalized.length - 1;
    let bestOffset = normalized.length - 1;

    while (charLow <= charHigh) {
      const mid = Math.floor((charLow + charHigh) / 2);
      const candidate = normalized.slice(mid);
      if (countRenderedTuiMarkdownLines(candidate, width) <= contentBudget) {
        bestOffset = mid;
        charHigh = mid - 1;
      } else {
        charLow = mid + 1;
      }
    }

    const trimmed = normalized.slice(bestOffset).trimStart();
    const safeTail = trimmed.length > 0 ? trimmed : normalized.slice(-Math.max(1, Math.floor(width / 2)));
    const safeRenderedLineCount = countRenderedTuiMarkdownLines(safeTail, width);
    return {
      text: `…\n${safeTail}`,
      truncated: true,
      renderedLineCount: 1 + safeRenderedLineCount,
    };
  }

  const tail = rawLines.slice(bestStart).join('\n');
  const tailRenderedLineCount = countRenderedTuiMarkdownLines(tail, width);
  return {
    text: `…\n${tail}`,
    truncated: true,
    renderedLineCount: 1 + tailRenderedLineCount,
  };
}
