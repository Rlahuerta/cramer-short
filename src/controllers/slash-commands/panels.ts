import { Container, Spacer, Text, type Component, type SlashCommand } from '@mariozechner/pi-tui';
import { discoverSkills } from '../../skills/registry.js';
import { theme } from '../../theme.js';
import type { WatchlistEntry } from '../watchlist-controller.js';
import {
  buildAsciiBar,
  buildEnrichedEntries,
  buildSnapshotDisplayData,
  calcPortfolioTotals,
  type PriceSnapshot,
} from '../watchlist-display.js';

export function createScreen(
  title: string,
  description: string,
  body: Component,
  footer?: string,
): Container {
  const container = new Container();
  if (title) {
    container.addChild(new Text(theme.bold(theme.primary(title)), 0, 0));
  }
  if (description) {
    container.addChild(new Text(theme.muted(description), 0, 0));
  }
  container.addChild(new Spacer(1));
  container.addChild(body);
  if (footer) {
    container.addChild(new Spacer(1));
    container.addChild(new Text(theme.muted(footer), 0, 0));
  }
  return container;
}

// ─── Slash command registry ───────────────────────────────────────────────────
// Keep this in sync with the handleSubmit switch below so /help always reflects
// the real set of available commands.

export const SLASH_COMMANDS: SlashCommand[] = [
  { name: 'help',      description: 'Show available commands and keyboard shortcuts' },
  { name: 'skills',    description: 'Browse available skills and how to invoke them' },
  { name: 'model',     description: 'Switch the LLM model or provider' },
  { name: 'sessions',  description: 'Browse and resume past conversations' },
  { name: 'full',      description: 'Open the full viewer for the last answer' },
  { name: 'find',      description: 'Search chat history for a keyword — or: find <keyword>' },

  { name: 'think',     description: 'Toggle Ollama extended thinking on/off (thinking models only)' },
  { name: 'watchlist', description: 'Portfolio briefing — or: add TICKER [cost] [shares] | remove TICKER | list | show TICKER | snapshot' },
  { name: 'dream',     description: 'Consolidate memory files — or: show (status), force (bypass conditions)' },
  { name: 'memory',    description: 'Show consolidated memory files (MEMORY.md + FINANCE.md)' },
  { name: 'config',    description: 'Show or set agent configuration — or: set <key> <value>' },
];

export function buildHelpPanel(): Container {
  const container = new Container();
  const COL = 10; // fixed width for the left (command/key) column

  const row = (label: string, desc: string) =>
    new Text(`  ${theme.primary(label.padEnd(COL))} ${theme.muted(desc)}`, 0, 0);

  container.addChild(new Text(theme.bold('Slash Commands'), 0, 0));
  container.addChild(new Spacer(1));
  for (const cmd of SLASH_COMMANDS) {
    container.addChild(row(`/${cmd.name}`, cmd.description ?? ''));
  }

  container.addChild(new Spacer(1));
  container.addChild(new Text(theme.bold('Keyboard Shortcuts'), 0, 0));
  container.addChild(new Spacer(1));

  const shortcuts: [string, string][] = [
    ['↑ / ↓',   'Browse input history'],
    ['Tab',      'Accept autocomplete suggestion'],
    ['Esc',      'Cancel current operation'],
    ['Ctrl+C',   'Exit Cramer-Short'],
  ];
  for (const [key, desc] of shortcuts) {
    container.addChild(row(key, desc));
  }

  container.addChild(new Spacer(1));
  container.addChild(new Text(theme.bold('Tips'), 0, 0));
  container.addChild(new Spacer(1));
  container.addChild(row('/', 'Type / to see available commands'));
  container.addChild(row('Thinking', 'Enabled automatically for qwen3, deepseek-r1, qwq models'));
  container.addChild(row('Fallback', 'Cramer-Short uses web search when financial APIs fail'));
  container.addChild(row('--deep', 'Launch with --deep flag for 40-iteration complex queries'));
  container.addChild(row('--export [path]', 'Auto-export session as Markdown on exit (omit path to auto-name)'));

  // Skills section — populated from discovered skills at render time
  const skills = discoverSkills();
  if (skills.length > 0) {
    const MAX_SKILL_DESC = 55;
    const skillCol = 22;
    const skillRow = (name: string, desc: string) => {
      const truncated = desc.length > MAX_SKILL_DESC ? `${desc.slice(0, MAX_SKILL_DESC - 1)}…` : desc;
      return new Text(`  ${theme.accent(name.padEnd(skillCol))} ${theme.muted(truncated)}`, 0, 0);
    };
    container.addChild(new Spacer(1));
    container.addChild(new Text(theme.bold('Available Skills') + theme.muted('  — invoke: "use the [name] skill for …"'), 0, 0));
    container.addChild(new Spacer(1));
    for (const skill of skills) {
      container.addChild(skillRow(skill.name, skill.description));
    }
  }

  return container;
}

export function fmtPct(n: number): string {
  const sign = n >= 0 ? '+' : '';
  return `${sign}${n.toFixed(1)}%`;
}

export function fmtMoney(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(2)}`;
}

export function colorPct(n: number, text: string): string {
  return n >= 0 ? theme.success(text) : theme.error(text);
}


export function buildWatchlistPanel(
  entries: WatchlistEntry[],
  prices: Map<string, PriceSnapshot> | null,
): Container {
  const container = new Container();
  if (entries.length === 0) {
    container.addChild(new Text(theme.muted('  No positions. Use /watchlist add TICKER [cost] [shares]'), 0, 0));
    return container;
  }

  if (prices === null) {
    // Loading state — show spinner line + stored data
    container.addChild(new Text(theme.muted('  ⏳ Fetching live prices…'), 0, 0));
    container.addChild(new Text('', 0, 0));
  }

  const enriched = prices ? buildEnrichedEntries(entries, prices) : null;
  const hasPrices = prices !== null && prices.size > 0;
  const hasPositions = entries.some((e) => e.costBasis !== undefined && e.shares !== undefined);

  // Header
  let header: string;
  if (hasPrices && hasPositions) {
    header = `  ${'TICKER'.padEnd(8)}  ${'CURRENT'.padStart(9)}  ${'DAY%'.padStart(7)}  ${'P&L'.padStart(10)}  ${'RETURN'.padStart(8)}  ${'ALLOC'.padStart(6)}  ${'CONF'.padStart(5)}`;
  } else if (hasPrices) {
    header = `  ${'TICKER'.padEnd(8)}  ${'CURRENT'.padStart(9)}  ${'DAY%'.padStart(7)}  ${'COST'.padStart(9)}  ${'SHARES'.padStart(8)}  ${'CONF'.padStart(5)}`;
  } else {
    header = `  ${'TICKER'.padEnd(8)}  ${'COST BASIS'.padStart(10)}  ${'SHARES'.padStart(8)}  ADDED`;
  }
  container.addChild(new Text(theme.bold(header), 0, 0));
  container.addChild(new Text(theme.muted('  ' + '─'.repeat(header.length - 2)), 0, 0));

  const rows = enriched ?? entries.map((e) => ({
    ticker: e.ticker, shares: e.shares, costBasis: e.costBasis, addedAt: e.addedAt,
    price: undefined, changePercent: undefined, pnl: undefined, returnPct: undefined,
    currentValue: undefined, allocPct: undefined, predictionConfidence: undefined,
  }));

  for (const row of rows) {
    const ticker = theme.primary(row.ticker.padEnd(8));
    const confStr = row.predictionConfidence !== undefined
      ? row.predictionConfidence >= 0.25
        ? theme.success('✓'.padStart(5))
        : theme.warning('⚠'.padStart(5))
      : '—'.padStart(5);

    if (hasPrices && hasPositions) {
      const price   = row.price !== undefined ? `$${row.price.toFixed(2)}`.padStart(9) : '         ';
      const day     = row.changePercent !== undefined
        ? colorPct(row.changePercent, fmtPct(row.changePercent).padStart(7))
        : '       ';
      const pnl     = row.pnl !== undefined
        ? colorPct(row.pnl, fmtMoney(row.pnl).padStart(10))
        : '          ';
      const ret     = row.returnPct !== undefined
        ? colorPct(row.returnPct, fmtPct(row.returnPct).padStart(8))
        : '        ';
      const alloc   = row.allocPct !== undefined
        ? `${row.allocPct.toFixed(0)}%`.padStart(6)
        : '      ';
      container.addChild(new Text(`  ${ticker}  ${price}  ${day}  ${pnl}  ${ret}  ${alloc}  ${confStr}`, 0, 0));
    } else if (hasPrices) {
      const price   = row.price !== undefined ? `$${row.price.toFixed(2)}`.padStart(9) : '         ';
      const day     = row.changePercent !== undefined
        ? colorPct(row.changePercent, fmtPct(row.changePercent).padStart(7))
        : '       ';
      const cost    = row.costBasis !== undefined ? `$${row.costBasis}`.padStart(9) : '         ';
      const shares  = row.shares !== undefined ? String(row.shares).padStart(8) : '        ';
      container.addChild(new Text(`  ${ticker}  ${price}  ${day}  ${cost}  ${shares}  ${confStr}`, 0, 0));
    } else {
      const cost    = row.costBasis !== undefined ? `$${row.costBasis}`.padStart(10) : '          ';
      const shares  = row.shares !== undefined ? String(row.shares).padStart(8) : '        ';
      const added   = theme.muted(row.addedAt);
      container.addChild(new Text(`  ${ticker}  ${cost}   ${shares}    ${added}`, 0, 0));
    }
  }

  // Portfolio totals row (only when we have full position data)
  if (hasPrices && hasPositions && prices) {
    const totals = calcPortfolioTotals(entries, prices);
    if (totals.totalInvested > 0) {
      container.addChild(new Text(theme.muted('  ' + '─'.repeat(header.length - 2)), 0, 0));
      const pnlColor = colorPct(totals.totalPnl, fmtMoney(totals.totalPnl).padStart(10));
      const retColor = colorPct(totals.totalReturnPct, fmtPct(totals.totalReturnPct).padStart(8));
      const totalLine = `  ${'TOTAL'.padEnd(8)}  ${fmtMoney(totals.totalCurrent).padStart(9)}  ${''.padStart(7)}  ${pnlColor}  ${retColor}`;
      container.addChild(new Text(theme.bold(totalLine), 0, 0));
    }
  }

  return container;
}

export function buildShowPanel(ticker: string, snap: PriceSnapshot): Container {
  const container = new Container();
  const w = 62;
  const bar = '─'.repeat(w);

  // Title
  const name = snap.name ? `${ticker} — ${snap.name}` : ticker;
  container.addChild(new Text(theme.bold(`  ┌─ ${name}`), 0, 0));

  // Price row
  const dayStr = snap.changePercent !== undefined
    ? colorPct(snap.changePercent, ` (${fmtPct(snap.changePercent)})`)
    : '';
  const priceRow = `  │ Price: ${theme.primary(`$${snap.price.toFixed(2)}`)}${dayStr}`;
  const rangeStr = snap.high52Week !== undefined && snap.low52Week !== undefined
    ? `  52-wk: $${snap.low52Week.toFixed(2)} – $${snap.high52Week.toFixed(2)}`
    : '';
  container.addChild(new Text(priceRow + rangeStr, 0, 0));

  if (snap.marketCap !== undefined) {
    const mcStr = snap.marketCap >= 1e12
      ? `$${(snap.marketCap / 1e12).toFixed(2)}T`
      : snap.marketCap >= 1e9
        ? `$${(snap.marketCap / 1e9).toFixed(1)}B`
        : `$${(snap.marketCap / 1e6).toFixed(0)}M`;
    container.addChild(new Text(`  │ Mkt Cap: ${mcStr}`, 0, 0));
  }

  // Ratios
  const ratios: string[] = [];
  if (snap.pe !== undefined)       ratios.push(`P/E: ${snap.pe.toFixed(1)}`);
  if (snap.pb !== undefined)       ratios.push(`P/B: ${snap.pb.toFixed(1)}`);
  if (snap.evEbitda !== undefined) ratios.push(`EV/EBITDA: ${snap.evEbitda.toFixed(1)}`);
  if (snap.peg !== undefined)      ratios.push(`PEG: ${snap.peg.toFixed(1)}`);
  if (ratios.length > 0) {
    container.addChild(new Text(theme.muted('  ├' + bar), 0, 0));
    container.addChild(new Text(`  │ ${ratios.join('   ')}`, 0, 0));
  }

  // Analyst
  if (snap.analystRating !== undefined || snap.analystAvgTarget !== undefined) {
    container.addChild(new Text(theme.muted('  ├' + bar), 0, 0));
    const rating = snap.analystRating ? theme.primary(snap.analystRating.toUpperCase()) : '';
    const target = snap.analystAvgTarget !== undefined
      ? `  Avg Target: $${snap.analystAvgTarget.toFixed(2)}`
      : '';
    const upside = snap.analystAvgTarget !== undefined && snap.price > 0
      ? colorPct(
          (snap.analystAvgTarget / snap.price - 1) * 100,
          `  (${fmtPct((snap.analystAvgTarget / snap.price - 1) * 100)})`,
        )
      : '';
    container.addChild(new Text(`  │ Analyst: ${rating}${target}${upside}`, 0, 0));
  }

  // News
  if (snap.news && snap.news.length > 0) {
    container.addChild(new Text(theme.muted('  ├' + bar), 0, 0));
    for (const item of snap.news.slice(0, 3)) {
      const date = theme.muted(`(${item.date.slice(0, 10)})`);
      const title = item.title.length > 55 ? item.title.slice(0, 52) + '…' : item.title;
      container.addChild(new Text(`  │ ${date} ${title}`, 0, 0));
    }
  }

  container.addChild(new Text(theme.muted('  └' + bar), 0, 0));
  return container;
}

export function buildSnapshotPanel(
  entries: WatchlistEntry[],
  prices: Map<string, PriceSnapshot> | null,
): Container {
  const container = new Container();

  if (prices === null) {
    container.addChild(new Text(theme.muted('  ⏳ Fetching live prices…'), 0, 0));
    return container;
  }

  const today = new Date().toISOString().slice(0, 10);
  container.addChild(new Text(theme.bold(`  Portfolio Snapshot — ${today}`), 0, 0));
  container.addChild(new Text(theme.muted('  ' + '━'.repeat(40)), 0, 0));

  const { positionEntries, watchOnlyEntries, totals, hasNoData, best, worst } =
    buildSnapshotDisplayData(entries, prices);

  // Portfolio summary totals (only when positions exist with cost basis)
  if (totals.totalInvested > 0) {
    container.addChild(new Text(`  Total Invested:  ${fmtMoney(totals.totalInvested)}`, 0, 0));
    container.addChild(new Text(`  Current Value:   ${fmtMoney(totals.totalCurrent)}`, 0, 0));
    const pnlLine = `  Total P&L:       ${fmtMoney(totals.totalPnl)}  (${fmtPct(totals.totalReturnPct)})`;
    container.addChild(new Text(colorPct(totals.totalPnl, pnlLine), 0, 0));
    container.addChild(new Text('', 0, 0));
  }

  // ASCII bar chart for allocation with Markov confidence
  if (positionEntries.length > 0) {
    container.addChild(new Text(theme.bold('  Allocation:'), 0, 0));
    const BAR_WIDTH = 22;
    for (const e of positionEntries) {
      const pct = e.allocPct!;
      const bar = buildAsciiBar(pct, BAR_WIDTH);
      const pctStr = `${pct.toFixed(0)}%`.padStart(4);
      const confIcon = e.predictionConfidence !== undefined
        ? e.predictionConfidence >= 0.25
          ? theme.success(' ✓')
          : theme.warning(' ⚠')
        : ' —';
      container.addChild(new Text(`  ${e.ticker.padEnd(6)} ${theme.primary(bar)} ${pctStr}${confIcon}`, 0, 0));
    }
    container.addChild(new Text('', 0, 0));
  }

  // Performance ranking (best / worst)
  if (best && worst) {
    container.addChild(new Text(`  Best:  ${theme.primary(best.ticker.padEnd(6))} ${colorPct(best.returnPct!, fmtPct(best.returnPct!))}`, 0, 0));
    container.addChild(new Text(`  Worst: ${theme.primary(worst.ticker.padEnd(6))} ${colorPct(worst.returnPct!, fmtPct(worst.returnPct!))}`, 0, 0));
    container.addChild(new Text('', 0, 0));
  }

  // Markov confidence histogram (only when we have confidence data)
  const confidenceValues = positionEntries
    .filter((e) => e.predictionConfidence !== undefined)
    .map((e) => e.predictionConfidence!);
  if (confidenceValues.length > 0) {
    const high = confidenceValues.filter((c) => c >= 0.40).length;
    const medium = confidenceValues.filter((c) => c >= 0.25 && c < 0.40).length;
    const low = confidenceValues.filter((c) => c < 0.25).length;
    const total = confidenceValues.length;
    const highPct = Math.round((high / total) * 100);
    const medPct = Math.round((medium / total) * 100);
    const lowPct = Math.round((low / total) * 100);
    const avgConf = confidenceValues.reduce((a, b) => a + b, 0) / total;
    container.addChild(new Text(theme.bold('  Markov Confidence Distribution:'), 0, 0));
    const highBar = buildAsciiBar(highPct, 20);
    const medBar = buildAsciiBar(medPct, 20);
    const lowBar = buildAsciiBar(lowPct, 20);
    container.addChild(new Text(`  High (≥0.40):   ${theme.success(highBar)} ${highPct}% (${high})`, 0, 0));
    container.addChild(new Text(`  Medium (0.25–0.40): ${theme.warning(medBar)} ${medPct}% (${medium})`, 0, 0));
    container.addChild(new Text(`  Low (<0.25):    ${theme.error(lowBar)} ${lowPct}% (${low})`, 0, 0));
    const avgIcon = avgConf >= 0.40 ? theme.success('✓') : avgConf >= 0.25 ? theme.warning('⚠') : theme.error('✗');
    container.addChild(new Text(`  Average: ${avgConf.toFixed(2)} ${avgIcon}`, 0, 0));
    container.addChild(new Text('', 0, 0));
  }

  // Watch-only section (tickers tracked without cost basis / shares)
  if (watchOnlyEntries.length > 0) {
    container.addChild(new Text(theme.muted('  Watching (no position):'), 0, 0));
    for (const e of watchOnlyEntries) {
      const day = e.changePercent !== undefined
        ? colorPct(e.changePercent, fmtPct(e.changePercent))
        : '';
      container.addChild(new Text(`  ${e.ticker.padEnd(8)} $${e.price!.toFixed(2)}  ${day}`, 0, 0));
    }
  }

  // "No data" fallback — only when nothing at all could be loaded
  if (hasNoData) {
    container.addChild(new Text(theme.muted('  No price data available. Add tickers with /watchlist add TICKER'), 0, 0));
    container.addChild(new Text(theme.muted('  To track P&L, provide cost basis: /watchlist add TICKER COST SHARES'), 0, 0));
  }

  return container;
}
