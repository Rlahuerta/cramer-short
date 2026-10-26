# PI-51: Markdown tables with one-sided outer pipes drop a column

- Severity: M
- Area: C9
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/ui/markdown-table.ts:88` — `.filter((_, i, arr) => i > 0 && i < arr.length - 1 || arr.length === 1);`
- Repro kind: runtime

## Summary

`formatResponse()` parses markdown tables and re-renders them as box tables for every assistant answer (live TUI and terminal scrollback). GFM allows a table to omit the leading **or** the trailing pipe. For those valid tables the header parser drops one header cell, and `renderBoxTable` then emits only `colWidths.length` cells per row, so the matching data column silently disappears from the rendered answer.

## Root cause

`parseMarkdownTable` splits the header line on `|` and filters cells with `i > 0 && i < arr.length - 1 || arr.length === 1`. That expression assumes both outer pipes exist: a one-sided table still produces one empty edge cell, so the positional filter removes a real cell instead of only the empty edge. The fallback that pushes all raw headers only runs when zero headers remain, so the one-sided case is never repaired. `renderBoxTable` iterates `colWidths.map(...)`, so any data cell beyond the truncated header count is discarded.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-51-markdown-table-one-sided-pipe-drops-column.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-51-markdown-table-one-sided-pipe-drops-column.ts
```

Expected: Both header cells (A, B) and both data cells (1, 2) appear in the rendered table.

Actual: Missing cells: B, 2

OBSERVED:
    ┌───┐
    │ A │
    ├───┤
    │ 1 │
    └───┘
    input="| A | B\n|---|---|\n| 1 | 2"

## Impact

Answers that emit a GFM table with a single outer pipe render a truncated table: a whole column of financial data is visible in the model text but absent from the UI and scrollback. The user reads a misleading result with no indication that data was dropped. Severity M (misleading output with no error).

## Suggested fix direction (not applied)

`src/utils/ui/markdown-table.ts:85-88` — replace the positional filter with the same edge-cell normalisation already used for data rows at lines 114-116:

current:

```ts
const headers = headerLine
  .split('|')
  .map(cell => cell.trim())
  .filter((_, i, arr) => i > 0 && i < arr.length - 1 || arr.length === 1);
```

proposed:

```ts
const headers = headerLine.split('|').map(cell => cell.trim());
if (headers[0] === '') headers.shift();
if (headers[headers.length - 1] === '') headers.pop();
```

## Evidence

- Repro: `issues/repro/PI-51-markdown-table-one-sided-pipe-drops-column.ts` (exit 42 at HEAD, `OBSERVED:` block above).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-17-C9.txt`.
