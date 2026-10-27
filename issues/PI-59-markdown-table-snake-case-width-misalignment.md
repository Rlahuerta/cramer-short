# PI-59: Table width treats snake_case underscores as italic markers

- Severity: L
- Area: C9
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/ui/markdown-table.ts:40` — `.replace(/_([^_\s][^_]*)_/g, '$1');`
- Repro kind: runtime

## Summary

`stripMarkdownFormatting()` computes the visible width of a table cell by stripping markdown markers before padding. Its underscore-italic regex has no word-boundary guard, so it also strips the `_bar_` inside a snake_case token such as `foo_bar_baz`. The later italic transform deliberately refuses to style that token (`(?<!\w)_..._(?!\w)`), so the characters stay visible while the padding was computed for a shorter string. The rendered data row overflows the box border.

## Root cause

`cellDisplayWidth` measures `foo_bar_baz` as 9 characters (it removes `_bar_`), but `transformItalic` uses `(?<!\w)_([^_\s][^_\n]*)_(?!\w)` and does not italicise an underscore run that is adjacent to word characters. The raw 11-character text is emitted into a column sized for 9, so the row is two characters wider than its border. `renderBoxTable` pads every cell from `cellDisplayWidth`, so any snake_case identifier in a table misaligns its row.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-59-markdown-table-snake-case-width-misalignment.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-59-markdown-table-snake-case-width-misalignment.ts
```

Expected: All rendered table lines have equal visible width (box borders align).

Actual: The data row is 2 columns wider than the border.

OBSERVED:
    ┌───────────┬──────┐
    │ metric    │ note │
    ├───────────┼──────┤
    │ foo_bar_baz │ x    │
    └───────────┴──────┘
    line widths: 20, 20, 20, 22, 20

## Impact

Financial tables routinely use snake_case keys (e.g. `price_to_earnings`, `ebitda_margin`); every such cell widens its row past the border, so the table is visually broken in the TUI and in exported scrollback. Cosmetic but deterministic and user-visible. Severity L.

## Suggested fix direction (not applied)

`src/utils/ui/markdown-table.ts:40` — give the width-stripping regex the same word-boundary guard the italic transform already uses at line 276:

current:

```ts
.replace(/_([^_\s][^_]*)_/g, '$1');   // _italic_ → content
```

proposed:

```ts
.replace(/(?<!\w)_([^_\s][^_\n]*)_(?!\w)/g, '$1');   // _italic_ → content
```

## Evidence

- Repro: `issues/repro/PI-59-markdown-table-snake-case-width-misalignment.ts` (exit 42 at HEAD, `OBSERVED:` block above).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-17-C9.txt`.
