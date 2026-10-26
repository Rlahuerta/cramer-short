# PI-43: Theme semantic colors render black because chalk.hex() is applied to color names

- Severity: M
- Area: C7b
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/theme.ts:19` — `chalk.hex(color)(text)`, `src/theme.ts:7` — `success: 'green',`
- Repro kind: runtime

## Summary

`src/theme.ts` routes every palette entry through `chalk.hex()`, but four entries are CSS color *names* (`green`, `red`, `yellow`, `cyan`), not hex strings. chalk's `hex()` parser cannot parse a name and emits `rgb(0, 0, 0)`. As a result `theme.success`, `theme.error`, `theme.warning`, and `theme.accent` all paint text black. Every success checkmark, tool error, limit warning, denial notice, P&L sign, and skill label is rendered in black instead of its intended semantic color.

## Root cause

The palette mixes hex strings with named colors, and the single `fg` helper unconditionally calls `chalk.hex(color)` (`src/theme.ts:19`). For the four named entries (`src/theme.ts:7-9` and `:12`), `chalk.hex('green')`/`hex('red')`/`hex('yellow')`/`hex('cyan')` produce the fallback `\x1b[38;2;0;0;0m` (RGB 0,0,0) rather than the ANSI named colors.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-43-theme-named-colors-render-black.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-43-theme-named-colors-render-black.ts
```

Expected: success/error/warning/accent each render a non-black semantic foreground color

Actual: all four render rgb(0,0,0) because chalk.hex() is applied to named colors

OBSERVED:
    blackForegrounds=["success","error","warning","accent"] success="\u001b[38;2;0;0;0mX\u001b[39m" error="\u001b[38;2;0;0;0mX\u001b[39m" warning="\u001b[38;2;0;0;0mX\u001b[39m" accent="\u001b[38;2;0;0;0mX\u001b[39m"

  4 !== 0

## Impact

On the dark terminal background the TUI targets, black-on-black text is unreadable: tool errors (`ToolEventComponent.setError`), limit warnings, denial notices, and debug-panel error lines effectively disappear. On lighter backgrounds the semantic distinction is still lost, so success/error/warning can no longer be told apart by color and P&L signs lose their positive/negative cue.

## Suggested fix direction (not applied)

Give the four named entries hex values so the existing `chalk.hex()` path is valid (`src/theme.ts:7-12`):

```
success: '#00c853',
error: '#ff5252',
warning: '#ffb300',
accent: '#4dd0e1',
```

Alternatively dispatch in `fg`: use `chalk.hex(color)` only when `color.startsWith('#')`, otherwise `chalk[color]`.

## Evidence

- Reproduction script: `issues/repro/PI-43-theme-named-colors-render-black.ts`
- Producing evidence file: `.omo/evidence/bug-hunt-reproducible-issues/task-14-C7b.txt`
- No theme color test exists; `chalk.hex('green')` returning rgb(0,0,0) is verifiable directly with chalk's `Instance({ level: 3 })`.
