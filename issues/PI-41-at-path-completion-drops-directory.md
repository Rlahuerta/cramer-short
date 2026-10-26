# PI-41: @ completion drops the directory prefix when the path ends with a slash

- Severity: M
- Area: C7b
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/components/at-path-provider.ts:77` — `const rel =`, `src/components/at-path-provider.ts:78` — `join(dirname(rawPath), e.name)`
- Repro kind: runtime

## Summary

`AtPathAutocompleteProvider` is the `@`-path completion provider used by the TUI. After a user types `@src` and accepts the `src/` directory suggestion, the next completion pass runs with the raw prefix `src/`. The suggestion value is then built with `join(dirname('src/'), name)`; `path.dirname('src/')` is `'.'`, so the value collapses to `@cli.ts` instead of `@src/cli.ts`. Accepting that suggestion replaces the whole `@src/` token with `@cli.ts`, which points at a same-named file in the repo root (or at nothing) rather than the file the user selected.

## Root cause

`getAtFileSuggestions` derives the completion path from `dirname(rawPath)` without accounting for a raw path that already ends in `/` (`src/components/at-path-provider.ts:77-78`). A trailing slash makes `dirname` return `'.'`, discarding the directory component. The returned `value` is what `CombinedAutocompleteProvider.applyCompletion` inserts verbatim, so the wrong prefix propagates into the editor.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-41-at-path-completion-drops-directory.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-41-at-path-completion-drops-directory.ts
```

Expected: @src/ completion suggests @src/cli.ts and inserts "@src/cli.ts "

Actual: @src/ completion suggests @cli.ts and inserts "@cli.ts ", dropping the src/ directory

OBSERVED:
    suggestionValue="@cli.ts" insertedLine="@cli.ts "; directory prefix lost in suggestion value
  + actual - expected

  + '@cli.ts'
  - '@src/cli.ts'

## Impact

Selecting a completion after entering a directory path silently rewrites the attachment to a different path. The agent then reads the wrong file (a same-named file at the repo root) or reports a missing file, and the user has no indication that the directory portion was discarded. The two-step `@dir` → `@dir/` flow that the autocomplete UI encourages is the exact flow that breaks.

## Suggested fix direction (not applied)

`src/components/at-path-provider.ts:77-78` — keep the directory when the raw path already ends with `/`:

```
const rel = rawPath.endsWith('/')
  ? `${rawPath}${e.name}`
  : rawPath.includes('/')
    ? join(dirname(rawPath), e.name)
    : e.name;
```

The `rawPath + e.name` form preserves `src/`, `./`, and `/` prefixes instead of normalising them away.

## Evidence

- Reproduction script: `issues/repro/PI-41-at-path-completion-drops-directory.ts`
- Producing evidence file: `.omo/evidence/bug-hunt-reproducible-issues/task-14-C7b.txt`
- Existing tests cover only the rendered labels for `@src/` (`src/components/at-path-provider.test.ts`), not the suggestion `value` or `applyCompletion` result.
