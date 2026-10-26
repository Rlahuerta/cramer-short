# PI-57: Dependency-cruiser CLI layering rule misses top-level CLI files

- Severity: L
- Area: C9
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `.dependency-cruiser.cjs:37` — `path: '^src/(cli|components|controllers)/',`, `.dependency-cruiser.cjs:48` — `path: '^src/(cli|components|controllers)/',`
- Repro kind: static

## Summary

The `tools-no-import-cli` and `agent-no-import-cli` rules are meant to stop the tools and agent layers importing the CLI layer, but their `to.path` regex requires a trailing slash after `cli`. The top-level CLI entry files (`src/cli.ts`, `src/cli-schedule.ts`, `src/cli-forecast-lab.ts`, `src/cli-replay-label.ts`) therefore do not match either rule and can be imported without a depcruise error.

## Root cause

Both rules match `^src/(cli|components|controllers)/`. `components` and `controllers` are directories, but `cli` exists both as a directory-less set of files (`src/cli.ts`, `src/cli-*.ts`) and as `src/controllers/cli-*`; the rule only covers the directory form. `bun run depcruise` (CI step "Check dependency boundaries") is the only enforcement point, so the uncovered files are an unguarded path for the documented layering contract.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-57-depcruise-cli-rule-misses-top-level-cli-files.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-57-depcruise-cli-rule-misses-top-level-cli-files.ts
```

Expected: Every CLI-layer entry file (src/cli.ts, src/cli-schedule.ts, ...) matches both layering rules.

Actual: 8 uncovered CLI-layer path(s); the rules only match paths with a trailing slash.

OBSERVED:
    37: path: '^src/(cli|components|controllers)/',
    48: path: '^src/(cli|components|controllers)/',
    tools-no-import-cli does not match src/cli.ts
    tools-no-import-cli does not match src/cli-schedule.ts
    tools-no-import-cli does not match src/cli-forecast-lab.ts
    tools-no-import-cli does not match src/cli-replay-label.ts
    agent-no-import-cli does not match src/cli.ts
    agent-no-import-cli does not match src/cli-schedule.ts
    agent-no-import-cli does not match src/cli-forecast-lab.ts
    agent-no-import-cli does not match src/cli-replay-label.ts

## Impact

A future `src/tools/**` or `src/agent/**` import of a top-level CLI module (which drags the TUI/controllers dependency chain into the tool layer) passes CI's boundary check silently. No current import violates the rule, so this is a latent control gap. Severity L.

## Suggested fix direction (not applied)

`.dependency-cruiser.cjs:37` and `.dependency-cruiser.cjs:48` — cover the file form as well as the directory form:

current:

```js
path: '^src/(cli|components|controllers)/',
```

proposed (both occurrences):

```js
path: '^src/(cli(?:-|[.])|components/|controllers/)',
```

## Evidence

- Repro: `issues/repro/PI-57-depcruise-cli-rule-misses-top-level-cli-files.ts` (exit 42 at HEAD, `OBSERVED:` block above).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-17-C9.txt`.
