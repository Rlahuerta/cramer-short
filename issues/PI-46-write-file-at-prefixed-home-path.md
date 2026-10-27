# PI-46: write_file rejects @~ home paths

- Severity: M
- Area: C8a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/filesystem/write-file.ts:46` — `const isHomePath = input.path.startsWith('~/') || input.path === '~';`, `src/tools/filesystem/write-file.ts:47` — `const root = isHomePath ? homedir() : cwd;`, `src/tools/filesystem/utils/path-utils.ts:21` — `return filePath.startsWith('@') ? filePath.slice(1) : filePath;`
- Repro kind: runtime

## Summary

The shared path helper treats a leading `@` as decoration: `expandPath` strips it
before expanding `~` (`normalizeAtPrefix`, asserted by the unit test
`expandPath('@~') === homedir()`), and the TUI/README present home-file requests
as `@~/reports/my-analysis.md`. `write_file` however decides the sandbox root from
the raw string, so `@~/reports/x.md` is not recognised as a home path, is resolved
against the project cwd, and is rejected with `Path escapes sandbox root`. The
identical path without the `@` succeeds. The failure is a false sandbox violation
on a documented input form.

## Root cause

`writeFileTool` computes `isHomePath` from `input.path.startsWith('~/') ||
input.path === '~'` (write-file.ts:46) instead of from the expanded string that
`resolveToCwd`/`expandPath` will actually use. `expandPath` strips `@`
(path-utils.ts:20-22), so the two layers disagree: `assertSandboxPath` resolves
`@~/reports/x.md` to `<home>/reports/x.md` while `root` is still `process.cwd()`.
When cwd is a subdirectory of home (a normal checkout at `~/cramer-short`), the
`relative()` check reports an escape and throws.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-46-write-file-at-prefixed-home-path.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-46-write-file-at-prefixed-home-path.ts
```

Expected: @~/reports/analysis.md is accepted exactly like ~/reports/analysis.md

Actual: @~/reports/analysis.md ok=false (Path escapes sandbox root: @~/reports/analysis.md); ~/reports/analysis.md ok=true

OBSERVED:
    {
      "expandPathAtTilde": "<HOME>",
      "documented": {
        "path": "@~/reports/analysis.md",
        "ok": false,
        "detail": "Path escapes sandbox root: @~/reports/analysis.md"
      },
      "control": {
        "path": "~/reports/analysis.md",
        "ok": true,
        "detail": "written"
      }
    }

## Impact

A model that echoes the user's `@~/reports/...` path into `write_file` gets a
security-flavoured error that is unrelated to its request, and the report is not
written. The agent then either retries blindly or reports that it cannot write
outside the workspace, which is false. Normal `~/reports/` writes and cwd-relative
writes are unaffected, so the workaround is to drop the `@`.

## Suggested fix direction (not applied)

Recognise the `@` prefix when deciding the sandbox root, in
`src/tools/filesystem/write-file.ts:46` — current
`const isHomePath = input.path.startsWith('~/') || input.path === '~';` →
proposed `const isHomePath = /^@?~(\/|$)/.test(input.path);`.

## Evidence

Script: `issues/repro/PI-46-write-file-at-prefixed-home-path.ts`
Evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-15-C8a.txt` (rc=42).
