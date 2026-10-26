# PI-49: pi-tui postinstall patch removes setRawMode even when its anchor misses

- Severity: M
- Area: C9
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `patches/pi-tui-terminal-fix.cjs:92` — `content = content.replace(originalBlock, replacementBlock);`, `patches/pi-tui-terminal-fix.cjs:100` — `content = content.replace(oldSetRawMode, '');`
- Repro kind: static

## Summary

The postinstall patch rewrites `@mariozechner/pi-tui/dist/terminal.js` to move `setRawMode(wasRaw)` before `pause()`. The insertion is an unconditional `content.replace(originalBlock, replacementBlock)`, and the script then removes the original `setRawMode` call whenever that regex matches. If a pi-tui update changes the insertion anchor but keeps the original call, the patch inserts nothing and still deletes the working call, leaving the terminal in raw mode after exit. If both regexes miss, the script writes the unchanged file and prints the success line.

## Root cause

The result of the insertion replace is discarded, so there is no check that the anchor matched before the removal and the write. The existing guard only checks the comment marker (`Restore raw mode state BEFORE pausing stdin`) to decide whether the patch was already applied; when the marker is absent the script assumes the layout is patchable. `package.json` depends on `@mariozechner/pi-tui` with a caret range, so a compatible minor update is enough to change the anchor.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-49-pi-tui-patch-removes-setrawmode-without-anchor-match.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-49-pi-tui-patch-removes-setrawmode-without-anchor-match.ts
```

Expected: The script compares the patch result with the pre-patch content and exits non-zero when the anchor missed.

Actual: The insertion result is discarded and the old setRawMode block is removed unconditionally, so a missed anchor deletes the fix.

OBSERVED:
    92: content = content.replace(originalBlock, replacementBlock);
    97: const oldSetRawMode = /\n\s+if \(process\.stdin\.setRawMode\) \{\n\s+process\.stdin\.setRawMode\(this\.wasRaw\);\n\s+\}/;
    99: if (oldSetRawMode.test(content)) {
    100: content = content.replace(oldSetRawMode, '');
    103: fs.writeFileSync(terminalPath, content, 'utf8');

## Impact

On any pi-tui upgrade that moves the anchor while retaining the original call, `bun install` reports a successful patch while the terminal.js fix has been removed — reproducing the original "Ctrl+C leaves the shell with no echo / Kitty escapes" bug, now with a green postinstall. The failure is silent, so it is only noticed when a user's terminal is left broken after exit. Severity M (silent regression in the install path; conditional on dependency drift).

## Suggested fix direction (not applied)

`patches/pi-tui-terminal-fix.cjs:92` — capture the insertion result and abort when the anchor did not match, before touching the old call:

current:

```js
content = content.replace(originalBlock, replacementBlock);
```

proposed:

```js
const patched = content.replace(originalBlock, replacementBlock);
if (patched === content) {
  console.error('❌ pi-tui terminal.js layout changed: anchor not found; refusing to patch');
  process.exit(1);
}
content = patched;
```

## Evidence

- Repro: `issues/repro/PI-49-pi-tui-patch-removes-setrawmode-without-anchor-match.ts` (exit 42 at HEAD, `OBSERVED:` block above).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-17-C9.txt`.
