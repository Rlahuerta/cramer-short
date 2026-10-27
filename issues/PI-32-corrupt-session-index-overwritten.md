# PI-32: Corrupt session index is overwritten by the next session write

- Severity: M
- Area: C5
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/session-store.ts:91` — `} catch {`, `src/utils/session-store.ts:92` — `return { sessions: [] };`, `src/utils/session-store.ts:164` — `await writeIndex(index, baseDir);`
- Repro kind: runtime

## Summary

`readIndex` maps a parse or read failure on `.cramer-short/sessions/_index.json` to an empty index. `createSession` then appends to that empty object and writes it back, replacing every prior index entry. Session JSON files survive on disk, but every earlier session disappears from the selector because the index is the only listing.

## Root cause

`src/utils/session-store.ts:86-94`:

```ts
async function readIndex(baseDir: string): Promise<SessionIndex> {
  const path = indexPath(baseDir);
  if (!existsSync(path)) return { sessions: [] };
  try {
    return JSON.parse(await readFile(path, 'utf-8')) as SessionIndex;
  } catch {
    return { sessions: [] };
  }
}
```

A corrupt or partially written index is swallowed exactly like a missing one. `createSession` (line 154) pushes the new entry onto the empty array and `writeIndex` (line 164) atomically replaces the file, so the previous index content is unrecoverable.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-32-corrupt-session-index-overwritten.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-32-corrupt-session-index-overwritten.ts
```

Expected: the unparseable _index.json is preserved (for example renamed to a .corrupt-* backup)

Actual: the old index was overwritten; the OLD-SESSION-MARKER entry is no longer present in any file

OBSERVED:
    {
      "corruptIndexRetained": false,
      "listedCount": 1
    }

## Impact

After an unclean shutdown or a manual edit leaves `_index.json` invalid, the next session creation silently hides every prior session from the session selector. The session files remain, but the UI exposes only the newest session; recovering the others requires hand-editing the index.

## Suggested fix direction (not applied)

Preserve the unreadable index before returning an empty one. In `src/utils/session-store.ts`:

- line 3: `import { existsSync } from 'node:fs';` → `import { existsSync, renameSync } from 'node:fs';`
- line 91-92 (catch body):

```ts
  } catch {
    try {
      renameSync(path, `${path}.corrupt-${Date.now()}`);
    } catch { /* keep in place */ }
    return { sessions: [] };
  }
```

The new index is then written to a clean file while the previous entries remain available in the `.corrupt-*` backup (the same move `config.ts` uses for invalid settings).

## Evidence

Reproduction script: `issues/repro/PI-32-corrupt-session-index-overwritten.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
