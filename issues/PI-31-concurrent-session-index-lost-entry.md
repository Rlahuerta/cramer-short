# PI-31: Concurrent session creation drops an index entry

- Severity: M
- Area: C5
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/session-store.ts:154` — `const index = await readIndex(baseDir);`, `src/utils/session-store.ts:164` — `await writeIndex(index, baseDir);`
- Repro kind: runtime

## Summary

`createSession` updates `_index.json` with an unserialized read-modify-write. Two sessions created in the same tick both read the index before either writes, so the second write replaces the first entry. Both session JSON files exist, but one of them is missing from the session list.

## Root cause

`src/utils/session-store.ts:152-164`:

```ts
  const index = await readIndex(baseDir);
  index.sessions.push({
    ...
  });
  await writeIndex(index, baseDir);
```

There is no mutex or queue around the index (unlike `MemoryStore.appendMemoryFile`, which uses `withFileLock`). `readIndex` is asynchronous, so two concurrent `createSession` calls both observe the pre-write index and each writes its own single-entry array; the last rename wins and the other session's index entry is lost.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-31-concurrent-session-index-lost-entry.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-31-concurrent-session-index-lost-entry.ts
```

Expected: both concurrently created sessions appear in the index (2 entries)

Actual: only 1 of 2 created sessions are listed

OBSERVED:
    {
      "createdCount": 2,
      "listedCount": 1
    }

## Impact

When two sessions start concurrently (parallel gateway messages, a controller restart racing a new query), one session silently vanishes from the selector while its file is orphaned on disk. Listing is the only way the UI reaches sessions, so the user believes the conversation was lost.

## Suggested fix direction (not applied)

Serialize the index read-modify-write. Add a module-level queue to `src/utils/session-store.ts` (above `writeIndex`):

```ts
let indexQueue: Promise<unknown> = Promise.resolve();
function serializeIndex<T>(fn: () => Promise<T>): Promise<T> {
  const run = indexQueue.then(fn, fn);
  indexQueue = run.then(() => undefined, () => undefined);
  return run;
}
```

Then wrap the RMW in `createSession` (lines 154-164) and `saveSession` (lines 181-199):

```ts
  await serializeIndex(async () => {
    const index = await readIndex(baseDir);
    // ...existing push/find + mutate...
    await writeIndex(index, baseDir);
  });
```

The second caller now reads the index after the first write has landed, so both entries survive.

## Evidence

Reproduction script: `issues/repro/PI-31-concurrent-session-index-lost-entry.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
