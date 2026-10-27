# PI-26: Chunk upsert never skips because source is not selected

- Severity: M
- Area: C5
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/memory/database.ts:521` — `SELECT id, file_path, start_line, end_line, content, content_hash, embedding FROM chunks WHERE content_hash = ?`
- Repro kind: runtime

## Summary

`getChunkByHash` does not select the `source` (or `tickers`) column, so the "skip unchanged" comparison in `upsertChunk` compares `undefined` against the real source and can never be true. Every indexer sync therefore rewrites every chunk row and its FTS entry and resets `updated_at`. Because session chunks use `updated_at` as their timestamp for temporal decay, recalled conversations never age.

## Root cause

`src/memory/database.ts:518-523`:

```ts
  getChunkByHash(contentHash: string): ChunkRow | null {
    return this.db
      .query<ChunkRow>(
        'SELECT id, file_path, start_line, end_line, content, content_hash, embedding FROM chunks WHERE content_hash = ?',
      )
      .get(contentHash);
  }
```

`upsertChunk` (line 533) compares `(existing as { source?: string }).source === source` before deciding to skip. The SELECT omits `source`, so the value is always `undefined` while `source` is `'memory'` or `'sessions'`; the skip branch is dead. The row is updated, `updated_at` is set to `Date.now()`, and the FTS row is deleted and re-inserted on every sync.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-26-chunk-source-not-selected.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-26-chunk-source-not-selected.ts
```

Expected: the unchanged second upsert is skipped, leaving updated_at unchanged

Actual: the second upsert rewrote the row, so updated_at advanced

OBSERVED:
    {
      "firstInserted": true,
      "secondInserted": false,
      "rewritten": true
    }

## Impact

Two effects. (1) Index churn: every watcher-triggered sync rewrites every chunk (and its FTS row), so the SQLite file's mtime changes on every sync even when nothing changed, and the "skip unchanged" optimization never applies. (2) Ranking drift: `temporal-decay.ts` uses `updatedAt` for `sessions/*` chunks, so session memories are always treated as freshly indexed and never decay, biasing recall toward old conversations.

## Suggested fix direction (not applied)

Select the columns the comparison uses. At `src/memory/database.ts:521`, change

```ts
        'SELECT id, file_path, start_line, end_line, content, content_hash, embedding FROM chunks WHERE content_hash = ?',
```

to

```ts
        'SELECT id, file_path, start_line, end_line, content, content_hash, embedding, source, tickers FROM chunks WHERE content_hash = ?',
```

`ChunkRow` already declares `source` and `tickers`, so the unchanged-chunk early return now fires and `updated_at` is preserved.

## Evidence

Reproduction script: `issues/repro/PI-26-chunk-source-not-selected.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
