# PI-54: Cross-session cache key truncation collides distinct entries

- Severity: L
- Area: C5
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/cross-session-cache.ts:32` — `return key.replace(/[/:?&]/g, '_').slice(0, 100);`
- Repro kind: runtime

## Summary

`sanitizeKey` builds the cache filename from the key with a 100-character cap and no uniqueness suffix. Two tool-cache keys that share their first 100 characters map to the same file; the second save replaces the first and the first key disappears from the cross-session cache.

## Root cause

`src/utils/cross-session-cache.ts:30-33`:

```ts
function sanitizeKey(key: string): string {
  return key.replace(/[/:?&]/g, '_').slice(0, 100);
}
```

`saveCacheToDisk` writes `join(CACHE_DIR, `${sanitizedKey}.json`)` (line 102). The long tool cache keys built from JSON-stringified arguments routinely exceed 100 characters (the repository's own `.cramer-short/cache/` filenames show keys longer than that), so two different argument sets with a common prefix collide and the later write wins. `loadCacheFromDisk` keys the returned map by the stored `entry.key`, so the overwritten key is simply absent — not even a wrong value.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-54-cross-session-cache-key-truncation-collision.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-54-cross-session-cache-key-truncation-collision.ts
```

Expected: two distinct keys that share a 100-character prefix are stored independently

Actual: only 1 of 2 entries survived; key A present=false, key B present=true

OBSERVED:
    {
      "cachedKeys": 1,
      "keyAPresent": false,
      "keyBPresent": true
    }

## Impact

Cache-only: the affected query is re-fetched instead of being served from disk. There is no wrong-value path (the surviving entry keeps its own `key`), but the cross-session cache silently loses entries for long, prefix-sharing tool arguments, which are common for JSON-heavy finance tools.

## Suggested fix direction (not applied)

Keep a hash of the full key in the filename. In `src/utils/cross-session-cache.ts`:

- add `import { createHash } from 'node:crypto';`
- replace `sanitizeKey` (lines 30-33) with:

```ts
function sanitizeKey(key: string): string {
  const stem = key.replace(/[/:?&]/g, '_').replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 100);
  return `${stem}-${createHash('md5').update(key).digest('hex').slice(0, 8)}`;
}
```

Distinct keys now always produce distinct filenames. (Existing cache files become unreachable and are re-written on demand; the cache is disposable and TTL-bound.)

## Evidence

Reproduction script: `issues/repro/PI-54-cross-session-cache-key-truncation-collision.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
