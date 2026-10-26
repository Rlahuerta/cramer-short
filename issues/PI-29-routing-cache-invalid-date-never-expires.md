# PI-29: A routing cache entry with an invalid date never expires

- Severity: M
- Area: C5
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/api-routing-cache.ts:42` — `return Date.now() - new Date(entry.updatedAt).getTime() > TTL_MS;`
- Repro kind: runtime

## Summary

`isStale` computes `Date.now() - new Date(updatedAt).getTime()`. When `updatedAt` cannot be parsed, the result is `NaN`, and `NaN > TTL_MS` is always false. A corrupted routing entry is therefore treated as fresh forever and keeps steering that ticker to a backend the cache was supposed to re-probe after 30 days.

## Root cause

`src/utils/api-routing-cache.ts:41-43`:

```ts
function isStale(entry: RoutingEntry): boolean {
  return Date.now() - new Date(entry.updatedAt).getTime() > TTL_MS;
}
```

`new Date('not-a-date').getTime()` is `NaN`; every comparison with `NaN` is false, so `isStale` returns false and `getPreferredApi` (line 74) returns the preference. The same file swallows `JSON.parse` errors in `load` and starts empty, so a malformed date is the only corruption that survives into the routing decision.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-29-routing-cache-invalid-date-never-expires.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-29-routing-cache-invalid-date-never-expires.ts
```

Expected: an entry whose updatedAt cannot be parsed is treated as stale and returns null

Actual: the invalid-date entry returned "fmp" instead of null

OBSERVED:
    {
      "invalidDatePreference": "fmp",
      "expiredPreference": null
    }

## Impact

A routing preference written by an older version, a manual edit, or a partially corrupted file can pin a ticker to a failing backend indefinitely. `clearRoutingEntry` is never called automatically (it is only an explicit helper), so the stale preference persists across sessions and defeats the 30-day TTL that is documented at the top of the module.

## Suggested fix direction (not applied)

Reject a non-finite timestamp. Replace the body of `isStale` at `src/utils/api-routing-cache.ts:41-43` with:

```ts
function isStale(entry: RoutingEntry): boolean {
  const updatedAt = new Date(entry.updatedAt).getTime();
  return !Number.isFinite(updatedAt) || Date.now() - updatedAt > TTL_MS;
}
```

An unparseable date now counts as stale, so the next call re-probes and rewrites the entry.

## Evidence

Reproduction script: `issues/repro/PI-29-routing-cache-invalid-date-never-expires.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
