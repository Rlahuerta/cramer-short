# PI-27: Keyword-only search drops the weakest BM25 match

- Severity: M
- Area: C5
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/memory/database.ts:731` — `score: range > 0 ? (maxRank - row.rank) / range : 1,`, `src/memory/search.ts:96` — `.filter((entry) => entry.finalScore >= minScore)`
- Repro kind: runtime

## Summary

`searchKeyword` min-max normalises the BM25 ranks of a batch to [0, 1]. The weakest row is therefore always assigned exactly 0, and `hybridSearch` immediately filters on `finalScore >= minScore` (default 0.1). In keyword-only mode (no embedding client available), a query with two distinct matches returns only one result.

## Root cause

`src/memory/database.ts:726-733`:

```ts
    const minRank = Math.min(...rows.map((r) => r.rank));
    const maxRank = Math.max(...rows.map((r) => r.rank));
    const range = maxRank - minRank;

    return rows.slice(0, maxResults).map((row) => ({
      chunkId: row.chunk_id,
      score: range > 0 ? (maxRank - row.rank) / range : 1,
    }));
```

The normalisation reserves 0 for the worst hit of every batch. `src/memory/search.ts:96` then applies the configured `minScore` (default `0.1` from `DEFAULT_CONFIG`) to the merged score. With `weights = {vector: 0, text: 1}` (no vector candidates), a keyword score of 0 becomes a final score of 0 and is dropped — so `maxResults` is never reachable in keyword-only mode when the FTS batch contains more than one row.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-27-keyword-only-second-match-dropped.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-27-keyword-only-second-match-dropped.ts
```

Expected: both keyword matches are returned when maxResults=2 and minScore=0.1

Actual: only 1 of 2 matching chunks were returned; the weakest BM25 hit scored 0 and was filtered

OBSERVED:
    {
      "keywordMatches": [
        1,
        0
      ],
      "returnedCount": 1,
      "returnedScores": [
        1
      ]
    }

## Impact

Whenever embeddings are unavailable (no embedding provider configured, embedding call failed, or keyword-only search is requested), every multi-match query loses its weakest match. With the default `maxResults: 6`, at most five keyword hits are ever returned, and a query whose only relevant hit happens to rank last in its small batch can lose it entirely. The loss is silent: the caller cannot distinguish "one match" from "one match plus a filtered second".

## Suggested fix direction (not applied)

Do not map the weakest BM25 hit to 0. At `src/memory/database.ts:731` change

```ts
      score: range > 0 ? (maxRank - row.rank) / range : 1,
```

to

```ts
      score: range > 0 ? 0.1 + 0.9 * ((maxRank - row.rank) / range) : 1,
```

Every FTS hit now scores at least 0.1, so the default `minScore` no longer discards real matches while still ranking the strongest hit at 1.0.

## Evidence

Reproduction script: `issues/repro/PI-27-keyword-only-second-match-dropped.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
