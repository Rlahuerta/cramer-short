# PI-28: Ticker-filtered insight search applies the FTS limit before filtering

- Severity: M
- Area: C5
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/memory/financial-store.ts:105` — `let results = this.db.searchInsightsFts(query, (options?.maxResults ?? 6) * 2).map(rowToInsight);`, `src/memory/database.ts:431` — `ORDER BY rank LIMIT ?`
- Repro kind: runtime

## Summary

`FinancialMemoryStore.search` asks FTS for `maxResults * 2` rows across all tickers, then applies the ticker filter in JavaScript. When another ticker owns those top-ranked rows, the requested ticker's matching insight is never fetched and the filtered search returns an empty list — even though `recallByTicker` shows the insight exists.

## Root cause

`src/memory/financial-store.ts:105-114`:

```ts
    let results = this.db.searchInsightsFts(query, (options?.maxResults ?? 6) * 2).map(rowToInsight);
    results = results.filter((i) => !isExpired(i));
    if (options?.namespace !== undefined) {
      results = results.filter((i) => (i.namespace ?? null) === options.namespace!);
    }
    if (options?.tags?.length) {
      results = results.filter((i) => options.tags!.some((t) => i.tags.includes(t)));
    }
    if (options?.ticker) {
      const upper = options.ticker.toUpperCase();
      results = results.filter((i) => i.ticker.toUpperCase() === upper);
    }
```

`searchInsightsFts` applies `LIMIT ?` inside SQLite (`src/memory/database.ts:431`), so the row window is chosen before any of these filters. The ticker (and namespace/tag) filters only see whatever survived that window.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-28-insight-search-limit-before-ticker-filter.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-28-insight-search-limit-before-ticker-filter.ts
```

Expected: the ticker-filtered search returns the stored AAPL insight

Actual: the ticker-filtered search returned 0 results although AAPL has a match

OBSERVED:
    {
      "unfilteredTickers": [
        "BBB",
        "BBB",
        "BBB"
      ],
      "tickerFilteredCount": 0,
      "tickerFilteredTickers": []
    }

## Impact

Ticker-scoped insight search (used by recall paths that constrain to one symbol) silently returns nothing when another symbol's insights dominate the BM25 ranking. The insight exists and `recallByTicker` can find it, but the search-based path reports no match, so the agent may answer without the stored context.

## Suggested fix direction (not applied)

Push the ticker filter into SQLite before the limit.

- `src/memory/database.ts:426`: `searchInsightsFts(query: string, maxResults: number, ticker?: string)`; when `ticker` is provided use

```ts
      ? this.db
          .query<{ insight_id: number; rank: number }>(
            'SELECT insight_id, bm25(financial_insights_fts) AS rank FROM financial_insights_fts WHERE financial_insights_fts MATCH ? AND insight_id IN (SELECT id FROM financial_insights WHERE UPPER(ticker) = UPPER(?)) ORDER BY rank LIMIT ?',
          )
          .all(sanitized, ticker, maxResults)
```

  and keep the existing query as the no-ticker branch.
- `src/memory/financial-store.ts:105`: pass the filter through — `this.db.searchInsightsFts(query, (options?.maxResults ?? 6) * 2, options?.ticker)`.

The JS ticker filter can stay as a harmless belt-and-braces check. (The namespace filter has the same shape; it is not part of this reproduction.)

## Evidence

Reproduction script: `issues/repro/PI-28-insight-search-limit-before-ticker-filter.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
