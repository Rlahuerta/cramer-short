# PI-45: web_fetch cache key lowercases the URL and serves another page

- Severity: M
- Area: C8a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/fetch/web-fetch.ts:335` — `fetch:${params.url}:${params.extractMode}:${params.maxChars}`, `src/tools/fetch/cache.ts:23` — `return value.trim().toLowerCase();`
- Repro kind: runtime

## Summary

`runWebFetch` builds its cache key from the raw URL and then lowercases the whole
key string. URLs that differ only by case share one entry, so fetching
`https://host/Report` and then `https://host/report` returns the first page's
body without touching the network — and reports `url: https://host/Report` for
the second request. Case-sensitive paths and case-sensitive query values
(`?Ticker=AAPL` vs `?ticker=aapl`) collide the same way. The tool's own result
object shows the mismatched `url`/`finalUrl`, so the caller cannot tell that the
content belongs to a different page.

## Root cause

`normalizeCacheKey` lowercases its input (cache.ts:23) — correct for a generic
key, wrong for a URL, where the path and query are case-significant. The
`web_fetch` cache key embeds `${params.url}` before normalisation
(web-fetch.ts:334-336), so the URL's case is discarded. `readCache` then returns
the first entry, `payload` is spread with `{ ...cached.value, cached: true }`, and
the transport is never called for the second URL.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-45-web-fetch-cache-key-url-case-collapse.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-45-web-fetch-cache-key-url-case-collapse.ts
```

Expected: the case-distinct URL is fetched on its own and returns its own body

Actual: transport calls=["https://example.com/Report"], second.cached=true, second.url=https://example.com/Report

OBSERVED:
    {
      "transportCalls": [
        "https://example.com/Report"
      ],
      "firstUrl": "https://example.com/Report",
      "firstHasUppercaseBody": true,
      "secondUrl": "https://example.com/Report",
      "secondCached": true,
      "secondHasLowercaseBody": false,
      "secondRepeatsUppercaseBody": true
    }

## Impact

Wrong page content is attributed to the requested URL for 15 minutes (the default
TTL). For financial research this can mean one instrument's page, or one
case-sensitive API response (`?Ticker=AAPL` vs `?ticker=aapl`), being reported as
another, with no error and no cache-miss signal.

## Suggested fix direction (not applied)

Do not case-fold the URL part of the key. In `src/tools/fetch/web-fetch.ts:334-336`
— current `const cacheKey = normalizeCacheKey(...)` → proposed:

```ts
const cacheKey = `fetch:${params.url}:${params.extractMode}:${params.maxChars}`;
```

Leave `normalizeCacheKey` unchanged for the callers that want trimming/case
folding, e.g. ticker keys.

## Evidence

Script: `issues/repro/PI-45-web-fetch-cache-key-url-case-collapse.ts`
Evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-15-C8a.txt` (rc=42).
