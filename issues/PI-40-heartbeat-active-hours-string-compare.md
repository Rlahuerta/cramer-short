# PI-40: Heartbeat active-hours gate compares HH:MM strings lexicographically

- Severity: M
- Area: C7a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/gateway/heartbeat/runner.ts:54` — `const currentTime = timeFormatter.format(now); // "HH:MM"`, `src/gateway/heartbeat/runner.ts:56` — `return currentTime >= activeHours.start && currentTime <= activeHours.end;`
- Repro kind: runtime

## Summary

`isWithinActiveHours()` formats the current time as `"HH:MM"` and compares the strings with `>=`/`<=`. Any unpadded configured bound (for example `"9:30"`, which is accepted by the config schema) makes the comparison wrong: `"10:05" >= "9:30"` is `false`, so the heartbeat is skipped for the entire morning window instead of running.

## Root cause

`Intl.DateTimeFormat(... hour: '2-digit')` yields zero-padded `"10:05"`, while the configured `activeHours.start`/`end` are free-form strings (`src/gateway/config.ts:37-38` defaults are `"09:30"`/`"16:00"` but any string is accepted). The window test at `runner.ts:56` compares those raw strings, which orders lexicographically, not chronologically. This is the deep-review §7 minor "Heartbeat active hours are compared as strings"; it is still present at HEAD.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-40-heartbeat-active-hours-string-compare.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-40-heartbeat-active-hours-string-compare.ts
```

Expected: the active-hours gate passes at 10:05 inside the 9:30–16:00 window

Actual: string compare "10:05" >= "9:30" is false, so every heartbeat in this window is skipped script=PI-40-heartbeat-active-hours-string-compare.ts

OBSERVED:
    now=2026-10-26T14:05:00Z (Mon 10:05 America/New_York), activeHours={"start":"9:30","end":"16:00"}
    gateway-debug.log contains "outside active hours": true

## Impact

- Any user who writes `"9:30"` (or `"9:00"`–`"9:59"` generally) as the start time gets no heartbeat alerts until 10:00, and `"8:00"`/`"7:00"` style starts fail for every hour whose first digit is larger. The scheduler reports nothing — the heartbeat simply never fires.
- The default config (`"09:30"`) masks the bug, so it only affects customized active hours.

## Suggested fix direction (not applied)

Compare minutes instead of strings, in `src/gateway/heartbeat/runner.ts`:

- `src/gateway/heartbeat/runner.ts:54-56` — replace the comparison with:
  ```ts
  const currentTime = timeFormatter.format(now); // "HH:MM"
  const toMinutes = (value: string): number => {
    const [h, m] = value.split(':').map(Number);
    return (h ?? 0) * 60 + (m ?? 0);
  };

  return toMinutes(currentTime) >= toMinutes(activeHours.start) && toMinutes(currentTime) <= toMinutes(activeHours.end);
  ```

(The helper may live next to `isWithinActiveHours`; `hour12: false` must stay so the formatted value uses a 0–23 hour range.)

## Evidence

- Script: `issues/repro/PI-40-heartbeat-active-hours-string-compare.ts`
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-13-C7a.txt`
- Related code: `src/gateway/config.ts:30-47`, `src/gateway/heartbeat/runner.ts:23-57`
