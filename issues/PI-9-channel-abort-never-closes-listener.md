# PI-9: Gateway shutdown hangs because channel abort never closes the listener

- Severity: H
- Area: C7a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/gateway/channels/whatsapp/runtime.ts:90` — `const closeReason = await listener.onClose;`, `src/gateway/channels/whatsapp/runtime.ts:49` — `while (!params.abortSignal.aborted) {`, `src/gateway/channels/manager.ts:110` — `await task.catch((err) => {`
- Repro kind: runtime

## Summary

`monitorWhatsAppChannel()` checks `abortSignal.aborted` only between connection attempts. While it is connected it blocks on `await listener.onClose`, and nothing in the function reacts to the signal, so `manager.stopAccount()` → `stopAll()` → `gateway.stop()` never resolves. `SIGINT`/`SIGTERM` therefore hang until the WhatsApp socket happens to disconnect on its own.

## Root cause

The loop condition at `runtime.ts:49` is the only abort check. After `monitorWebInbox()` returns, `runtime.ts:90` awaits `listener.onClose` with no abort listener registered, and the WhatsApp plugin defines no `gateway.stopAccount` (`plugin.ts:19-42`) to close the socket itself. `manager.stopAccount()` aborts the controller and then awaits the still-pending account task (`manager.ts:110`), so the shutdown path deadlocks.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-9-channel-abort-never-closes-listener.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-9-channel-abort-never-closes-listener.ts
```

Expected: aborting the account signal closes the active listener so monitorWhatsAppChannel resolves

Actual: runtime awaits listener.onClose forever; manager.stopAccount()/stopAll() hangs (Ctrl+C hang) script=PI-9-channel-abort-never-closes-listener.ts

OBSERVED:
    abort() called at t=50ms; monitorWhatsAppChannel still pending at t=450ms (settled=false)

## Impact

- `bun run gateway` cannot be stopped with Ctrl+C or SIGTERM while connected; operators must kill the process, which can interrupt in-flight replies.
- `GatewayService.stop()` is awaited by the CLI shutdown handler (`src/gateway/index.ts:129-134`), so the process does not exit and no cleanup after `manager.stopAll()` runs.
- Deep-review §7 already recorded "First Ctrl+C hangs"; the reconnect-wait part was made abortable, but this connected-state hang is still present at HEAD.

## Suggested fix direction (not applied)

Close the listener when the account signal aborts, in `src/gateway/channels/whatsapp/runtime.ts`:

- After `src/gateway/channels/whatsapp/runtime.ts:71` (`setActiveWebListener(params.accountId, listener.sock);`) add:
  ```ts
  const onAbort = () => { void listener.close(); };
  params.abortSignal.addEventListener('abort', onAbort, { once: true });
  ```
- `src/gateway/channels/whatsapp/runtime.ts:90` — keep the wait, then remove the listener:
  ```ts
  const closeReason = await listener.onClose;
  params.abortSignal.removeEventListener('abort', onAbort);
  ```

`listener.close()` already resolves `onClose` (`inbound.ts:348-352`), so the loop then falls through, observes the aborted signal, and returns.

## Evidence

- Script: `issues/repro/PI-9-channel-abort-never-closes-listener.ts`
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-13-C7a.txt`
- Related code: `src/gateway/channels/manager.ts:94-115`, `src/gateway/channels/whatsapp/inbound.ts:345-358`, `src/gateway/index.ts:129-134`
