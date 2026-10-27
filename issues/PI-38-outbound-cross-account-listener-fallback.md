# PI-38: Outbound send borrows another account's socket when the requested account is missing

- Severity: M
- Area: C7a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/gateway/channels/whatsapp/outbound.ts:74` — `function getActive(accountId?: string): ActiveListener {`, `src/gateway/channels/whatsapp/outbound.ts:81` — `const first = listeners.values().next().value as ActiveListener | undefined;`, `src/gateway/channels/whatsapp/outbound.ts:142` — `const active = getActive(params.accountId);`
- Repro kind: runtime

## Summary

`getActive(accountId)` falls back to the first registered listener when the requested account has no active socket. With two linked WhatsApp accounts, a reply processed for a disconnected account is sent through the other account's socket, so it leaves from the wrong phone number and bypasses the other account's allowlist.

## Root cause

`getActive()` looks up the requested account (`outbound.ts:75-80`) but, when the lookup misses, continues to `listeners.values().next()` (`outbound.ts:81-85`) and returns whichever socket was registered first. Both `sendMessageWhatsApp()` (`outbound.ts:142`) and `sendComposing()` (`outbound.ts:166`) call `getActive()` with the caller's account id and then send through the returned socket. An account whose socket is between reconnects (`runtime.ts:91-92` clears it) therefore borrows the other account's session.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-38-outbound-cross-account-listener-fallback.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-38-outbound-cross-account-listener-fallback.ts
```

Expected: sendMessageWhatsApp({accountId:"work"}) throws when the work socket is absent

Actual: getActive("work") falls back to the first socket (personal) and sends from the wrong number script=PI-38-outbound-cross-account-listener-fallback.ts

OBSERVED:
    requested account=work (no listener registered); sockets used=["personal"]
    result={"messageId":"msg-personal","toJid":"15551234567@s.whatsapp.net"} error=null

## Impact

- Reply identity confusion: the recipient sees a message from the personal number, and the personal account's allowlist is never consulted for that recipient.
- Heartbeats and error replies routed to the disconnected account are silently delivered from the remaining account instead of failing/reporting "account unavailable".
- The stale `.cramer-short/gateway-debug.log` and console still attribute the send to the requested account, hiding the misroute.

## Suggested fix direction (not applied)

Make an explicit account selection strict, in `src/gateway/channels/whatsapp/outbound.ts`:

- `src/gateway/channels/whatsapp/outbound.ts:75-80` — replace with:
  ```ts
  if (accountId) {
    const found = listeners.get(accountId);
    if (!found) {
      throw new Error(`No active WhatsApp listener for account ${accountId}. Run dexter gateway run.`);
    }
    return found;
  }
  ```

This keeps the first-listener fallback only for callers that pass no account id (`getActive(undefined)`), and makes a requested-but-disconnected account fail loudly.

## Evidence

- Script: `issues/repro/PI-38-outbound-cross-account-listener-fallback.ts`
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-13-C7a.txt`
- Related code: `src/gateway/channels/whatsapp/outbound.ts:66-86`, `src/gateway/channels/whatsapp/runtime.ts:56-92`
