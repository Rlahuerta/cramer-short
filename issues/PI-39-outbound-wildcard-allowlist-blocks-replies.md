# PI-39: Setup wizard's wildcard allowlist is accepted inbound but blocks every outbound reply

- Severity: M
- Area: C7a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/gateway/channels/whatsapp/outbound.ts:122` — `.filter((entry) => entry !== '*')`, `src/gateway/channels/whatsapp/outbound.ts:125` — `throw new Error('Outbound blocked: no explicit allowFrom recipient configured.');`, `src/gateway/index.ts:68` — `if (input === '*') {`, `src/gateway/index.ts:84` — `allowFrom: phones,`
- Repro kind: runtime

## Summary

`gateway login` bot mode explicitly offers "type `*` to allow anyone" and persists that wildcard in the account/top-level allowlists. Inbound access control honours the wildcard (`access-control.ts:127`), but `assertOutboundAllowed()` strips `*` and then throws because no explicit recipient remains. Every DM from an allowed-by-wildcard sender is therefore accepted and then silently never answered.

## Root cause

`promptSetupMode()` accepts `*` (`src/gateway/index.ts:68-71`) and writes it into both allowlists (`src/gateway/index.ts:84-89`). The outbound guard filters the wildcard out (`.filter((entry) => entry !== '*')`, `outbound.ts:122`) and treats the resulting empty list as a hard block (`outbound.ts:124-126`). Inbound and outbound disagree about what the same allowlist means; the outbound side is the one that decides whether a reply is ever sent.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-39-outbound-wildcard-allowlist-blocks-replies.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-39-outbound-wildcard-allowlist-blocks-replies.ts
```

Expected: inbound allowed=true and outbound allowed (same "*" allowlist)

Actual: inbound accepts the sender, outbound rejects every recipient -> gateway never replies script=PI-39-outbound-wildcard-allowlist-blocks-replies.ts

OBSERVED:
    inbound allowed=true (allowFrom=["*"])
    outbound assertOutboundAllowed threw: Outbound blocked: no explicit allowFrom recipient configured.

## Impact

- A supported setup path ("allow anyone") yields a gateway that receives messages but never sends a single reply. The failure is silent from the user's side (no error is surfaced in the chat; only a debug log).
- Same mismatch exists for `dmPolicy: 'open'` installs that use `allowFrom: ['*']` instead of leaving the list empty.

## Suggested fix direction (not applied)

Teach the outbound guard the same wildcard semantics as inbound, in `src/gateway/channels/whatsapp/outbound.ts`:

- `src/gateway/channels/whatsapp/outbound.ts:121-129` — replace the strict-only block with:
  ```ts
  const allowWildcard = account.allowFrom.includes('*');
  const explicitAllowedRecipients = account.allowFrom
    .filter((entry) => entry !== '*')
    .map(normalizeE164);
  if (!allowWildcard && explicitAllowedRecipients.length === 0) {
    throw new Error('Outbound blocked: no explicit allowFrom recipient configured.');
  }
  if (!allowWildcard && !explicitAllowedRecipients.includes(recipientE164)) {
    throw new Error(`Outbound blocked: ${recipientE164} is not in allowFrom.`);
  }
  ```

If the intended policy is to keep outbound strict, the alternative minimal change is to make `promptSetupMode()` reject `*` with a clear message instead of persisting an unusable list; either change removes the inbound/outbound contradiction.

## Evidence

- Script: `issues/repro/PI-39-outbound-wildcard-allowlist-blocks-replies.ts`
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-13-C7a.txt`
- Related code: `src/gateway/access-control.ts:127-130`, `src/gateway/config.ts:212-222`, `src/gateway/gateway.ts:185-193`
