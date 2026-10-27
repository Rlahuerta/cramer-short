# PI-10: read_file deny-list misses .npmrc, .netrc and other key stores

- Severity: H
- Area: C8a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/filesystem/read-file.ts:45` — `name.startsWith('.env') ||`, `src/tools/filesystem/read-file.ts:47` — `name.endsWith('.key') ||`, `src/tools/filesystem/read-file.ts:68` — `if (isSecretPath(relative(cwd, absolutePath))) {`
- Repro kind: runtime

## Summary

`read_file` refuses `.env*`, `*.pem`, `*.key`, `.cramer-short/settings.json` and
`.cramer-short/credentials/**`, but that list is a closed set of names matched on
the basename. Other canonical credential stores that live inside a project (or in
the home directory when the agent runs there) are returned verbatim to the model:
`.npmrc` with a registry `_authToken`, `.netrc` with machine logins, `*.p12`/`*.pfx`
key bundles, `service-account*.json`, and `~/.ssh/id_rsa` / `id_ed25519` (no
extension match). The `.env` control in the same run is denied, proving the gate
exists and simply does not cover these names.

## Root cause

`isSecretPath` (read-file.ts:41-52) decides purely from the lowercased basename:
the `.env` prefix, the `.pem`/`.key` suffixes, and two exact `.cramer-short`
paths. `read_file` then only throws at read-file.ts:68 when that predicate fires.
Files whose secrecy is carried by a well-known name that is not in the list —
`.npmrc`, `.netrc`, `.git-credentials`, `id_rsa`, `id_ed25519`, `*.p12`,
`*.pfx` — pass the check and are read.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-10-read-file-secret-denylist-gaps.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-10-read-file-secret-denylist-gaps.ts
```

Expected: read_file denies .npmrc and .netrc the same way it denies .env

Actual: .npmrc read=true, .netrc read=true, .env read=false

OBSERVED:
    {
      "results": [
        {
          "path": ".npmrc",
          "read": true,
          "snippet": "//registry.npmjs.org/:_authToken=FAKE_NPM_TOKEN\n"
        },
        {
          "path": ".netrc",
          "read": true,
          "snippet": "machine example.com login bot password FAKE_NETR"
        },
        {
          "path": ".env",
          "read": false,
          "snippet": ""
        }
      ],
      "blockedControl": true,
      "leakedSecretStores": [
        ".npmrc",
        ".netrc"
      ]
    }

## Impact

Completes the prompt-injection exfiltration chain the deny-list was added to
close: page or search text that reaches the agent can instruct it to `read_file`
`.npmrc`/`.netrc`/`id_rsa` and the credential lands in the model context and in
the cross-session cache, then leaves through `web_fetch` or the chat surface.
Severity matches the original `read_file` secret-exposure finding (H): the
control exists, this is a coverage bypass.

## Suggested fix direction (not applied)

Extend the predicate in `src/tools/filesystem/read-file.ts:41-52` — current
`return (name.startsWith('.env') || name.endsWith('.pem') || ...)` → proposed add
the missing canonical names, e.g.:

```ts
name === '.npmrc' || name === '.netrc' || name === '.git-credentials' ||
name === 'id_rsa' || name === 'id_ed25519' || name === '.htpasswd' ||
name.endsWith('.p12') || name.endsWith('.pfx') ||
/^service-account.*\.json$/i.test(name)
```

and keep the existing `.env`/`.pem`/`.key`/`.cramer-short` clauses.

## Evidence

Script: `issues/repro/PI-10-read-file-secret-denylist-gaps.ts`
Evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-15-C8a.txt` (rc=42).
