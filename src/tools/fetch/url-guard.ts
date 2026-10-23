import { lookup } from 'node:dns/promises';
import { BlockList, isIP } from 'node:net';

/**
 * SSRF guard shared by `web_fetch` and the `browser` tool.
 *
 * ponytail: resolve-then-fetch leaves a DNS-rebinding TOCTOU window; pin the
 * resolved IP in the HTTP agent if that becomes a realistic threat.
 */

const BLOCKED = new BlockList();
for (const [net, prefix] of [
  ['0.0.0.0', 8], // "this" network (0.0.0.0 reaches localhost on Linux)
  ['10.0.0.0', 8],
  ['100.64.0.0', 10], // CGNAT
  ['127.0.0.0', 8],
  ['169.254.0.0', 16], // link-local / cloud metadata
  ['172.16.0.0', 12],
  ['192.0.0.0', 24],
  ['192.168.0.0', 16],
  ['198.18.0.0', 15],
  ['224.0.0.0', 3], // multicast + reserved + broadcast
] as const) {
  BLOCKED.addSubnet(net, prefix, 'ipv4');
}
for (const [net, prefix] of [
  ['::', 96], // unspecified, loopback, IPv4-compatible
  ['64:ff9b::', 96], // NAT64 well-known prefix
  ['64:ff9b:1::', 48], // NAT64 local-use
  ['fc00::', 7], // unique local (includes fd00:ec2::254 AWS metadata)
  ['fe80::', 10], // link-local
  ['ff00::', 8], // multicast
] as const) {
  BLOCKED.addSubnet(net, prefix, 'ipv6');
}
// IPv4-mapped IPv6 (::ffff:a.b.c.d) is checked against the IPv4 rules by BlockList.

const BLOCKED_HOSTNAMES = new Set(['localhost', 'metadata.google.internal']);

type LookupFn = (hostname: string) => Promise<Array<{ address: string }>>;
const defaultLookup: LookupFn = (hostname) => lookup(hostname, { all: true, verbatim: true });
let resolveHost: LookupFn = defaultLookup;

/** @internal Test-only: inject a DNS resolver so unit tests stay offline. */
export function _setDnsLookupForTest(fn?: LookupFn): void {
  resolveHost = fn ?? defaultLookup;
}

function isBlockedIp(ip: string): boolean {
  return BLOCKED.check(ip, isIP(ip) === 6 ? 'ipv6' : 'ipv4');
}

function reject(host: string): never {
  throw new Error(`[URL Guard] URL targets an internal or reserved host: ${host}`);
}

/**
 * Asserts that a URL uses http or https and that its host string is not an
 * internal or reserved name/literal IP. Synchronous; does not resolve DNS —
 * use `assertPublicUrl` before any network request.
 */
export function assertPublicHttpUrl(raw: string): void {
  publicHostOf(raw);
}

/**
 * Asserts that a URL is http(s) and that its host — and every address it
 * resolves to — is public. Throws a descriptive Error otherwise.
 */
export async function assertPublicUrl(raw: string): Promise<void> {
  const host = publicHostOf(raw);
  if (isIP(host)) return;

  let addresses: Array<{ address: string }>;
  try {
    addresses = await resolveHost(host);
  } catch {
    throw new Error(`[URL Guard] Could not resolve host: ${host}`);
  }
  if (addresses.length === 0) throw new Error(`[URL Guard] Could not resolve host: ${host}`);
  for (const { address } of addresses) {
    if (isBlockedIp(address)) reject(`${host} (${address})`);
  }
}

/** Validates scheme and host string; returns the normalized host. */
function publicHostOf(raw: string): string {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error('[URL Guard] Invalid URL: must be http or https');
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('[URL Guard] Invalid URL: must be http or https');
  }

  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/\.+$/, '');
  if (!host || BLOCKED_HOSTNAMES.has(host) || host.endsWith('.localhost')) reject(host);
  if (isIP(host) && isBlockedIp(host)) reject(host);
  return host;
}
