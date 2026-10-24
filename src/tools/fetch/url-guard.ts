import { lookup } from 'node:dns/promises';
import http from 'node:http';
import https from 'node:https';
import { BlockList, isIP, type LookupFunction } from 'node:net';
import { Readable } from 'node:stream';

/**
 * SSRF guard shared by `web_fetch` and the `browser` tool.
 *
 * Network code passes `guardedLookup` as the socket `lookup`, so the address a
 * connection uses is the very address that was checked (no DNS-rebinding window
 * between a pre-check and a second resolution).
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
  if (!isIP(host)) await resolvePublic(host);
}

/** Resolves a hostname once; throws unless every address is public. */
async function resolvePublic(host: string): Promise<string[]> {
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
  return addresses.map((a) => a.address);
}

/**
 * `dns.lookup`-compatible resolver for http/https/net `lookup` options: resolves
 * once and fails the connection unless every address is public. Not consulted
 * for IP-literal hosts, so pair it with `assertPublicHttpUrl`.
 */
export const guardedLookup = ((hostname: string, options: unknown, callback?: unknown) => {
  const cb = (typeof options === 'function' ? options : callback) as (
    err: Error | null,
    address?: string | Array<{ address: string; family: number }>,
    family?: number,
  ) => void;
  const all = typeof options === 'object' && options !== null && (options as { all?: boolean }).all === true;
  const host = hostname.toLowerCase().replace(/\.+$/, '');
  Promise.resolve()
    .then(() => {
      checkHost(host);
      return resolvePublic(host);
    })
    .then((addresses) => {
      if (all) cb(null, addresses.map((address) => ({ address, family: isIP(address) })));
      else cb(null, addresses[0], isIP(addresses[0]));
    })
    .catch((err: Error) => cb(err));
}) as unknown as LookupFunction;

/**
 * Minimal fetch over node http(s) that connects only to guard-validated
 * addresses (literal IPs via `assertPublicHttpUrl`, hostnames via
 * `guardedLookup`). Never follows redirects — callers re-check each hop.
 */
export async function fetchPinned(
  url: string,
  init: { headers?: Record<string, string>; signal?: AbortSignal } = {},
): Promise<Response> {
  assertPublicHttpUrl(url);
  const client = url.startsWith('https:') ? https : http;
  return new Promise((resolve, rejectPromise) => {
    const req = client.request(
      url,
      {
        // identity: node does not auto-decompress like fetch does.
        headers: { ...init.headers, 'Accept-Encoding': 'identity' },
        signal: init.signal,
        lookup: guardedLookup,
      },
      (res) => {
        const headers = new Headers();
        for (const [name, value] of Object.entries(res.headers)) {
          for (const v of [value ?? []].flat()) headers.append(name, v);
        }
        const status = res.statusCode ?? 502;
        const noBody = status === 204 || status === 205 || status === 304;
        if (noBody) res.resume();
        const body = noBody ? null : (Readable.toWeb(res) as unknown as ReadableStream);
        resolve(new Response(body, { status, statusText: res.statusMessage, headers }));
      },
    );
    req.on('error', rejectPromise);
    req.end();
  });
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
  checkHost(host);
  return host;
}

function checkHost(host: string): void {
  if (!host || BLOCKED_HOSTNAMES.has(host) || host.endsWith('.localhost')) reject(host);
  if (isIP(host) && isBlockedIp(host)) reject(host);
}
