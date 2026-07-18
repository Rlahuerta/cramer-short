import { isIP } from 'node:net';

const PRIVATE_HOSTNAMES = new Set([
  'localhost',
  'metadata.google.internal',
]);

const PRIVATE_IPV4_PREFIXES = [
  '0.',       // "this" network / unspecified (0.0.0.0 routes to localhost on Linux)
  '127.',     // loopback
  '10.',      // private class A
  '169.254.', // link-local / cloud metadata
  '172.16.',
  '172.17.',
  '172.18.',
  '172.19.',
  '172.20.',
  '172.21.',
  '172.22.',
  '172.23.',
  '172.24.',
  '172.25.',
  '172.26.',
  '172.27.',
  '172.28.',
  '172.29.',
  '172.30.',
  '172.31.',
  '192.168.', // private class C
];

const IPV6_LOOPBACK = '::1';
const IPV6_UNSPECIFIED = '::';

function isPrivateIpv4(host: string): boolean {
  return PRIVATE_IPV4_PREFIXES.some((prefix) => host.startsWith(prefix));
}

/**
 * Extracts the embedded IPv4 from an IPv4-mapped IPv6 host, if present.
 * Handles both the dotted form (`::ffff:127.0.0.1`) and the hextet form the
 * WHATWG URL parser normalizes to (`::ffff:7f00:1`). Returns undefined otherwise.
 */
function mappedIpv4(host: string): string | undefined {
  const dotted = host.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (dotted) {
    return dotted[1];
  }
  const hextets = host.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (hextets) {
    const high = parseInt(hextets[1], 16);
    const low = parseInt(hextets[2], 16);
    return [high >> 8, high & 0xff, low >> 8, low & 0xff].join('.');
  }
  return undefined;
}

/**
 * Asserts that a URL uses http or https and targets a public (non-internal,
 * non-reserved) host. Throws a descriptive Error otherwise.
 *
 * Used by `web_fetch` and the `browser` tool to prevent SSRF and local file
 * disclosure via `file://`, `data:`, loopback, private, and link-local targets.
 *
 * Scope note: this validates the URL host string (literal IPs + a small hostname
 * blocklist). It does NOT resolve DNS, so hostnames that resolve to private IPs
 * (DNS rebinding) and exotic IP encodings (decimal/octal) are out of scope and
 * left as a documented follow-up.
 */
export function assertPublicHttpUrl(raw: string): void {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error('[URL Guard] Invalid URL: must be http or https');
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('[URL Guard] Invalid URL: must be http or https');
  }

  const hostname = parsed.hostname.toLowerCase().replace(/^\[|]$/g, ''); // strip IPv6 brackets

  if (PRIVATE_HOSTNAMES.has(hostname)) {
    throw new Error(`[URL Guard] URL targets an internal or reserved host: ${hostname}`);
  }

  if (hostname === IPV6_LOOPBACK || hostname === IPV6_UNSPECIFIED) {
    throw new Error(`[URL Guard] URL targets an internal or reserved host: ${hostname}`);
  }

  if (isIP(hostname) !== 0) {
    // It's a literal IP address.
    if (hostname.includes(':')) {
      // IPv6. Reject IPv4-mapped addresses whose embedded IPv4 is private/reserved
      // (e.g. ::ffff:127.0.0.1). Other IPv6 is allowed.
      const mapped = mappedIpv4(hostname);
      if (mapped && isPrivateIpv4(mapped)) {
        throw new Error(`[URL Guard] URL targets an internal or reserved host: ${hostname}`);
      }
      return;
    }
    if (isPrivateIpv4(hostname)) {
      throw new Error(`[URL Guard] URL targets an internal or reserved host: ${hostname}`);
    }
  }
}
