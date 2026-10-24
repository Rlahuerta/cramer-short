import { afterEach, describe, expect, it } from 'bun:test';
import { createServer, type Server } from 'node:http';
import { connect } from 'node:net';
import { _setDnsLookupForTest } from '../fetch/url-guard.js';
import { startGuardProxy } from './guard-proxy.js';

/** Local target that counts hits — reaching it means the proxy was bypassed. */
async function startTarget(): Promise<{ server: Server; port: number; hits: () => number }> {
  let hits = 0;
  const server = createServer((_req, res) => {
    hits++;
    res.end('internal-secret');
  });
  server.on('connection', () => hits++);
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  return { server, port: (server.address() as { port: number }).port, hits: () => hits };
}

/** Sends a raw request to the proxy and returns everything it answers. */
function rawProxyRequest(proxyPort: number, payload: string): Promise<string> {
  return new Promise((resolve, reject) => {
    let out = '';
    const socket = connect(proxyPort, '127.0.0.1', () => socket.write(payload));
    socket.on('data', (d) => {
      out += d;
    });
    socket.on('end', () => resolve(out));
    socket.on('close', () => resolve(out));
    socket.on('error', reject);
    setTimeout(() => {
      socket.destroy();
      resolve(out);
    }, 2000);
  });
}

afterEach(() => _setDnsLookupForTest());

describe('browser guard proxy', () => {
  for (const [label, host] of [
    ['a hostname resolving to an internal address', 'internal.example'],
    ['an internal IP literal', '127.0.0.1'],
  ] as const) {
    it(`refuses CONNECT (HTTPS/WSS tunnels) to ${label}`, async () => {
      _setDnsLookupForTest(async () => [{ address: '127.0.0.1' }]);
      const target = await startTarget();
      const proxy = await startGuardProxy();
      try {
        const port = new URL(proxy.server).port;
        const reply = await rawProxyRequest(
          Number(port),
          `CONNECT ${host}:${target.port} HTTP/1.1\r\nHost: ${host}:${target.port}\r\n\r\n`,
        );
        expect(reply).toStartWith('HTTP/1.1 403');
        expect(target.hits()).toBe(0);
      } finally {
        await proxy.close();
        target.server.close();
      }
    });

    it(`refuses plain HTTP (and ws://) requests to ${label}`, async () => {
      _setDnsLookupForTest(async () => [{ address: '127.0.0.1' }]);
      const target = await startTarget();
      const proxy = await startGuardProxy();
      try {
        const port = new URL(proxy.server).port;
        const reply = await rawProxyRequest(
          Number(port),
          `GET http://${host}:${target.port}/ HTTP/1.1\r\nHost: ${host}:${target.port}\r\nConnection: close\r\n\r\n`,
        );
        expect(reply).toStartWith('HTTP/1.1 403');
        expect(reply).not.toContain('internal-secret');
        expect(target.hits()).toBe(0);
      } finally {
        await proxy.close();
        target.server.close();
      }
    });
  }
});
