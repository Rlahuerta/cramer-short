import { createServer, request, type IncomingHttpHeaders } from 'node:http';
import { connect } from 'node:net';
import { assertPublicHttpUrl, guardedLookup } from '../fetch/url-guard.js';

/**
 * Local forward proxy for the Playwright browser. Chromium sends all traffic
 * here (plain HTTP, and HTTPS/WSS/ws:// tunnelled via CONNECT; Playwright forces
 * `<-loopback>` so loopback is proxied too). Every upstream connection resolves
 * through `guardedLookup`, so the browser can only reach the exact public
 * address the guard checked: no DNS rebinding, and WebSockets are covered.
 */
export async function startGuardProxy(): Promise<{ server: string; close(): Promise<void> }> {
  const proxy = createServer((req, res) => {
    const url = req.url ?? '';
    try {
      assertPublicHttpUrl(url);
    } catch (err) {
      res.writeHead(403).end(String((err as Error).message));
      return;
    }
    const headers: IncomingHttpHeaders = { ...req.headers };
    delete headers['proxy-connection'];
    delete headers['proxy-authorization'];
    const upstream = request(url, { method: req.method, headers, lookup: guardedLookup }, (up) => {
      res.writeHead(up.statusCode ?? 502, up.headers);
      up.pipe(res);
    });
    upstream.on('error', (err) => {
      if (!res.headersSent) res.writeHead(403);
      res.end(err.message);
    });
    req.pipe(upstream);
  });

  proxy.on('connect', (req, socket, head) => {
    const deny = () => socket.end('HTTP/1.1 403 Forbidden\r\n\r\n');
    let target: URL;
    try {
      target = new URL(`https://${req.url ?? ''}`);
      assertPublicHttpUrl(target.href);
    } catch {
      deny();
      return;
    }
    const upstream = connect(
      { host: target.hostname.replace(/^\[|\]$/g, ''), port: Number(target.port || 443), lookup: guardedLookup },
      () => {
        socket.write('HTTP/1.1 200 Connection Established\r\n\r\n');
        upstream.write(head);
        upstream.pipe(socket);
        socket.pipe(upstream);
      },
    );
    upstream.on('error', deny);
    socket.on('error', () => upstream.destroy());
  });

  await new Promise<void>((resolve) => proxy.listen(0, '127.0.0.1', resolve));
  const { port } = proxy.address() as { port: number };
  return {
    server: `http://127.0.0.1:${port}`,
    close: () =>
      new Promise<void>((resolve) => {
        proxy.closeAllConnections();
        proxy.close(() => resolve());
      }),
  };
}
