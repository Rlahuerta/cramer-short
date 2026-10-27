import { afterEach, beforeEach, describe, test, expect, mock } from 'bun:test';
import { useNoNetwork } from '../../utils/test-no-network.js';
import { _setHttpFetchForTest, webFetchTool } from './web-fetch.js';
import { _setDnsLookupForTest } from './url-guard.js';
import { extractText as extractPdfText } from 'unpdf';
// Warm the HTML extractor's dynamic imports outside individual test timers.
import '@mozilla/readability';
import 'linkedom';

useNoNetwork();

// web_fetch uses its own pinned transport (not globalThis.fetch), so block it
// here too; tests opt into a stub, or into the real transport with stubbed DNS.
beforeEach(() =>
  _setHttpFetchForTest(async (url: string) => {
    throw new Error(`unit test blocked network call to ${new URL(url).host}`);
  }),
);
afterEach(() => _setHttpFetchForTest());

// Minimal valid 1-page PDF with the text "Hello PDF World"
const MINIMAL_PDF_BASE64 =
  'JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCA2MTIgNzkyXQovQ29udGVudHMgNCAwIFIgL1Jlc291cmNlcyA8PCAvRm9udCA8PCAvRjEgNSAwIFIgPj4gPj4gPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCA0NCA+PgpzdHJlYW0KQlQKL0YxIDEyIFRmCjEwMCA3MDAgVGQKKEhlbGxvIFBERiBXb3JsZCkgVGoKRVQKZW5kc3RyZWFtCmVuZG9iago1IDAgb2JqCjw8IC9UeXBlIC9Gb250IC9TdWJ0eXBlIC9UeXBlMSAvQmFzZUZvbnQgL0hlbHZldGljYSA+PgplbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAKMDAwMDAwMDExNSAwMDAwMCBuIAowMDAwMDAwMjY3IDAwMDAwIG4gCjAwMDAwMDAzNjEgMDAwMDAgbiAKdHJhaWxlcgo8PCAvU2l6ZSA2IC9Sb290IDEgMCBSID4+CnN0YXJ0eHJlZgo0NDMKJSVFT0YK';

function unwrapExternalContent(value: string | undefined): string | undefined {
  if (!value) {
    return value;
  }
  const start = value.indexOf('<<<EXTERNAL_UNTRUSTED_CONTENT>>>');
  const end = value.indexOf('<<<END_EXTERNAL_UNTRUSTED_CONTENT>>>');
  if (start === -1 || end === -1 || end <= start) {
    return value;
  }
  const contentStart = value.indexOf('\n---\n', start);
  if (contentStart === -1 || contentStart >= end) {
    return value;
  }
  return value.slice(contentStart + '\n---\n'.length, end).trim();
}

// ---------------------------------------------------------------------------
// Blocked-domain guard — these tests confirm that known anti-scraping domains
// are rejected immediately (no network request) with an actionable message.
// ---------------------------------------------------------------------------

describe('web_fetch blocked-domain guard', () => {
  async function invoke(url: string): Promise<string> {
    return webFetchTool.invoke({ url }) as Promise<string>;
  }

  test('rejects finance.yahoo.com with a hint to use financial_search', async () => {
    const result = await invoke('https://finance.yahoo.com/quote/VWS.CO/');
    expect(result).toContain('Blocked domain');
    expect(result).toContain('financial_search');
  });

  test('rejects all finance.yahoo.com sub-paths', async () => {
    const result = await invoke('https://finance.yahoo.com/quote/AAPL/financials/');
    expect(result).toContain('Blocked domain');
  });

  test('rejects bloomberg.com with a hint to use web_search', async () => {
    const result = await invoke('https://www.bloomberg.com/news/articles/2025-01-01/example');
    expect(result).toContain('Blocked domain');
    expect(result).toContain('web_search');
  });

  test('rejects wsj.com with a hint about the paywall', async () => {
    const result = await invoke('https://www.wsj.com/articles/some-article');
    expect(result).toContain('Blocked domain');
    expect(result).toContain('paywalled');
  });

  test('does NOT reject legitimate news URLs', async () => {
    // cnbc.com, reuters.com, ft.com etc. should pass through to fetch
    // (DNS and fetch are stubbed: we only verify the guard does not fire).
    _setDnsLookupForTest(async () => [{ address: '93.184.216.34' }]);
    const transport = mock(
      async (_url: string) => new Response('<p>ok</p>', { status: 200, headers: { 'content-type': 'text/html' } }),
    );
    _setHttpFetchForTest(transport);
    try {
      const result = await invoke('https://httpbin.org/get');
      expect(result).not.toContain('Blocked domain');
      expect(transport).toHaveBeenCalled();
    } finally {
      _setDnsLookupForTest();
    }
  });
});

// ---------------------------------------------------------------------------
// PDF extraction — unit-tests for the unpdf integration.
// The web_fetch HTTP layer is not exercised here; we test the PDF library
// directly to confirm text extraction works for financial report PDFs.
// ---------------------------------------------------------------------------

describe('PDF text extraction (unpdf)', () => {
  test('extracts text from a minimal synthetic PDF', async () => {
    const buf = Buffer.from(MINIMAL_PDF_BASE64, 'base64');
    const { totalPages, text } = await extractPdfText(new Uint8Array(buf), { mergePages: true });
    expect(totalPages).toBe(1);
    expect(text.trim()).toBe('Hello PDF World');
  });

  test('reports correct page count', async () => {
    const buf = Buffer.from(MINIMAL_PDF_BASE64, 'base64');
    const { totalPages } = await extractPdfText(new Uint8Array(buf), { mergePages: true });
    expect(totalPages).toBeGreaterThan(0);
  });

  test('returns an array of strings when mergePages is false', async () => {
    const buf = Buffer.from(MINIMAL_PDF_BASE64, 'base64');
    const { totalPages, text } = await extractPdfText(new Uint8Array(buf), { mergePages: false });
    expect(Array.isArray(text)).toBe(true);
    expect(text).toHaveLength(totalPages);
    expect((text as string[]).join(' ').trim()).toContain('Hello PDF World');
  });

  test('mergePages=true returns a single string (not an array)', async () => {
    const buf = Buffer.from(MINIMAL_PDF_BASE64, 'base64');
    const { text } = await extractPdfText(new Uint8Array(buf), { mergePages: true });
    expect(typeof text).toBe('string');
  });
});

// ---------------------------------------------------------------------------
// web_fetch PDF content-type detection — mock fetch so we can verify that
// application/pdf responses are routed through the PDF extractor, not the
// HTML extractor.  We use a synthetic PDF buffer as the response body.
// ---------------------------------------------------------------------------

describe('web_fetch PDF content-type routing', () => {
  const originalFetch = globalThis.fetch;

  async function invokePdfUrl(url: string, contentType = 'application/pdf'): Promise<string> {
    const pdfBuf = Buffer.from(MINIMAL_PDF_BASE64, 'base64');
    // Stub global fetch to return a fake PDF response for any URL
    const fetchPdf: typeof globalThis.fetch = Object.assign(
      async (_url: Parameters<typeof globalThis.fetch>[0], _init?: Parameters<typeof globalThis.fetch>[1]) =>
        new Response(pdfBuf, {
          status: 200,
          headers: { 'content-type': contentType },
        }),
      { preconnect: originalFetch.preconnect },
    );
    _setHttpFetchForTest(fetchPdf);
    _setDnsLookupForTest(async () => [{ address: '93.184.216.34' }]);
    try {
      return await (webFetchTool.invoke({ url }) as Promise<string>);
    } finally {
      _setHttpFetchForTest();
      _setDnsLookupForTest();
    }
  }

  test('extracts PDF text when content-type is application/pdf', async () => {
    const result = await invokePdfUrl('https://example.com/report.pdf');
    expect(result).toContain('Hello PDF World');
  });

  test('uses pdf extractor label in result for application/pdf responses', async () => {
    const result = await invokePdfUrl('https://example.com/report.pdf');
    // extractor field should be "pdf" not "htmlToMarkdown" or "readability"
    expect(result).toContain('"extractor":"pdf"');
  });

  test('detects PDF by .pdf URL extension even if content-type is octet-stream', async () => {
    const result = await invokePdfUrl(
      'https://example.com/vestas-q2-2025-interim-report.pdf',
      'application/octet-stream',
    );
    expect(result).toContain('Hello PDF World');
  });

  test('does not use PDF extractor for regular HTML URLs', async () => {
    const htmlBuf = '<html><head><title>Test</title></head><body><p>Hello HTML</p></body></html>';
    const fetchHtml: typeof globalThis.fetch = Object.assign(
      async () => new Response(htmlBuf, { status: 200, headers: { 'content-type': 'text/html' } }),
      { preconnect: originalFetch.preconnect },
    );
    _setHttpFetchForTest(fetchHtml);
    try {
      const result = await (webFetchTool.invoke({ url: 'https://example.com/' }) as Promise<string>);
      const payload = JSON.parse(result) as { data?: { extractor?: string; text?: string } };
      expect(payload.data?.extractor).not.toBe('pdf');
      expect(payload.data?.text).toContain('<<<EXTERNAL_UNTRUSTED_CONTENT>>>');
      expect(unwrapExternalContent(payload.data?.text)).toContain('Hello HTML');
    } finally {
      _setHttpFetchForTest();
    }
  });
});

describe('web_fetch HTML extractor labeling', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    _setHttpFetchForTest();
  });

  async function invokeHtml(
    html: string,
    url: string,
  ): Promise<{ data?: { extractor?: string; text?: string; title?: string } }> {
    const fetchHtml: typeof globalThis.fetch = Object.assign(
      async () => new Response(html, { status: 200, headers: { 'content-type': 'text/html' } }),
      { preconnect: originalFetch.preconnect },
    );
    _setHttpFetchForTest(fetchHtml);
    return JSON.parse(await (webFetchTool.invoke({ url }) as Promise<string>)) as {
      data?: { extractor?: string; text?: string; title?: string };
    };
  }

  test('labels simple html fragments as htmlToMarkdown', async () => {
    const payload = await invokeHtml('<p>Fragment fallback</p>', 'https://example.com/fragment');
    expect(payload.data?.extractor).toBe('htmlToMarkdown');
    expect(payload.data?.text).toContain('Fragment fallback');
  });

  test('keeps short article-like html labeled as readability', async () => {
    const payload = await invokeHtml(
      [
        '<!doctype html>',
        '<html><head><title>Short Article</title></head><body>',
        '<main><article><h1>Short Article</h1>',
        '<p>Lead paragraph with enough substance for readability extraction.</p>',
        '<p>Second paragraph confirms this is more than a bare fragment.</p>',
        '</article></main></body></html>',
      ].join(''),
      'https://example.com/article',
    );
    expect(payload.data?.extractor).toBe('readability');
    expect(payload.data?.title).toContain('<<<EXTERNAL_UNTRUSTED_CONTENT>>>');
    expect(unwrapExternalContent(payload.data?.title)).toBe('Short Article');
    expect(unwrapExternalContent(payload.data?.text)).toContain('Lead paragraph');
  });
});

// ---------------------------------------------------------------------------
// SSRF guard — internal/private hosts must be rejected before any fetch.
// ---------------------------------------------------------------------------

describe('web_fetch SSRF guard', () => {
  async function invoke(url: string): Promise<string> {
    return webFetchTool.invoke({ url }) as Promise<string>;
  }

  test('rejects file:// URLs', async () => {
    const result = await invoke('file:///etc/passwd');
    expect(result).toContain('must be http or https');
  });

  test('rejects 127.0.0.1', async () => {
    const result = await invoke('http://127.0.0.1:11434/api/tags');
    expect(result).toContain('internal or reserved');
  });

  test('rejects 169.254.169.254 (cloud metadata)', async () => {
    const result = await invoke('http://169.254.169.254/latest/meta-data/');
    expect(result).toContain('internal or reserved');
  });

  test('rejects localhost', async () => {
    const result = await invoke('http://localhost:3000/');
    expect(result).toContain('internal or reserved');
  });

  test('does NOT reject a public URL (passes through to fetch or blocked-domain guard)', async () => {
    _setHttpFetchForTest(async () => new Response('<p>ok</p>', { status: 200, headers: { 'content-type': 'text/html' } }));
    const result = await invoke('https://example.com/').catch((e: Error) => e.message);
    expect(result).not.toContain('internal or reserved');
    expect(result).not.toContain('must be http or https');
  });
});

// ---------------------------------------------------------------------------
// SSRF guard — every redirect hop is re-checked.
// ---------------------------------------------------------------------------

describe('web_fetch SSRF guard (redirects)', () => {
  const originalFetch = globalThis.fetch;
  afterEach(() => {
    _setHttpFetchForTest();
    _setDnsLookupForTest();
  });

  test('rejects a redirect from a public host to an internal one', async () => {
    _setDnsLookupForTest(async () => [{ address: '93.184.216.34' }]);
    const fetched: string[] = [];
    _setHttpFetchForTest(async (url: string) => {
      fetched.push(url);
      return new Response(null, { status: 302, headers: { location: 'http://127.0.0.1:8080/admin' } });
    });
    await expect(webFetchTool.invoke({ url: 'https://example.com/redirect' })).rejects.toThrow(
      /internal or reserved/,
    );
    expect(fetched).toEqual(['https://example.com/redirect']);
  });
});

describe('web_fetch default transport', () => {
  afterEach(() => _setDnsLookupForTest());

  test('never reaches a server when the hostname resolves to an internal address', async () => {
    const { createServer } = await import('node:http');
    let hits = 0;
    const server = createServer((_req, res) => {
      hits++;
      res.end('internal-secret');
    });
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
    const port = (server.address() as { port: number }).port;
    _setDnsLookupForTest(async () => [{ address: '127.0.0.1' }]);
    _setHttpFetchForTest(); // real pinned transport; DNS is stubbed so nothing leaves the host
    try {
      const result = await (webFetchTool.invoke({ url: `http://internal.example:${port}/` }) as Promise<string>).catch(
        (e: Error) => e.message,
      );
      expect(result).toMatch(/internal or reserved/);
      expect(hits).toBe(0);
    } finally {
      server.close();
    }
  });
});

describe('web_fetch cache key (PI-45)', () => {
  function stubPages(calls: string[]): void {
    _setHttpFetchForTest(async (url: string) => {
      calls.push(url);
      return new Response(`<p>body of ${url}</p>`, { status: 200, headers: { 'content-type': 'text/html' } });
    });
  }

  async function fetchText(url: string): Promise<{ text: string; cached?: boolean }> {
    return (JSON.parse(await (webFetchTool.invoke({ url }) as Promise<string>)) as { data: { text: string; cached?: boolean } })
      .data;
  }

  test('URLs that differ only by path or query case are fetched separately', async () => {
    const calls: string[] = [];
    stubPages(calls);

    const upper = await fetchText('https://case-path.example.com/Report?Ticker=AAPL');
    const lower = await fetchText('https://case-path.example.com/report?Ticker=AAPL');
    const lowerQuery = await fetchText('https://case-path.example.com/Report?ticker=aapl');

    expect(calls).toEqual([
      'https://case-path.example.com/Report?Ticker=AAPL',
      'https://case-path.example.com/report?Ticker=AAPL',
      'https://case-path.example.com/Report?ticker=aapl',
    ]);
    expect(upper.text).toContain('/Report?Ticker=AAPL');
    expect(lower.text).toContain('/report?Ticker=AAPL');
    expect(lower.cached).toBeUndefined();
    expect(lowerQuery.text).toContain('/Report?ticker=aapl');
  });

  test('URLs that differ only by scheme or host case still share one cache entry', async () => {
    const calls: string[] = [];
    stubPages(calls);

    await fetchText('https://case-host.example.com/Report');
    const again = await fetchText('HTTPS://Case-Host.Example.COM/Report');

    expect(calls).toHaveLength(1);
    expect(again.cached).toBe(true);
  });
});
