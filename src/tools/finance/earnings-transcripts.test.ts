import { describe, it, expect, mock, beforeEach, afterEach } from 'bun:test';
import {
  parseEdgarAtomFeed,
  stripHtml,
  extractGuidancePhrases,
  splitTranscriptSections,
  extractQaHighlights,
  getEarningsTranscript,
} from './earnings-transcripts.js';

// ── helpers ───────────────────────────────────────────────────────────────────

function makeSampleAtomXml(entries: Array<{ url: string; date: string; title: string }>): string {
  const entryXml = entries
    .map(
      (e) => `
    <entry>
      <title type="text">${e.title}</title>
      <link rel="alternate" type="text/html" href="${e.url}" />
      <updated>${e.date}</updated>
    </entry>`,
    )
    .join('');
  return `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>EDGAR Full-Text Search</title>${entryXml}
</feed>`;
}

function makeSampleTranscriptHtml(content: string): string {
  return `<!DOCTYPE html>
<html><head><title>8-K Exhibit</title><style>body { font-family: Arial; }</style></head>
<body>
  <div class="transcript">${content}</div>
</body></html>`;
}

// ── mock fetch ────────────────────────────────────────────────────────────────

let originalFetch: typeof fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ── EDGAR Atom XML parsing ────────────────────────────────────────────────────

describe('parseEdgarAtomFeed', () => {
  it('extracts filing entries from valid Atom XML', () => {
    const xml = makeSampleAtomXml([
      {
        url: 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193',
        date: '2025-01-30T00:00:00-05:00',
        title: 'AAPL 8-K Filing 2025-01-30',
      },
      {
        url: 'https://www.sec.gov/Archives/edgar/data/320193/000032019325000001',
        date: '2024-11-01T00:00:00-05:00',
        title: 'AAPL 8-K Filing 2024-11-01',
      },
    ]);

    const entries = parseEdgarAtomFeed(xml);
    expect(entries).toHaveLength(2);
    expect(entries[0].url).toContain('CIK=0000320193');
    expect(entries[0].date).toBe('2025-01-30T00:00:00-05:00');
    expect(entries[0].title).toBe('AAPL 8-K Filing 2025-01-30');
    expect(entries[1].date).toBe('2024-11-01T00:00:00-05:00');
  });

  it('returns empty array when no entries in feed', () => {
    const xml = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Empty</title></feed>`;
    const entries = parseEdgarAtomFeed(xml);
    expect(entries).toHaveLength(0);
  });

  it('handles malformed XML gracefully', () => {
    const entries = parseEdgarAtomFeed('<not valid xml>><<');
    expect(Array.isArray(entries)).toBe(true);
    expect(entries).toHaveLength(0);
  });
});

// ── HTML stripping ────────────────────────────────────────────────────────────

describe('stripHtml', () => {
  it('removes HTML tags from transcript HTML', () => {
    const html = makeSampleTranscriptHtml(
      '<p>Good morning everyone.</p><p>We are pleased to report strong results.</p>',
    );
    const text = stripHtml(html);
    expect(text).not.toContain('<p>');
    expect(text).not.toContain('</div>');
    expect(text).toContain('Good morning everyone');
    expect(text).toContain('We are pleased to report strong results');
  });

  it('strips style blocks', () => {
    const html = '<style>body { color: red; }</style><p>Hello</p>';
    const text = stripHtml(html);
    expect(text).not.toContain('color: red');
    expect(text).toContain('Hello');
  });

  it('decodes HTML entities', () => {
    const html = '<p>Revenue &amp; profits &gt; expectations &lt; guidance</p>';
    const text = stripHtml(html);
    expect(text).toContain('Revenue & profits');
    expect(text).toContain('> expectations');
  });

  it('collapses multiple spaces', () => {
    const html = '<p>Hello   world</p>';
    const text = stripHtml(html);
    expect(text).not.toMatch(/\s{2,}/);
  });
});

// ── Guidance phrase extraction ────────────────────────────────────────────────

describe('extractGuidancePhrases', () => {
  it('extracts lines with forward-looking keywords', () => {
    const text = [
      'We expect revenue to grow by 10 percent next quarter.',
      'Our outlook for fiscal 2025 remains strong.',
      'The product launched last week.',
      'We forecast operating income of 5 billion dollars for the year.',
      'Guidance for Q3 is approximately 2 million units.',
      'It was a great quarter overall.',
    ].join('\n');

    const phrases = extractGuidancePhrases(text);
    expect(phrases.length).toBeGreaterThan(0);
    expect(phrases.some((p) => /expect/i.test(p))).toBe(true);
    expect(phrases.some((p) => /outlook/i.test(p))).toBe(true);
    expect(phrases.some((p) => /forecast/i.test(p))).toBe(true);
  });

  it('excludes short lines', () => {
    const text = 'expect\nWe expect strong growth in revenue for the next fiscal year ending December.';
    const phrases = extractGuidancePhrases(text);
    expect(phrases.every((p) => p.length > 20)).toBe(true);
  });

  it('returns at most 10 phrases', () => {
    const lines = Array.from(
      { length: 20 },
      (_, i) => `We expect revenue growth of ${i * 100} million dollars next year.`,
    );
    const phrases = extractGuidancePhrases(lines.join('\n'));
    expect(phrases.length).toBeLessThanOrEqual(10);
  });
});

// ── Transcript section splitting ──────────────────────────────────────────────

describe('splitTranscriptSections', () => {
  it('splits on "Questions and Answers" marker', () => {
    const text = [
      'Good morning. I am pleased to share our results.',
      'Revenue was 10 billion dollars.',
      'Questions and Answers',
      'Analyst: What is your outlook?',
      'CEO: We expect continued growth.',
    ].join('\n');

    const { remarks, qa } = splitTranscriptSections(text);
    expect(remarks).toContain('Revenue was 10 billion');
    expect(qa).toContain('Analyst:');
    expect(qa).toContain('What is your outlook');
  });

  it('splits on "Q&A Session" marker', () => {
    const text = 'Prepared remarks here. Q&A Session Analyst question follows.';
    const { remarks, qa } = splitTranscriptSections(text);
    expect(remarks).toContain('Prepared remarks here');
    expect(qa).toContain('Analyst question follows');
  });

  it('returns full text as remarks when no Q&A marker found', () => {
    const text = 'Management discussion only. Revenue grew strongly. No follow-up session recorded.';
    const { remarks, qa } = splitTranscriptSections(text);
    expect(remarks).toBe(text);
    expect(qa).toBe('');
  });
});

// ── Q&A highlights extraction ─────────────────────────────────────────────────

describe('extractQaHighlights', () => {
  it('extracts analyst questions and executive answers', () => {
    const qaText = [
      'Operator: The first question comes from John Smith.',
      'Analyst: What is your revenue guidance for next quarter?',
      'Answer: We expect 15 billion in Q2 driven by services growth.',
      'Analyst: How is the supply chain situation?',
      'Answer: We have resolved most issues and expect full capacity.',
      'Analyst: Any comments on international markets?',
      'Answer: International grew 20 percent year over year.',
    ].join('\n');

    const highlights = extractQaHighlights(qaText, 3);
    expect(Array.isArray(highlights)).toBe(true);
    expect(highlights.length).toBeGreaterThan(0);
  });

  it('returns empty array for empty Q&A text', () => {
    const highlights = extractQaHighlights('');
    expect(highlights).toHaveLength(0);
  });

  it('truncates questions and answers to 200 chars', () => {
    const longQuestion = 'A'.repeat(500);
    const qaText = `Analyst: ${longQuestion}\nAnswer: Response here.`;
    const highlights = extractQaHighlights(qaText, 1);
    for (const h of highlights) {
      expect(h.question.length).toBeLessThanOrEqual(200);
      expect(h.answer.length).toBeLessThanOrEqual(200);
    }
  });
});

// ── Tool: graceful error on EDGAR 404 ────────────────────────────────────────

describe('getEarningsTranscript tool', () => {
  it('returns error string (not throws) when EDGAR returns 404', async () => {
    globalThis.fetch = mock(async () =>
      new Response('Not Found', { status: 404 }),
    ) as unknown as typeof fetch;

    const result = await getEarningsTranscript.invoke({ ticker: 'UNKNOWNTICKER' });
    expect(typeof result).toBe('string');
    const parsed = JSON.parse(result);
    expect(parsed.data?.error).toBeDefined();
  });

  it('returns error string (not throws) when fetch throws', async () => {
    globalThis.fetch = mock(async () => {
      throw new Error('Network error');
    }) as unknown as typeof fetch;

    const result = await getEarningsTranscript.invoke({ ticker: 'AAPL' });
    expect(typeof result).toBe('string');
    const parsed = JSON.parse(result);
    expect(parsed.data?.error).toBeDefined();
  });

  it('returns a string result (never throws)', async () => {
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const urlStr = url.toString();
      if (urlStr.includes('output=atom')) {
        const xml = makeSampleAtomXml([
          {
            url: 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=320193&accession-number=0000320193-25-000001',
            date: '2025-02-01T00:00:00-05:00',
            title: 'AAPL 8-K 2025-02-01',
          },
        ]);
        return new Response(xml, { status: 200, headers: { 'Content-Type': 'application/atom+xml' } });
      }
      if (urlStr.includes('index.htm')) {
        const html = `<html><body><table><tr><td>ex-99 exhibit</td><td><a href="/Archives/edgar/data/320193/000032019325000001/exhibit99.htm">transcript</a></td></tr></table></body></html>`;
        return new Response(html, { status: 200, headers: { 'Content-Type': 'text/html' } });
      }
      if (urlStr.includes('.htm')) {
        const html = makeSampleTranscriptHtml(
          '<p>Good morning. Revenue was 10 billion.</p>' +
            '<p>We expect strong growth next quarter with guidance of 11 billion dollars.</p>' +
            '<p>Questions and Answers</p>' +
            '<p>Analyst: What is the outlook? Answer: We expect continued momentum.</p>',
        );
        return new Response(html, { status: 200, headers: { 'Content-Type': 'text/html' } });
      }
      return new Response('Not Found', { status: 404 });
    }) as unknown as typeof fetch;

    const result = await getEarningsTranscript.invoke({ ticker: 'AAPL' });
    expect(typeof result).toBe('string');
    // Should not throw and must parse as JSON
    expect(() => JSON.parse(result)).not.toThrow();
  });
});

// ── Tool: quarter selection ──────────────────────────────────────────────────

describe('getEarningsTranscript quarter selection', () => {
  const originalTavilyKey = process.env.TAVILY_API_KEY;
  let requests: string[];

  /** Serve an EDGAR 8-K feed whose filings are dated `dates`, newest first. */
  function mockEdgar(dates: string[]): void {
    requests = [];
    const dir = (date: string) =>
      `https://www.sec.gov/Archives/edgar/data/320193/${date.replace(/-/g, '')}`;
    globalThis.fetch = mock(async (url: string | URL | Request) => {
      const urlStr = url.toString();
      requests.push(urlStr);
      if (urlStr.includes('output=atom')) {
        const xml = makeSampleAtomXml(
          dates.map((date) => ({ url: `${dir(date)}/${date}-index.htm`, date: `${date}T16:30:00-04:00`, title: `8-K ${date}` })),
        );
        return new Response(xml, { status: 200 });
      }
      if (urlStr.endsWith('-index.htm')) {
        const exhibit = urlStr.replace(/[^/]+-index\.htm$/, 'ex99.htm').replace('https://www.sec.gov', '');
        return new Response(`<table><tr><td>EX-99.1</td><td><a href="${exhibit}">ex99.htm</a></td></tr></table>`, { status: 200 });
      }
      if (urlStr.endsWith('ex99.htm')) {
        return new Response(makeSampleTranscriptHtml('<p>We expect revenue to grow next quarter.</p>'), { status: 200 });
      }
      return new Response('Not Found', { status: 404 });
    }) as unknown as typeof fetch;
  }

  const invoke = async (quarter?: string) =>
    JSON.parse(await getEarningsTranscript.invoke({ ticker: 'AAPL', ...(quarter ? { quarter } : {}) })).data;

  beforeEach(() => {
    delete process.env.TAVILY_API_KEY;
  });
  afterEach(() => {
    if (originalTavilyKey === undefined) delete process.env.TAVILY_API_KEY;
    else process.env.TAVILY_API_KEY = originalTavilyKey;
  });

  it('asks EDGAR for about 40 8-K filings, not just the latest 5', async () => {
    mockEdgar(['2025-10-30']);
    await invoke();
    expect(requests[0]).toContain('count=40');
  });

  it('matches a quarter to the release filed in the following quarter', async () => {
    mockEdgar(['2025-10-30', '2025-07-31', '2025-05-01']);
    const data = await invoke('Q2 2025');
    expect(data.filingDate).toBe('2025-07-31');
    expect(data.inferredQuarter).toBe('Q2 2025');
  });

  it('matches Q4 to a release filed in January to March of the next year', async () => {
    mockEdgar(['2026-04-30', '2026-01-29', '2025-10-30']);
    const data = await invoke('Q4 2025');
    expect(data.filingDate).toBe('2026-01-29');
    expect(data.inferredQuarter).toBe('Q4 2025');
  });

  it('labels the latest filing with the quarter it reports, not the quarter it was filed in', async () => {
    mockEdgar(['2025-01-30']);
    expect((await invoke()).inferredQuarter).toBe('Q4 2024');
  });

  it('returns a clear error instead of the latest filing when the quarter is not found', async () => {
    mockEdgar(['2025-10-30', '2025-07-31']);
    const data = await invoke('Q2 2023');
    expect(data.error).toContain('Q2 2023');
    expect(requests.some((u) => u.endsWith('-index.htm'))).toBe(false);
  });

  it('returns a clear error for a quarter it cannot parse', async () => {
    mockEdgar(['2025-10-30']);
    const data = await invoke('second quarter');
    expect(data.error).toContain('second quarter');
    expect(requests.some((u) => u.endsWith('-index.htm'))).toBe(false);
  });
});
