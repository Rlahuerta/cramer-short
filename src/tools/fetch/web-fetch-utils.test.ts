import { describe, it, expect } from 'bun:test';
import {
  htmlToMarkdown,
  markdownToText,
  truncateText,
  extractReadableContent,
  type ExtractMode,
} from './web-fetch-utils.js';

// ---------------------------------------------------------------------------
// htmlToMarkdown
// ---------------------------------------------------------------------------

describe('htmlToMarkdown', () => {
  it('strips script and style tags', () => {
    const html = '<p>Content</p><script>alert("xss")</script><style>body{color:red}</style>';
    const { text } = htmlToMarkdown(html);
    expect(text).not.toContain('alert');
    expect(text).not.toContain('color:red');
    expect(text).toContain('Content');
  });

  it('extracts title from title tag', () => {
    const html = '<title>My Page</title><p>Body</p>';
    const { title } = htmlToMarkdown(html);
    expect(title).toBe('My Page');
  });

  it('converts headings to markdown', () => {
    const html = '<h1>Big Heading</h1><h2>Sub Heading</h2>';
    const { text } = htmlToMarkdown(html);
    expect(text).toContain('# Big Heading');
    expect(text).toContain('## Sub Heading');
  });

  it('converts anchor tags to markdown links', () => {
    const html = '<a href="https://example.com">Example</a>';
    const { text } = htmlToMarkdown(html);
    expect(text).toContain('[Example](https://example.com)');
  });

  it('converts list items to markdown bullets', () => {
    const html = '<ul><li>Item 1</li><li>Item 2</li></ul>';
    const { text } = htmlToMarkdown(html);
    expect(text).toContain('- Item 1');
    expect(text).toContain('- Item 2');
  });

  it('decodes HTML entities', () => {
    const html = '<p>AT&amp;T earns &quot;profits&quot; &gt; $1B</p>';
    const { text } = htmlToMarkdown(html);
    expect(text).toContain('AT&T');
    expect(text).toContain('"profits"');
    expect(text).toContain('> $1B');
  });

  it('returns undefined title when no title tag present', () => {
    const { title } = htmlToMarkdown('<p>No title here</p>');
    expect(title).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// markdownToText
// ---------------------------------------------------------------------------

describe('markdownToText', () => {
  it('removes markdown headings', () => {
    const md = '# Heading\n## Sub\nBody text';
    const result = markdownToText(md);
    expect(result).not.toContain('#');
    expect(result).toContain('Heading');
    expect(result).toContain('Body text');
  });

  it('removes inline code backticks', () => {
    const result = markdownToText('Use `console.log` for debugging.');
    expect(result).toContain('console.log');
    expect(result).not.toContain('`');
  });

  it('removes fenced code blocks', () => {
    const md = 'Before\n```js\nconst x = 1;\n```\nAfter';
    const result = markdownToText(md);
    expect(result).toContain('Before');
    expect(result).toContain('After');
    expect(result).not.toContain('```');
  });

  it('strips link syntax, keeping the label', () => {
    const result = markdownToText('[Click here](https://example.com)');
    expect(result).toContain('Click here');
    expect(result).not.toContain('https://example.com');
  });

  it('removes image syntax entirely', () => {
    const result = markdownToText('![Alt text](image.png) followed by text');
    expect(result).not.toContain('![');
    expect(result).toContain('followed by text');
  });

  it('removes unordered list bullets', () => {
    const result = markdownToText('- Item one\n- Item two');
    expect(result).not.toMatch(/^-\s/m);
    expect(result).toContain('Item one');
  });

  it('removes ordered list numbers', () => {
    const result = markdownToText('1. First\n2. Second');
    expect(result).not.toMatch(/^\d+\./m);
    expect(result).toContain('First');
  });
});

// ---------------------------------------------------------------------------
// truncateText
// ---------------------------------------------------------------------------

describe('truncateText', () => {
  it('returns text unchanged when within limit', () => {
    const { text, truncated } = truncateText('hello', 10);
    expect(text).toBe('hello');
    expect(truncated).toBe(false);
  });

  it('truncates and sets truncated=true when over limit', () => {
    const { text, truncated } = truncateText('hello world', 5);
    expect(text).toBe('hello');
    expect(truncated).toBe(true);
  });

  it('does not truncate when text length equals maxChars', () => {
    const { truncated } = truncateText('exact', 5);
    expect(truncated).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// extractReadableContent — fallback path (no DOM)
// ---------------------------------------------------------------------------

describe('extractReadableContent', () => {
  it('returns text from simple HTML via htmlToMarkdown fallback', async () => {
    const html = '<p>This is the content</p>';
    const result = await extractReadableContent({ html, url: 'https://example.com', extractMode: 'markdown' });
    expect(result).not.toBeNull();
    expect(result!.text).toContain('content');
  });

  it('returns plain text when extractMode is "text"', async () => {
    const html = '<h1>Title</h1><p>Body paragraph</p>';
    const result = await extractReadableContent({ html, url: 'https://example.com', extractMode: 'text' });
    expect(result).not.toBeNull();
    expect(result!.text).toContain('Title');
    expect(result!.text).toContain('Body');
  });

  it('returns null-safe result for empty HTML', async () => {
    const result = await extractReadableContent({ html: '', url: 'https://example.com', extractMode: 'markdown' });
    // Can be null or have empty text - either is acceptable
    if (result !== null) {
      expect(typeof result.text).toBe('string');
    }
  });

  it('handles HTML with title in extractMode markdown', async () => {
    const html = '<title>Page Title</title><p>Content here</p>';
    const result = await extractReadableContent({ html, url: 'https://example.com', extractMode: 'markdown' });
    expect(result).not.toBeNull();
  });
});
