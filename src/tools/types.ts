import { wrapWebContent } from './fetch/external-content.js';

export interface ToolResult {
  data: unknown;
  sourceUrls?: string[];
}

export function formatToolResult(data: unknown, sourceUrls?: string[]): string {
  const result: ToolResult = { data };
  if (sourceUrls?.length) {
    result.sourceUrls = sourceUrls;
  }
  return JSON.stringify(result);
}

/**
 * Parse search results from a search provider response.
 * Handles both string and object responses, extracting URLs from results.
 * Supports multiple response shapes from different providers.
 */
export function parseSearchResults(result: unknown): { parsed: unknown; urls: string[] } {
  // Safely parse JSON strings
  let parsed: unknown;
  if (typeof result === 'string') {
    try {
      parsed = JSON.parse(result);
    } catch {
      // If parsing fails, treat the string as the result itself
      parsed = result;
    }
  } else {
    parsed = result;
  }

  // Extract URLs from multiple possible response shapes
  let urls: string[] = [];

  // Shape 1: { results: [{ url: string }] } (Exa format)
  if (parsed && typeof parsed === 'object' && 'results' in parsed) {
    const results = (parsed as { results?: unknown[] }).results;
    if (Array.isArray(results)) {
      urls = results
        .map((r) => (r && typeof r === 'object' && 'url' in r ? (r as { url?: string }).url : null))
        .filter((url): url is string => Boolean(url));
    }
  }
  // Shape 2: [{ url: string }] (direct array, Tavily format)
  else if (Array.isArray(parsed)) {
    urls = parsed
      .map((r) => (r && typeof r === 'object' && 'url' in r ? (r as { url?: string }).url : null))
      .filter((url): url is string => Boolean(url));
  }

  return { parsed, urls };
}

const RESULT_TEXT_KEYS = ['title', 'text', 'content', 'raw_content', 'snippet', 'summary'];

function wrapSearchResult(result: unknown): unknown {
  if (!result || typeof result !== 'object') return result;
  const out: Record<string, unknown> = { ...(result as Record<string, unknown>) };
  for (const key of RESULT_TEXT_KEYS) {
    if (typeof out[key] === 'string') out[key] = wrapWebContent(out[key]);
  }
  if (Array.isArray(out.highlights)) {
    out.highlights = out.highlights.map((h) => (typeof h === 'string' ? wrapWebContent(h) : h));
  }
  return out;
}

/**
 * Wrap the third-party text in web-search results (title/text/content/snippet/highlights and a
 * synthesized `answer`) in EXTERNAL_UNTRUSTED_CONTENT markers, as web_fetch does. Urls, ids and
 * scores stay untouched.
 */
export function wrapSearchResults(data: unknown): unknown {
  if (typeof data === 'string') return wrapWebContent(data);
  if (Array.isArray(data)) return data.map(wrapSearchResult);
  if (!data || typeof data !== 'object') return data;
  const out: Record<string, unknown> = { ...(data as Record<string, unknown>) };
  if (typeof out.answer === 'string') out.answer = wrapWebContent(out.answer);
  if (Array.isArray(out.results)) out.results = out.results.map(wrapSearchResult);
  return out;
}
