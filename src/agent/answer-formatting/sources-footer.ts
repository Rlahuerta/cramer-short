/**
 * Build a compact Sources footer from a deduplicated list of URLs.
 * Only appended to answers when the model hasn't already cited inline links.
 * Limits to 10 URLs to keep the footer scannable.
 *
 * Social media post URLs (Reddit, X/Twitter, etc.) are excluded — these are
 * inputs to sentiment analysis, not authoritative research citations.
 */

/** Domains excluded from the Sources footer (social media / UGC). */
const EXCLUDED_SOURCE_DOMAINS = [
  'reddit.com',
  'x.com',
  'twitter.com',
  'threads.net',
  'bsky.app',
  'bluesky.app',
];

export function buildSourcesFooter(urls: string[]): string {
  const filtered = urls.filter(u => {
    try {
      const host = new URL(u).hostname.replace(/^www\./, '');
      return !EXCLUDED_SOURCE_DOMAINS.some(d => host === d || host.endsWith(`.${d}`));
    } catch {
      return false;
    }
  });
  const unique = [...new Set(filtered)].slice(0, 10);
  if (unique.length === 0) return '';
  const lines = unique.map((u, i) => `${i + 1}. ${u}`).join('\n');
  return `\n\n---\n**Sources**\n${lines}`;
}
