/**
 * P2b — metaforecast.org cross-platform fusion.
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §6
 *
 * metaforecast.org aggregates Metaculus, Manifold, Kalshi, GJOpen, etc.
 * When Polymarket's price diverges from the cross-platform consensus by
 * more than ~10pp, that disagreement is a useful uncertainty signal —
 * fold it into MarketInput.warnings and apply a quality discount.
 *
 * This module exposes pure helpers (parsing + matching + scoring) and
 * a thin HTTP fetch that callers can opt into. The HTTP layer is not
 * unit-tested; mocks should target the pure helpers.
 */

const METAFORECAST_API = 'https://metaforecast.org/api/v2/questions';
const CROSS_PLATFORM_DELTA_THRESHOLD = 0.10;

export interface MetaforecastEstimate {
  /** Question title as published on the source platform. */
  title: string;
  /** YES-side probability, clamped to [0, 1]. */
  probability: number;
  /** Source platform (metaculus, manifold, kalshi, ...). */
  platform: string;
  /** Quality score (0 = unknown, 4 = highest). */
  stars: number;
  /** Optional URL back to the source. */
  url?: string;
}

/**
 * Parse a metaforecast `/api/v2/questions` payload into a typed list.
 * Tolerates missing fields and malformed entries (skips them).
 */
export function parseMetaforecastResponse(raw: unknown): MetaforecastEstimate[] {
  if (!Array.isArray(raw)) return [];
  const out: MetaforecastEstimate[] = [];
  for (const r of raw) {
    if (!r || typeof r !== 'object') continue;
    const rec = r as Record<string, any>;
    const title = typeof rec.title === 'string' ? rec.title : null;
    const platform = typeof rec.platform === 'string' ? rec.platform : null;
    const opts = Array.isArray(rec.options) ? rec.options : null;
    if (!title || !platform || !opts || opts.length === 0) continue;
    // Pick the YES probability — first option whose name looks affirmative,
    // else the first numeric one.
    let prob: number | null = null;
    for (const opt of opts) {
      if (!opt || typeof opt !== 'object') continue;
      const name = String(opt.name ?? '').toLowerCase();
      const p = Number(opt.probability);
      if (!Number.isFinite(p)) continue;
      if (name === 'yes' || name.includes('yes')) {
        prob = p;
        break;
      }
      if (prob === null) prob = p;
    }
    if (prob === null) continue;
    const clamped = Math.max(0, Math.min(1, prob));
    const stars = Number(rec.qualityindicators?.stars ?? 0);
    out.push({
      title,
      probability: clamped,
      platform,
      stars: Number.isFinite(stars) ? stars : 0,
      ...(typeof rec.url === 'string' ? { url: rec.url } : {}),
    });
  }
  return out;
}

const _STOPWORDS = new Set([
  'will', 'the', 'a', 'an', 'be', 'by', 'in', 'on', 'of', 'to', 'and',
  'or', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'with', 'for',
  'at', 'as', 'this', 'that', 'these', 'those', 'do', 'does', 'did',
]);

function _tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9%\s]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 1 && !_STOPWORDS.has(t)),
  );
}

/**
 * Fuzzy keyword-overlap match. Returns the best candidate when the
 * Jaccard overlap is ≥ 0.20, else null. Ties broken by `stars`.
 */
export function findBestMetaforecastMatch(
  question: string,
  candidates: MetaforecastEstimate[],
): MetaforecastEstimate | null {
  if (!candidates.length) return null;
  const qTokens = _tokenize(question);
  if (qTokens.size === 0) return null;
  let best: MetaforecastEstimate | null = null;
  let bestScore = -1;
  for (const c of candidates) {
    const cTokens = _tokenize(c.title);
    if (cTokens.size === 0) continue;
    let inter = 0;
    for (const t of qTokens) if (cTokens.has(t)) inter += 1;
    const union = qTokens.size + cTokens.size - inter;
    const jaccard = union === 0 ? 0 : inter / union;
    if (jaccard < 0.20) continue;
    // Composite score: jaccard primary, stars tiebreaker.
    const score = jaccard + c.stars * 1e-3;
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

export function computeCrossPlatformDelta(polyProb: number, metaProb: number): number {
  return Math.abs(polyProb - metaProb);
}

export function shouldFlagCrossPlatform(delta: number): boolean {
  return delta > CROSS_PLATFORM_DELTA_THRESHOLD;
}

/**
 * Thin HTTP wrapper around metaforecast `/api/v2/questions`.
 * Network-dependent — call only from production code paths, not tests.
 */
export async function fetchMetaforecastQuestions(
  query: string,
  opts: { limit?: number; signal?: AbortSignal } = {},
): Promise<MetaforecastEstimate[]> {
  const limit = Math.max(1, Math.min(50, opts.limit ?? 10));
  const url = `${METAFORECAST_API}?query=${encodeURIComponent(query)}&limit=${limit}`;
  const res = await fetch(url, { signal: opts.signal });
  if (!res.ok) return [];
  const json = await res.json().catch(() => null);
  return parseMetaforecastResponse(json);
}
