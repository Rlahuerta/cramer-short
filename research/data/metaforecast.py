"""metaforecast.org cross-platform fusion (P2b mirror).

Mirrors ``src/tools/finance/metaforecast.ts``. Pure helpers; no live HTTP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

CROSS_PLATFORM_DELTA_THRESHOLD = 0.10

_STOPWORDS = frozenset({
    "will", "the", "a", "an", "be", "by", "in", "on", "of", "to", "and",
    "or", "is", "are", "was", "were", "has", "have", "had", "with", "for",
    "at", "as", "this", "that", "these", "those", "do", "does", "did",
})

_PUNCT_RE = re.compile(r"[^a-z0-9%\s]")


@dataclass(frozen=True)
class MetaforecastEstimate:
    title: str
    probability: float
    platform: str
    stars: int = 0
    url: str | None = None


def parse_metaforecast_response(raw: Any) -> list[MetaforecastEstimate]:
    """Parse a metaforecast `/api/v2/questions` payload."""
    if not isinstance(raw, list):
        return []
    out: list[MetaforecastEstimate] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        title = r.get("title")
        platform = r.get("platform")
        opts = r.get("options")
        if not isinstance(title, str) or not isinstance(platform, str):
            continue
        if not isinstance(opts, list) or not opts:
            continue
        prob: float | None = None
        for opt in opts:
            if not isinstance(opt, dict):
                continue
            try:
                p = float(opt.get("probability"))
            except (TypeError, ValueError):
                continue
            name = str(opt.get("name", "")).lower()
            if name == "yes" or "yes" in name:
                prob = p
                break
            if prob is None:
                prob = p
        if prob is None:
            continue
        clamped = max(0.0, min(1.0, prob))
        stars_raw = (r.get("qualityindicators") or {}).get("stars", 0)
        try:
            stars = int(stars_raw)
        except (TypeError, ValueError):
            stars = 0
        url = r.get("url") if isinstance(r.get("url"), str) else None
        out.append(MetaforecastEstimate(
            title=title,
            probability=clamped,
            platform=platform,
            stars=stars,
            url=url,
        ))
    return out


def _tokenize(text: str) -> set[str]:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return {t for t in cleaned.split() if len(t) > 1 and t not in _STOPWORDS}


def find_best_metaforecast_match(
    question: str,
    candidates: Iterable[MetaforecastEstimate],
) -> MetaforecastEstimate | None:
    """Fuzzy keyword-overlap match (Jaccard ≥ 0.20)."""
    cand_list = list(candidates)
    if not cand_list:
        return None
    q_tokens = _tokenize(question)
    if not q_tokens:
        return None
    best: MetaforecastEstimate | None = None
    best_score = -1.0
    for c in cand_list:
        c_tokens = _tokenize(c.title)
        if not c_tokens:
            continue
        inter = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        jaccard = inter / union if union else 0.0
        if jaccard < 0.20:
            continue
        score = jaccard + c.stars * 1e-3
        if score > best_score:
            best_score = score
            best = c
    return best


def compute_cross_platform_delta(poly_prob: float, meta_prob: float) -> float:
    return abs(poly_prob - meta_prob)


def should_flag_cross_platform(delta: float) -> bool:
    return delta > CROSS_PLATFORM_DELTA_THRESHOLD


__all__ = [
    "MetaforecastEstimate",
    "parse_metaforecast_response",
    "find_best_metaforecast_match",
    "compute_cross_platform_delta",
    "should_flag_cross_platform",
    "CROSS_PLATFORM_DELTA_THRESHOLD",
]
