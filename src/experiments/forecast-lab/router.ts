import {
  listForecastLabProfiles,
  type ForecastLabProfile,
  type ForecastLabProfileId,
  type ForecastLabProfileRoutingKeywordGroup,
} from './profiles.js';

export type ForecastLabRouteIntent = 'none' | 'improvement';

export interface ForecastLabQueryRoute {
  readonly intent: ForecastLabRouteIntent;
  readonly preferredProfileId: ForecastLabProfileId | null;
  readonly reasons: readonly string[];
}

const IMPROVEMENT_TERMS = [
  'improve',
  'improvement',
  'optimize',
  'optimization',
  'tune',
  'tuning',
  'refine',
  'refinement',
  'calibrate',
  'calibration',
  'fix',
  'repair',
] as const;

const MINIMUM_PROFILE_SCORE = 2;

function normalizeForContainment(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/\s+/g, ' ').trim();
}

function includesTerm(rawQuery: string, normalizedQuery: string, term: string): boolean {
  const rawTerm = term.trim().toLowerCase();
  if (rawTerm.length === 0) {
    return false;
  }

  if (rawQuery.includes(rawTerm)) {
    return true;
  }

  const normalizedTerm = normalizeForContainment(rawTerm);
  return normalizedTerm.length > 0 && normalizedQuery.includes(normalizedTerm);
}

function summarizeMatchedGroup(
  rawQuery: string,
  normalizedQuery: string,
  group: ForecastLabProfileRoutingKeywordGroup,
): { readonly score: number; readonly reason: string } | null {
  const matchedTerms = group.terms.filter((term) => includesTerm(rawQuery, normalizedQuery, term));
  if (matchedTerms.length === 0) {
    return null;
  }

  return {
    score: group.weight ?? 1,
    reason: `${group.label} (${matchedTerms.join(', ')})`,
  };
}

function scoreProfile(
  rawQuery: string,
  normalizedQuery: string,
  profile: ForecastLabProfile,
): { readonly profileId: ForecastLabProfileId; readonly score: number; readonly reasons: readonly string[] } {
  const matches = profile.routing.keywordGroups
    .map((group) => summarizeMatchedGroup(rawQuery, normalizedQuery, group))
    .filter((match): match is NonNullable<typeof match> => match !== null);

  return {
    profileId: profile.id,
    score: matches.reduce((total, match) => total + match.score, 0),
    reasons: matches.map((match) => match.reason),
  };
}

export function routeForecastLabQuery(query: string): ForecastLabQueryRoute {
  const rawQuery = query.trim().toLowerCase();
  const normalizedQuery = normalizeForContainment(query);
  const matchedImprovementTerms = IMPROVEMENT_TERMS.filter((term) =>
    includesTerm(rawQuery, normalizedQuery, term),
  );

  if (matchedImprovementTerms.length === 0) {
    return {
      intent: 'none',
      preferredProfileId: null,
      reasons: ['No forecast-lab improvement cues matched; treating the query as ordinary forecast usage or analysis.'],
    };
  }

  const scoredProfiles = listForecastLabProfiles()
    .map((profile) => scoreProfile(rawQuery, normalizedQuery, profile))
    .sort((left, right) => right.score - left.score);
  const bestProfile = scoredProfiles[0];
  const reasons = [`Matched improvement cues: ${matchedImprovementTerms.join(', ')}.`];

  if (!bestProfile || bestProfile.score < MINIMUM_PROFILE_SCORE) {
    return {
      intent: 'improvement',
      preferredProfileId: null,
      reasons: [
        ...reasons,
        'No forecast-lab profile received enough routing signal to choose deterministically.',
      ],
    };
  }

  return {
    intent: 'improvement',
    preferredProfileId: bestProfile.profileId,
    reasons: [
      ...reasons,
      `Preferred ${bestProfile.profileId} because it matched ${bestProfile.reasons.join(', ')}.`,
    ],
  };
}
