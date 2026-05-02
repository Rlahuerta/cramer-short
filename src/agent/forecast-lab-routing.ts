import { getForecastLabProfile, type ForecastLabProfileId } from '../experiments/forecast-lab/profiles.js';
import { routeForecastLabQuery } from '../experiments/forecast-lab/router.js';

export interface ForecastLabRoutingHint {
  readonly recommendedProfileId: ForecastLabProfileId | null;
  readonly whyMatched: string;
  readonly mutationAllowed: boolean;
  readonly shouldInvokeSkill: boolean;
}

export interface ForecastLabRoutingOptions {
  readonly enableAutoRoute?: boolean;
  readonly enableSkillHint?: boolean;
}

export function getForecastLabRoutingHint(
  query: string,
  options: ForecastLabRoutingOptions = {},
): ForecastLabRoutingHint | null {
  if (options.enableAutoRoute === false) {
    return null;
  }

  const route = routeForecastLabQuery(query);
  if (route.intent !== 'improvement') {
    return null;
  }

  if (options.enableSkillHint === false) {
    return null;
  }

  const profile = route.preferredProfileId ? getForecastLabProfile(route.preferredProfileId) : null;

  return {
    recommendedProfileId: route.preferredProfileId,
    whyMatched: route.reasons.join(' '),
    mutationAllowed: profile !== null && profile.mutation.mode !== 'dry-run',
    shouldInvokeSkill: true,
  };
}
