import { getForecastLabProfile, type ForecastLabProfileId } from '../experiments/forecast-lab/profiles.js';
import { routeForecastLabQuery } from '../experiments/forecast-lab/router.js';

export interface ForecastLabRoutingHint {
  readonly recommendedProfileId: ForecastLabProfileId | null;
  readonly whyMatched: string;
  readonly mutationAllowed: boolean;
  readonly shouldInvokeSkill: boolean;
}

export function getForecastLabRoutingHint(query: string): ForecastLabRoutingHint | null {
  const route = routeForecastLabQuery(query);
  if (route.intent !== 'improvement') {
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
