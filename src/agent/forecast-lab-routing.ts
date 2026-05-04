import { getForecastLabProfile, type ForecastLabProfileId } from '../experiments/forecast-lab/profiles.js';
import { routeForecastLabQuery } from '../experiments/forecast-lab/router.js';

export interface ForecastLabRoutingHint {
  readonly recommendedProfileId: ForecastLabProfileId | null;
  readonly whyMatched: string;
  readonly mutationAllowed: boolean;
  readonly shouldInvokeSkill: boolean;
  readonly requestedMutatorId?: string;
}

export interface ForecastLabRoutingOptions {
  readonly enableAutoRoute?: boolean;
  readonly enableSkillHint?: boolean;
}

const FORECAST_LAB_MUTATOR_ID = /(?:using\s+mutator|mutator)\s+([A-Za-z0-9][A-Za-z0-9_.-]*)/i;

export function extractForecastLabMutatorId(query: string): string | undefined {
  const match = query.match(FORECAST_LAB_MUTATOR_ID);
  const candidate = match?.[1];
  return candidate?.replace(/[.,!?;:]+$/, '');
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

  const requestedMutatorId = extractForecastLabMutatorId(query);

  return {
    recommendedProfileId: route.preferredProfileId,
    whyMatched: route.reasons.join(' '),
    mutationAllowed: profile !== null && profile.mutation.mode !== 'dry-run',
    shouldInvokeSkill: true,
    ...(requestedMutatorId ? { requestedMutatorId } : {}),
  };
}
