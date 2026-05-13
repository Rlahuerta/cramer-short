import {
  getForecastLabProfile,
  listForecastLabProfiles,
  listForecastLabStructuredMutations,
  type ForecastLabProfileId,
} from './profiles.js';
import { routeForecastLabQuery } from './router.js';
import type { InMemoryChatHistory } from '../../utils/in-memory-chat-history.js';

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

const FORECAST_LAB_APPROVAL_PATTERNS = [
  /\bapprove(?:d|s|ing)?\b/i,
  /\bgo ahead\b/i,
  /\bproceed\b/i,
  /\bpromote\b/i,
  /^\s*yes\b/i,
  /^\s*sure\b/i,
  /^\s*do it\b/i,
] as const;

const FORECAST_LAB_RESET_PATTERNS = [
  /\breset\b/i,
  /\brestore\b/i,
  /\broll\s+back\b/i,
] as const;

const SAFE_FORECAST_LAB_RUN_ID = /[A-Za-z0-9][A-Za-z0-9_.-]*/;

export interface ForecastLabPromotionApprovalHint {
  readonly profileId?: string;
  readonly sourceRunId?: string;
}

export interface ForecastLabResetHint {
  readonly profileId?: string;
  readonly mode: 'defaults' | 'last-known-good';
}

export interface ForecastLabComparisonHint {
  readonly profileId?: string;
  readonly mutationId?: string;
}

export interface ForecastLabResultsHint {
  readonly profileId?: string;
}

export interface ForecastLabMutatorListHint {
  readonly profileId?: string;
}

export interface ForecastLabKeepCurrentBestHint {
  readonly profileId?: string;
}

export interface ForecastLabCatalogExtensionHint {
  readonly profileId?: string;
}


function extractForecastLabProfileId(text: string): string | undefined {
  const lower = text.toLowerCase();
  return listForecastLabProfiles().find((profile) => lower.includes(profile.id.toLowerCase()))?.id;
}

function extractForecastLabSourceRunId(text: string): string | undefined {
  const match = text.match(new RegExp(`(?:source\\s+run|run)\\s+(${SAFE_FORECAST_LAB_RUN_ID.source})`, 'i'));
  const candidate = match?.[1];
  if (!candidate) {
    return undefined;
  }
  if (!candidate.includes('-') && !candidate.includes('.')) {
    return undefined;
  }
  return candidate;
}

function isForecastLabApprovalIntent(query: string): boolean {
  if (
    /\bhow\s+(?:do\s+i\s+)?(?:approve|promote)\b/i.test(query)
    || /\bhow\s+to\s+(?:approve|promote)\b/i.test(query)
    || /\bwhat(?:'s| is)\b[\s\S]{0,40}\b(?:approve|promote)\b/i.test(query)
    || /\bwhich\s+command\b[\s\S]{0,40}\b(?:approve|promote)\b/i.test(query)
  ) {
    return false;
  }
  return FORECAST_LAB_APPROVAL_PATTERNS.some((pattern) => pattern.test(query));
}

function isForecastLabResetIntent(query: string): boolean {
  return FORECAST_LAB_RESET_PATTERNS.some((pattern) => pattern.test(query));
}

function isForecastLabComparisonIntent(query: string): boolean {
  const hasBestReference = /\bcurrent best\b|\bbest kept\b|\bbest lineage\b|\blatest kept\b/i.test(query);
  const hasBaselineReference = /\bshipped default baseline\b|\bshipped baseline\b|\bdefault baseline\b/i.test(query);
  const hasComparisonCue = /\bbetter than\b|\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|\bstack up\b/i.test(query);
  return (hasBestReference && hasBaselineReference) || (hasBaselineReference && hasComparisonCue);
}

const FORECAST_LAB_NAMED_MUTATION_ID = /(?<![A-Za-z0-9_.-])((?:gold-)?markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i;
const FORECAST_LAB_REQUESTED_MUTATION_ID = /\brequested\s+mutator\s+id:\s*((?:gold-)?markov-[A-Za-z0-9][A-Za-z0-9_.-]*)/gi;
const FORECAST_LAB_REQUESTED_MUTATION_ID_CUE = /\brequested mutator id:\s*(?:gold-)?markov-/i;

function extractForecastLabNamedMutationId(query: string): string | undefined {
  const candidate = query.match(FORECAST_LAB_NAMED_MUTATION_ID)?.[1];
  return candidate?.replace(/[.,!?;:]+$/, '');
}

function extractForecastLabComparisonMutationId(query: string): string | undefined {
  return extractForecastLabNamedMutationId(query);
}

function extractForecastLabRequestedMutationId(text: string): string | undefined {
  const matches = [...text.matchAll(FORECAST_LAB_REQUESTED_MUTATION_ID)];
  const candidate = matches.at(-1)?.[1];
  return candidate?.replace(/[.,!?;:]+$/, '');
}

function isKnownForecastLabStructuredMutationId(mutationId: string): boolean {
  const candidate = mutationId.toLowerCase();
  return listForecastLabProfiles().some((profile) =>
    profile.mutation.mode === 'structured'
    && listForecastLabStructuredMutations(profile.id).some((entry) => entry.id.toLowerCase() === candidate),
  );
}

function isForecastLabMutatorVsActiveIntent(query: string, historyText = ''): boolean {
  const mutationId = extractForecastLabComparisonMutationId(query)
    ?? extractForecastLabRequestedMutationId(historyText);
  if (!mutationId) {
    return false;
  }

  const hasComparisonCue = /\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b/i.test(query);
  const hasMetricCue = /\baccurac\w*\b|\baccurace\b|\bnumbers\b|\bmetrics?\b/i.test(query);
  const hasActiveCue = /\bactive one\b|\bactive baseline\b|\bactive run\b|\bactive mutation\b|\blive one\b|\blive baseline\b|\blive run\b|\bcurrently live\b/i.test(query);
  const hasCreatedCandidateCue = /\bnew\s+mutat(?:e|or)\b|\bnew\s+one\b|\bthat\s+i\s+created\b|\bi\s+created\b|\bnot\s+promoted\b|\bunpromoted\b/i.test(query);
  const contextText = `${query}\n${historyText}`;
  const hasForecastLabContext =
    mutationId !== undefined
    || /\bforecast-lab\b/i.test(contextText)
    || /\bbtc-markov-ultra-short-horizon\b/i.test(contextText)
    || /\bactive baseline\b/i.test(contextText)
    || /\bpromoted parameters\b/i.test(contextText)
    || /src\/tools\/finance\/markov-distribution\.ts/i.test(contextText);

  return hasForecastLabContext && hasActiveCue && (hasComparisonCue || hasMetricCue || hasCreatedCandidateCue);
}

function isForecastLabResultsIntent(query: string): boolean {
  const hasResultsCue = /\bresult(?:s)?\b|\boutcome(?:s)?\b|\bstatus\b|\bsummary\b|\brecap\b|\bwhat happened\b/i.test(query);
  const hasRequestCue = /\bprovide\b|\bshow\b|\bgive\b|\bsummarize\b|\breport\b|\brecap\b|\bwhat\b/i.test(query);
  const hasWorkflowCue = /\boptimi[sz]e\b|\bimprov(?:e|ement)\b|\bworkflow\b|\bforecast-lab\b/i.test(query);
  return hasResultsCue && hasRequestCue && hasWorkflowCue;
}

function isForecastLabMutatorListIntent(query: string): boolean {
  const hasListCue = /\blist\b|\bshow\b|\bgive\b|\bwhat\b|\bwhich\b/i.test(query);
  const hasMutatorCue = /\bmutat(?:e|or|ors|ion|ions)\b/i.test(query);
  const hasCatalogCue = /\bids?\b|\bavail\w*\b|\bshipped\b|\bcatalog\b|\bcurrent\b/i.test(query);
  return hasMutatorCue && hasListCue && hasCatalogCue;
}

function isForecastLabKeepCurrentBestIntent(query: string): boolean {
  return /\bkeep\b[\s\S]{0,30}\bcurrent best candidate\b/i.test(query)
    || /\bkeep\b[\s\S]{0,20}\bbest candidate\b/i.test(query)
    || /\bkeep\b[\s\S]{0,20}\bbest run\b/i.test(query);
}

function hasForecastLabCatalogExtensionContext(text: string): boolean {
  return /\bforecast-lab\b/i.test(text)
    || /\bbtc-markov-ultra-short-horizon\b/i.test(text)
    || /\bbtc\s+1d\/2d\/3d\b/i.test(text)
    || /\bmarkov forecast workflow\b/i.test(text)
    || /\bultra-short-horizon\b/i.test(text)
    || /src\/tools\/finance\/markov-distribution\.ts/i.test(text);
}

function hasForecastLabCatalogImplementationCue(text: string): boolean {
  return /\bkeep it bounded\b/i.test(text)
    || /\badd\b[\s\S]{0,30}\bcatalog\b/i.test(text)
    || /\bvalidate\b[\s\S]{0,40}\b(?:walk-forward|gate)\b/i.test(text)
    || /\bsuggested starting values\b/i.test(text)
    || /\bsearch-replace\b/i.test(text)
    || /\bsoft-regime weighting\b/i.test(text);
}

function isForecastLabCatalogExtensionIntent(query: string, historyText = ''): boolean {
  const requestedMutationId = extractForecastLabNamedMutationId(query)
    ?? extractForecastLabRequestedMutationId(historyText);
  const hasImplementationExecutionCue =
    requestedMutationId !== undefined
    && !isKnownForecastLabStructuredMutationId(requestedMutationId)
    && /\bimplement\b|\bregister\b/i.test(query)
    && /\brun\b|\bre-?run\b|\bexecute\b/i.test(query);
  const hasMutatorCue = /\bmutator\b/i.test(query);
  const hasCatalogCue = /\bcatalog\b|\bshipped\b/i.test(query);
  const hasExtensionCue = /\bdesign\b|\badd\b|\bcreate\b|\bnew\b|\bextend\b|\boutside\b/i.test(query);
  const hasLineageCue = /\blineage\b|\bre-?run\b/i.test(query);
  const contextText = `${query}\n${historyText}`;
  return (
    hasImplementationExecutionCue
    && (
      /\bcatalog-extension plan\b/i.test(historyText)
      || FORECAST_LAB_REQUESTED_MUTATION_ID_CUE.test(historyText)
      || hasForecastLabCatalogExtensionContext(contextText)
    )
  ) || (
    hasMutatorCue
    && hasCatalogCue
    && hasExtensionCue
    && (
      hasLineageCue
      || hasForecastLabCatalogExtensionContext(contextText)
      || hasForecastLabCatalogImplementationCue(query)
    )
  );
}

export function detectForecastLabPromotionApproval(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabPromotionApprovalHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const assistantHistory = recentTurns
    .filter((entry) => entry.role === 'assistant')
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${assistantHistory}`;
  const hasForecastLabContext =
    /\bforecast-lab\b/i.test(contextText)
    || /\bpromotion-ready\b/i.test(contextText)
    || /\bapproval required\b/i.test(contextText);

  if (!hasForecastLabContext || !isForecastLabApprovalIntent(query)) {
    return null;
  }

  const profileId = extractForecastLabProfileId(contextText);
  const sourceRunId = extractForecastLabSourceRunId(query) ?? extractForecastLabSourceRunId(assistantHistory);

  if (!profileId && !sourceRunId && !/\bforecast-lab\b/i.test(query)) {
    return null;
  }

  return {
    ...(profileId ? { profileId } : {}),
    ...(sourceRunId ? { sourceRunId } : {}),
  };
}

export function detectForecastLabResetRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabResetHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const assistantHistory = recentTurns
    .filter((entry) => entry.role === 'assistant')
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${assistantHistory}`;
  const hasForecastLabContext =
    /\bforecast-lab\b/i.test(contextText)
    || /\bactive baseline\b/i.test(contextText)
    || /\bpromoted parameters\b/i.test(contextText);

  if (!hasForecastLabContext || !isForecastLabResetIntent(query)) {
    return null;
  }

  const mode = /\bdefault(?:s)?\b|\bshipped defaults\b/i.test(query)
    ? 'defaults'
    : /\blast known good\b|\bprevious(?:ly)? activated\b|\bprevious baseline\b/i.test(query)
      ? 'last-known-good'
      : null;
  if (!mode) {
    return null;
  }

  const profileId = extractForecastLabProfileId(contextText);
  if (!profileId) {
    return null;
  }

  return { profileId, mode };
}

export function detectForecastLabComparisonRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabComparisonHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${historyText}`;
  const mutationId = extractForecastLabComparisonMutationId(query)
    ?? extractForecastLabRequestedMutationId(historyText);
  const routedProfileId = routeForecastLabQuery(query).preferredProfileId ?? undefined;
  const profileId = extractForecastLabProfileId(contextText) ?? routedProfileId;

  if (isForecastLabComparisonIntent(query)) {
    return profileId ? { profileId } : {};
  }

  if (!isForecastLabMutatorVsActiveIntent(query, historyText)) {
    return null;
  }

  return {
    ...(profileId ? { profileId } : {}),
    ...(mutationId ? { mutationId } : {}),
  };
}

export function detectForecastLabResultsRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabResultsHint | null {
  if (!isForecastLabResultsIntent(query)) {
    return null;
  }

  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${historyText}`;
  const routedProfileId = routeForecastLabQuery(query).preferredProfileId ?? undefined;
  const profileId = extractForecastLabProfileId(contextText) ?? routedProfileId;

  return profileId ? { profileId } : {};
}

export function detectForecastLabMutatorListRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabMutatorListHint | null {
  if (!isForecastLabMutatorListIntent(query)) {
    return null;
  }

  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${historyText}`;
  const routedProfileId = routeForecastLabQuery(query).preferredProfileId ?? undefined;
  const profileId = extractForecastLabProfileId(contextText) ?? routedProfileId;

  return profileId ? { profileId } : {};
}

export function detectForecastLabKeepCurrentBestRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabKeepCurrentBestHint | null {
  if (!isForecastLabKeepCurrentBestIntent(query)) {
    return null;
  }

  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const explicitProfileId = extractForecastLabProfileId(query);
  const contextText = explicitProfileId ? `${query}\n${historyText}` : historyText;
  const hasForecastLabContext =
    /\bforecast-lab\b/i.test(query)
    || explicitProfileId !== undefined
    || /\bforecast-lab\b/i.test(historyText)
    || /\bapproval required\b/i.test(historyText)
    || /\bcurrent best\b/i.test(historyText)
    || /\bkept lineage\b/i.test(historyText)
    || /\bbtc-markov-ultra-short-horizon\b/i.test(historyText);
  if (!hasForecastLabContext) {
    return null;
  }

  const profileId = explicitProfileId ?? extractForecastLabProfileId(historyText);
  return profileId ? { profileId } : {};
}

export function detectForecastLabCatalogExtensionRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabCatalogExtensionHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  if (!isForecastLabCatalogExtensionIntent(query, historyText)) {
    return null;
  }
  const profileId = extractForecastLabProfileId(`${query}\n${historyText}`);

  return profileId ? { profileId } : {};
}
