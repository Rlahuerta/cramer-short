import type {
  ForecastLabCandidateWorkspaceMetadata,
  ForecastLabMutationLineage,
  ForecastLabMutationMode,
  ForecastLabProfileMutationConfig,
  ForecastLabMutationSpecSummary,
} from './mutation.js';
import type { ForecastLabMarkovParameterMutationReplayPayload } from './mutators/markov-parameters.js';

export type ForecastLabDecision = 'keep' | 'drop';
export type ForecastLabRoutingInvocationSource = 'auto-routed' | 'manual-request';

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface ForecastLabRoutingContext {
  originatingQuery: string;
  selectedProfileId: string;
  routerReason: string;
  invocationSource: ForecastLabRoutingInvocationSource;
}

export interface ForecastLabProfileRoutingStats {
  totalRuns: number;
  keptRuns: number;
  droppedRuns: number;
  autoRoutedRuns: number;
  manualRequestedRuns: number;
  lastDecision: ForecastLabDecision;
  lastRunAt: string;
}

export interface ForecastLabRoutingStats {
  version: 1;
  profiles: Record<string, ForecastLabProfileRoutingStats>;
}

export interface ForecastLabLedgerEntry {
  runId: string;
  startedAt: string;
  profileId: string;
  targetSubsystem: string;
  candidateBranch: string;
  allowedGlobs: string[];
  routingContext?: ForecastLabRoutingContext;
  effectiveMutationContract?: ForecastLabProfileMutationConfig;
  mutationMode?: ForecastLabMutationMode;
  parentRunId?: string;
  mutationId?: string;
  mutationSummary?: string;
  lineage?: ForecastLabMutationLineage;
  mutationSpecSummary?: ForecastLabMutationSpecSummary;
  candidateWorkspace?: ForecastLabCandidateWorkspaceMetadata;
  baselineSummary: JsonValue;
  candidateSummary: JsonValue;
  decision: ForecastLabDecision;
  reason: string;
  artifactsPath: string;
}

export interface ForecastLabRunManifest {
  runId: string;
  startedAt: string;
  profileId: string;
  targetSubsystem: string;
  baselineCommit?: string;
  candidateBranch: string;
  allowedGlobs: string[];
  routingContext?: ForecastLabRoutingContext;
  effectiveMutationContract?: ForecastLabProfileMutationConfig;
  mutationMode?: ForecastLabMutationMode;
  parentRunId?: string;
  mutationId?: string;
  mutationSummary?: string;
  lineage?: ForecastLabMutationLineage;
  mutationSpecSummary?: ForecastLabMutationSpecSummary;
  mutationReplayPayload?: ForecastLabMarkovParameterMutationReplayPayload;
  candidateWorkspace?: ForecastLabCandidateWorkspaceMetadata;
  artifactsPath: string;
}
