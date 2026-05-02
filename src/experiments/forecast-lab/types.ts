import type {
  ForecastLabCandidateWorkspaceMetadata,
  ForecastLabMutationLineage,
  ForecastLabMutationMode,
  ForecastLabProfileMutationConfig,
  ForecastLabMutationSpecSummary,
} from './mutation.js';
import type { ForecastLabMarkovParameterMutationReplayPayload } from './mutators/markov-parameters.js';

export type ForecastLabDecision = 'keep' | 'drop';

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface ForecastLabLedgerEntry {
  runId: string;
  startedAt: string;
  profileId: string;
  targetSubsystem: string;
  candidateBranch: string;
  allowedGlobs: string[];
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
