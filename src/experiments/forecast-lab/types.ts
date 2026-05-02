import type {
  ForecastLabCandidateWorkspaceMetadata,
  ForecastLabMutationLineage,
  ForecastLabMutationMode,
  ForecastLabProfileMutationConfig,
  ForecastLabMutationSpecSummary,
} from './mutation.js';

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
  lineage?: ForecastLabMutationLineage;
  mutationSpecSummary?: ForecastLabMutationSpecSummary;
  candidateWorkspace?: ForecastLabCandidateWorkspaceMetadata;
  artifactsPath: string;
}
