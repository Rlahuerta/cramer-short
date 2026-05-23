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
export type ForecastLabPromotionStatus = 'approval-required' | 'approved' | 'promoted' | 'activated';

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

export interface ForecastLabPromotionSourceRef {
  runId: string;
  manifestPath: string;
}

export interface ForecastLabPromotionActivationRef {
  runId: string;
  manifestPath: string;
  artifactsPath: string;
  workspace: ForecastLabCandidateWorkspaceMetadata;
}

export interface ForecastLabApprovalRequiredPromotionState {
  status: 'approval-required';
  source: ForecastLabPromotionSourceRef;
  requestedAt: string;
}

export interface ForecastLabApprovedPromotionState {
  status: 'approved';
  source: ForecastLabPromotionSourceRef;
  requestedAt: string;
  approvedAt: string;
}

export interface ForecastLabPromotedPromotionState {
  status: 'promoted';
  source: ForecastLabPromotionSourceRef;
  requestedAt: string;
  approvedAt: string;
  promotedAt: string;
  activation: ForecastLabPromotionActivationRef;
}

export interface ForecastLabActivatedPromotionState {
  status: 'activated';
  source: ForecastLabPromotionSourceRef;
  requestedAt: string;
  approvedAt: string;
  promotedAt: string;
  activatedAt: string;
  activation: ForecastLabPromotionActivationRef;
}

export type ForecastLabPromotionState =
  | ForecastLabApprovalRequiredPromotionState
  | ForecastLabApprovedPromotionState
  | ForecastLabPromotedPromotionState
  | ForecastLabActivatedPromotionState;

export type ForecastLabPromotionOutcome =
  | {
      kind: 'approval-required';
      promotion: ForecastLabApprovalRequiredPromotionState;
    }
  | {
      kind: 'approved';
      promotion: ForecastLabApprovedPromotionState;
    }
  | {
      kind: 'promoted';
      promotion: ForecastLabPromotedPromotionState;
    }
  | {
      kind: 'activated';
      promotion: ForecastLabActivatedPromotionState;
    };

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
  promotion?: ForecastLabPromotionState;
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
  promotion?: ForecastLabPromotionState;
  promotionSource?: ForecastLabPromotionSourceRef;
  artifactsPath: string;
}
