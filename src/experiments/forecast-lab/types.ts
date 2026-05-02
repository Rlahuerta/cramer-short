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
  candidateBranch: string;
  allowedGlobs: string[];
  artifactsPath: string;
}
