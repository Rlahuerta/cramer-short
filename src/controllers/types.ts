import type { DisplayEvent, TokenUsage } from '../agent/types.js';

export type WorkingState =
  | { status: 'idle' }
  | { status: 'thinking'; iteration?: number; maxIterations?: number }
  | { status: 'tool'; toolName: string; iteration?: number; maxIterations?: number }
  | { status: 'approval'; toolName: string };

export type HistoryItemStatus = 'processing' | 'complete' | 'error' | 'interrupted';

export interface HistoryItem {
  id: string;
  query: string;
  events: DisplayEvent[];
  answer: string;
  status: HistoryItemStatus;
  activeToolId?: string;
  startTime?: number;
  duration?: number;
  tokenUsage?: TokenUsage;
  tokensPerSecond?: number;
}

/** True once an exchange has ended (complete, interrupted or error) and can be flushed to scrollback. */
export function isFinishedHistoryItem(item: HistoryItem): boolean {
  return item.status !== 'processing';
}
