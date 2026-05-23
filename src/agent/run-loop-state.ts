export interface MemoryFlushState {
  alreadyFlushed: boolean;
}

export interface PeriodicMemoryFlushState {
  lastFlushedIteration: number;
}

export interface RunLoopState {
  sequentialThinkingUsed: boolean;
  sequentialThinkingRetries: number;
  sequentialThinkingCallCount: number;
  overflowRetries: number;
  forcedToolErrorPromptRebuilt: boolean;
  memoryFlush: MemoryFlushState;
  periodicFlush: PeriodicMemoryFlushState;
}

export function createRunLoopState(): RunLoopState {
  return {
    sequentialThinkingUsed: false,
    sequentialThinkingRetries: 0,
    sequentialThinkingCallCount: 0,
    overflowRetries: 0,
    forcedToolErrorPromptRebuilt: false,
    memoryFlush: { alreadyFlushed: false },
    periodicFlush: { lastFlushedIteration: 0 },
  };
}
