import { describe, it, expect, beforeEach } from 'bun:test';
import { TokenCounter } from './token-counter.js';

describe('TokenCounter', () => {
  let counter: TokenCounter;

  beforeEach(() => {
    counter = new TokenCounter();
  });

  it('getUsage returns undefined when no tokens tracked', () => {
    expect(counter.getUsage()).toBeUndefined();
  });

  it('add accumulates token usage', () => {
    counter.add({ inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    const usage = counter.getUsage();
    expect(usage).toBeDefined();
    expect(usage!.inputTokens).toBe(10);
    expect(usage!.outputTokens).toBe(5);
    expect(usage!.totalTokens).toBe(15);
  });

  it('getUsage returns a copy of usage when tokens > 0', () => {
    counter.add({ inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    const usage1 = counter.getUsage();
    const usage2 = counter.getUsage();
    expect(usage1).not.toBe(usage2); // Different references
    expect(usage1).toEqual(usage2);  // Same values
  });

  it('add with undefined is a no-op', () => {
    counter.add(undefined);
    expect(counter.getUsage()).toBeUndefined();
  });

  it('getTokensPerSecond returns undefined when no tokens tracked', () => {
    expect(counter.getTokensPerSecond(1000)).toBeUndefined();
  });

  it('getTokensPerSecond returns undefined when elapsedMs <= 0', () => {
    counter.add({ inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    expect(counter.getTokensPerSecond(0)).toBeUndefined();
    expect(counter.getTokensPerSecond(-100)).toBeUndefined();
  });

  it('getTokensPerSecond calculates correctly', () => {
    counter.add({ inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    const tps = counter.getTokensPerSecond(1000); // 1 second
    expect(tps).toBe(15); // 15 tokens / 1 second
  });

  it('getTokensPerSecond handles fractional seconds', () => {
    counter.add({ inputTokens: 50, outputTokens: 50, totalTokens: 100 });
    const tps = counter.getTokensPerSecond(2000); // 2 seconds
    expect(tps).toBe(50); // 100 tokens / 2 seconds
  });

  it('accumulates from multiple add calls', () => {
    counter.add({ inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    counter.add({ inputTokens: 20, outputTokens: 10, totalTokens: 30 });
    const usage = counter.getUsage();
    expect(usage!.inputTokens).toBe(30);
    expect(usage!.outputTokens).toBe(15);
    expect(usage!.totalTokens).toBe(45);
  });

  it('getUsage remains undefined after adding zero tokens', () => {
    counter.add({ inputTokens: 0, outputTokens: 0, totalTokens: 0 });
    expect(counter.getUsage()).toBeUndefined();
  });
});
