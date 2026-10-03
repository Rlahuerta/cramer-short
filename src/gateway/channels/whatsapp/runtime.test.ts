import { describe, expect, it, mock } from 'bun:test';
import { nextTestId } from '@/utils/test-determinism.js';
import type { ReconnectPolicy } from './reconnect.js';

const monitorWebInboxMock = mock(async () => {
  throw new Error('connect failed');
});

mock.module('./inbound.js', () => ({ monitorWebInbox: monitorWebInboxMock }));

const { monitorWhatsAppChannel } = await import(
  `./runtime.js?t=${nextTestId('runtime-abort')}`
) as typeof import('./runtime.js');

const fastBackoff: ReconnectPolicy = {
  initialMs: 5_000,
  maxMs: 5_000,
  factor: 1,
  jitter: 0,
  maxAttempts: 12,
};

describe('monitorWhatsAppChannel shutdown', () => {
  it('completes promptly when aborted during a reconnect backoff wait', async () => {
    const controller = new AbortController();
    let markBackoff: (() => void) | undefined;
    const backoffReached = new Promise<void>((resolve) => {
      markBackoff = resolve;
    });

    const run = monitorWhatsAppChannel({
      accountId: 'default',
      authDir: '/tmp/does-not-exist',
      verbose: false,
      allowFrom: [],
      dmPolicy: 'open',
      groupPolicy: 'disabled',
      groupAllowFrom: [],
      reconnect: fastBackoff,
      abortSignal: controller.signal,
      onMessage: async () => {},
      onStatus: (status) => {
        if (status.lastError?.includes('connect failed')) {
          markBackoff?.();
        }
      },
    });

    await backoffReached;
    const startedAt = Date.now();
    controller.abort();
    await run;

    expect(Date.now() - startedAt).toBeLessThan(100);
  });
});
