import { describe, expect, it, mock, afterAll } from 'bun:test';
import { nextTestId } from '@/utils/test-determinism.js';
import type { ReconnectPolicy } from './reconnect.js';

const monitorWebInboxMock = mock(async () => {
  throw new Error('connect failed');
});

// Snapshot the real modules so afterAll can undo mock.module (Bun never does);
// otherwise these mocks leak into every test file that runs later.
const realInboundModule = { ...(await import('./inbound.js')) };
afterAll(() => {
  mock.module('./inbound.js', () => realInboundModule);
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

  // Fake listener whose close() settles onClose, like monitorWebInbox().close().
  function fakeListener() {
    let resolveClose: (reason: { status: number; isLoggedOut: boolean }) => void = () => {};
    const onClose = new Promise<{ status: number; isLoggedOut: boolean }>((resolve) => {
      resolveClose = resolve;
    });
    const close = mock(async () => resolveClose({ status: 499, isLoggedOut: false }));
    return { sock: {}, onClose, close };
  }

  const settlesWithin = async (run: Promise<void>, ms: number): Promise<boolean> => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<boolean>((resolve) => {
      timer = setTimeout(() => resolve(false), ms);
    });
    const settled = await Promise.race([run.then(() => true), timeout]);
    clearTimeout(timer);
    return settled;
  };

  const baseParams = {
    accountId: 'default',
    authDir: '/tmp/does-not-exist',
    verbose: false,
    allowFrom: [],
    dmPolicy: 'open' as const,
    groupPolicy: 'disabled' as const,
    groupAllowFrom: [],
    reconnect: fastBackoff,
    onMessage: async () => {},
  };

  it('closes the connected listener and resolves when aborted while connected', async () => {
    const controller = new AbortController();
    const listener = fakeListener();
    monitorWebInboxMock.mockImplementationOnce((async () => listener) as never);
    let markConnected: (() => void) | undefined;
    const connected = new Promise<void>((resolve) => {
      markConnected = resolve;
    });

    const run = monitorWhatsAppChannel({
      ...baseParams,
      abortSignal: controller.signal,
      onStatus: (status) => {
        if (status.connected) markConnected?.();
      },
    });

    await connected;
    controller.abort();

    expect(await settlesWithin(run, 500)).toBe(true);
    expect(listener.close).toHaveBeenCalled();
  });

  it('closes the listener when the signal aborted while the socket was still connecting', async () => {
    const controller = new AbortController();
    const listener = fakeListener();
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    monitorWebInboxMock.mockImplementationOnce((async () => {
      await gate;
      return listener;
    }) as never);

    const run = monitorWhatsAppChannel({ ...baseParams, abortSignal: controller.signal });

    controller.abort();
    release?.();

    expect(await settlesWithin(run, 500)).toBe(true);
    expect(listener.close).toHaveBeenCalled();
  });
});
