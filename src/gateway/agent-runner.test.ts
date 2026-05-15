import { describe, expect, it, mock } from 'bun:test';
import { BufferedEventForwarder } from './agent-runner.js';
import type { AgentEvent } from '../agent/types.js';

describe('BufferedEventForwarder', () => {
  it('delivers events emitted before the request handler is attached', async () => {
    const forwarder = new BufferedEventForwarder();
    const earlyEvent = { type: 'thinking', message: 'initializing' } satisfies AgentEvent;
    const onEvent = mock(async (_event: AgentEvent) => {});

    await forwarder.forward(earlyEvent);
    await forwarder.setHandler(onEvent);

    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledWith(earlyEvent);
  });

  it('does not deliver inactive-session events to the next request handler', async () => {
    const forwarder = new BufferedEventForwarder();
    const firstHandler = mock(async (_event: AgentEvent) => {});
    const secondHandler = mock(async (_event: AgentEvent) => {});

    await forwarder.setHandler(firstHandler);
    forwarder.clearHandler();
    await forwarder.forward({ type: 'thinking', message: 'late event' });
    await forwarder.setHandler(secondHandler);

    expect(secondHandler).not.toHaveBeenCalled();
  });
});
