import { afterEach, beforeEach, describe, expect, it, mock } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { assertOutboundAllowed, sendComposing, sendMessageWhatsApp, setActiveWebListener } from './outbound.js';
import { useTempCramerShortDir } from '../../../utils/test-cramer-short-dir.js';

useTempCramerShortDir();

let tmpDir: string;
let previousConfigPath: string | undefined;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'gateway-outbound-test-'));
  previousConfigPath = process.env.DEXTER_GATEWAY_CONFIG;
  writeFileSync(
    join(tmpDir, 'default.json'),
    JSON.stringify({ channels: { whatsapp: { allowFrom: ['+19998887777'] } } }),
    'utf-8',
  );
  process.env.DEXTER_GATEWAY_CONFIG = join(tmpDir, 'default.json');
});

afterEach(() => {
  if (previousConfigPath === undefined) {
    delete process.env.DEXTER_GATEWAY_CONFIG;
  } else {
    process.env.DEXTER_GATEWAY_CONFIG = previousConfigPath;
  }
  rmSync(tmpDir, { recursive: true, force: true });
});

describe('assertOutboundAllowed config path', () => {
  it('honors a caller-provided config path instead of the default', () => {
    const allowPath = join(tmpDir, 'allow.json');
    writeFileSync(
      allowPath,
      JSON.stringify({ channels: { whatsapp: { allowFrom: ['+12025550100'] } } }),
      'utf-8',
    );

    const result = assertOutboundAllowed({
      to: '+12025550100',
      accountId: 'default',
      configPath: allowPath,
    });

    expect(result.recipientE164).toBe('+12025550100');
  });

  it('still rejects a recipient absent from the provided config path', () => {
    const allowPath = join(tmpDir, 'allow.json');
    writeFileSync(
      allowPath,
      JSON.stringify({ channels: { whatsapp: { allowFrom: ['+12025550100'] } } }),
      'utf-8',
    );

    expect(() => assertOutboundAllowed({
      to: '+12025550999',
      accountId: 'default',
      configPath: allowPath,
    })).toThrow(/not in allowFrom/);
  });
});

describe('outbound account socket selection', () => {
  const fakeSock = () => ({
    sendMessage: mock(async () => ({ key: { id: 'msg-1' } })),
    sendPresenceUpdate: mock(async () => {}),
  });

  afterEach(() => {
    setActiveWebListener('work', null);
    setActiveWebListener('personal', null);
  });

  it('fails instead of borrowing another account socket when the requested account is not connected', async () => {
    const personal = fakeSock();
    setActiveWebListener('personal', personal as never);

    await expect(
      sendMessageWhatsApp({ to: '+19998887777', body: 'hi', accountId: 'work' }),
    ).rejects.toThrow(/account work/);
    await expect(sendComposing({ to: '+19998887777', accountId: 'work' })).rejects.toThrow(
      /account work/,
    );

    expect(personal.sendMessage).not.toHaveBeenCalled();
    expect(personal.sendPresenceUpdate).not.toHaveBeenCalled();
  });

  it('sends through the requested account socket when several accounts are connected', async () => {
    const personal = fakeSock();
    const work = fakeSock();
    setActiveWebListener('personal', personal as never);
    setActiveWebListener('work', work as never);

    await sendMessageWhatsApp({ to: '+19998887777', body: 'hi', accountId: 'work' });

    expect(work.sendMessage).toHaveBeenCalledTimes(1);
    expect(personal.sendMessage).not.toHaveBeenCalled();
  });
});

describe('outbound wildcard / open policy allowlist', () => {
  const writeConfig = (account: Record<string, unknown>): string => {
    const path = join(tmpDir, 'wildcard.json');
    writeFileSync(
      path,
      JSON.stringify({ channels: { whatsapp: { accounts: { default: account } } } }),
      'utf-8',
    );
    return path;
  };

  it('allows a reply to an inbound sender when the allowlist is the wildcard', () => {
    const configPath = writeConfig({ dmPolicy: 'allowlist', allowFrom: ['*'] });

    const result = assertOutboundAllowed({
      to: '+12025550999',
      accountId: 'default',
      configPath,
      replyToInbound: true,
    });

    expect(result.recipientE164).toBe('+12025550999');
  });

  it('allows a reply to an inbound sender under dmPolicy open with an empty allowFrom', () => {
    const configPath = writeConfig({ dmPolicy: 'open', allowFrom: [] });

    const result = assertOutboundAllowed({
      to: '+12025550999',
      accountId: 'default',
      configPath,
      replyToInbound: true,
    });

    expect(result.recipientE164).toBe('+12025550999');
  });

  it('still blocks a proactive (non-reply) send when the allowlist is only the wildcard', () => {
    const configPath = writeConfig({ dmPolicy: 'allowlist', allowFrom: ['*'] });

    expect(() =>
      assertOutboundAllowed({ to: '+12025550999', accountId: 'default', configPath }),
    ).toThrow(/no explicit allowFrom recipient/);
  });

  it('still blocks a proactive send to a number outside the explicit entries next to a wildcard', () => {
    const configPath = writeConfig({ dmPolicy: 'allowlist', allowFrom: ['*', '+12025550100'] });

    expect(() =>
      assertOutboundAllowed({ to: '+12025550999', accountId: 'default', configPath }),
    ).toThrow(/not in allowFrom/);
    expect(
      assertOutboundAllowed({ to: '+12025550100', accountId: 'default', configPath }).recipientE164,
    ).toBe('+12025550100');
  });

  it('still blocks a proactive send under dmPolicy open with an empty allowFrom', () => {
    const configPath = writeConfig({ dmPolicy: 'open', allowFrom: [] });

    expect(() =>
      assertOutboundAllowed({ to: '+12025550999', accountId: 'default', configPath }),
    ).toThrow(/no explicit allowFrom recipient/);
  });

  it('does not widen an explicit allowlist for replies', () => {
    const configPath = writeConfig({ dmPolicy: 'allowlist', allowFrom: ['+12025550100'] });

    expect(() =>
      assertOutboundAllowed({
        to: '+12025550999',
        accountId: 'default',
        configPath,
        replyToInbound: true,
      }),
    ).toThrow(/not in allowFrom/);
    expect(
      assertOutboundAllowed({
        to: '+12025550100',
        accountId: 'default',
        configPath,
        replyToInbound: true,
      }).recipientE164,
    ).toBe('+12025550100');
  });

  it('sendMessageWhatsApp and sendComposing honour the reply flag end to end', async () => {
    const configPath = writeConfig({ dmPolicy: 'allowlist', allowFrom: ['*'] });
    const sock = {
      sendMessage: mock(async () => ({ key: { id: 'msg-1' } })),
      sendPresenceUpdate: mock(async () => {}),
    };
    setActiveWebListener('default', sock as never);
    try {
      await expect(
        sendMessageWhatsApp({ to: '+12025550999', body: 'hi', accountId: 'default', configPath }),
      ).rejects.toThrow(/no explicit allowFrom recipient/);
      await expect(
        sendComposing({ to: '+12025550999', accountId: 'default', configPath }),
      ).rejects.toThrow(/no explicit allowFrom recipient/);
      expect(sock.sendMessage).not.toHaveBeenCalled();

      await sendMessageWhatsApp({
        to: '+12025550999',
        body: 'hi',
        accountId: 'default',
        configPath,
        replyToInbound: true,
      });
      await sendComposing({
        to: '+12025550999',
        accountId: 'default',
        configPath,
        replyToInbound: true,
      });
      expect(sock.sendMessage).toHaveBeenCalledTimes(1);
      expect(sock.sendPresenceUpdate).toHaveBeenCalledTimes(1);
    } finally {
      setActiveWebListener('default', null);
    }
  });
});
