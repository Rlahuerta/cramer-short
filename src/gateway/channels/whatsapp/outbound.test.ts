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
