import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { assertOutboundAllowed } from './outbound.js';

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
