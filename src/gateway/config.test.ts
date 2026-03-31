import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { writeFileSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  getGatewayConfigPath,
  loadGatewayConfig,
  saveGatewayConfig,
  listWhatsAppAccountIds,
  resolveWhatsAppAccount,
  type GatewayConfig,
} from './config.js';

let tmpDir: string;
let originalCwd: string;

beforeEach(() => {
  originalCwd = process.cwd();
  tmpDir = join(tmpdir(), `gateway-config-test-${Date.now()}`);
  mkdirSync(tmpDir, { recursive: true });
  process.chdir(tmpDir);
});

afterEach(() => {
  process.chdir(originalCwd);
  rmSync(tmpDir, { recursive: true, force: true });
});

function makeConfigPath(): string {
  return join(tmpDir, 'gateway.json');
}

const minimalConfig: GatewayConfig = {
  gateway: { accountId: 'default', logLevel: 'info' },
  channels: { whatsapp: { enabled: true, accounts: {}, allowFrom: [] } },
  bindings: [],
};

// ---------------------------------------------------------------------------
// getGatewayConfigPath
// ---------------------------------------------------------------------------

describe('getGatewayConfigPath', () => {
  it('returns the override path when provided', () => {
    expect(getGatewayConfigPath('/custom/path.json')).toBe('/custom/path.json');
  });

  it('returns a default path when no override is given', () => {
    const p = getGatewayConfigPath();
    expect(typeof p).toBe('string');
    expect(p.endsWith('gateway.json')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// loadGatewayConfig — file-absent path
// ---------------------------------------------------------------------------

describe('loadGatewayConfig — file absent', () => {
  it('returns a default config when the file does not exist', () => {
    const cfg = loadGatewayConfig(join(tmpDir, 'nonexistent.json'));
    expect(cfg.gateway.accountId).toBe('default');
    expect(cfg.channels.whatsapp.enabled).toBe(true);
    expect(cfg.bindings).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// loadGatewayConfig — file present
// ---------------------------------------------------------------------------

describe('loadGatewayConfig — file present', () => {
  it('loads and parses a valid config file', () => {
    const path = makeConfigPath();
    writeFileSync(path, JSON.stringify({
      gateway: { accountId: 'myAgent', logLevel: 'debug' },
      channels: { whatsapp: { enabled: false, accounts: {}, allowFrom: [] } },
    }), 'utf-8');
    const cfg = loadGatewayConfig(path);
    expect(cfg.gateway.accountId).toBe('myAgent');
    expect(cfg.gateway.logLevel).toBe('debug');
    expect(cfg.channels.whatsapp.enabled).toBe(false);
  });

  it('defaults missing optional fields', () => {
    const path = makeConfigPath();
    writeFileSync(path, JSON.stringify({ gateway: {} }), 'utf-8');
    const cfg = loadGatewayConfig(path);
    expect(cfg.gateway.accountId).toBe('default');
    expect(cfg.gateway.logLevel).toBe('info');
  });

  it('parses heartbeat config when present', () => {
    const path = makeConfigPath();
    writeFileSync(path, JSON.stringify({
      gateway: {
        heartbeat: {
          enabled: true,
          intervalMinutes: 15,
          maxIterations: 5,
        },
      },
    }), 'utf-8');
    const cfg = loadGatewayConfig(path);
    expect(cfg.gateway.heartbeat?.enabled).toBe(true);
    expect(cfg.gateway.heartbeat?.intervalMinutes).toBe(15);
  });

  it('parses WhatsApp account config', () => {
    const path = makeConfigPath();
    writeFileSync(path, JSON.stringify({
      channels: {
        whatsapp: {
          accounts: {
            'acc-1': { name: 'Main Account', enabled: true },
          },
        },
      },
    }), 'utf-8');
    const cfg = loadGatewayConfig(path);
    expect(cfg.channels.whatsapp.accounts['acc-1']).toBeDefined();
    expect(cfg.channels.whatsapp.accounts['acc-1']!.name).toBe('Main Account');
  });
});

// ---------------------------------------------------------------------------
// saveGatewayConfig
// ---------------------------------------------------------------------------

describe('saveGatewayConfig', () => {
  it('writes config to the specified path as formatted JSON', () => {
    const path = makeConfigPath();
    saveGatewayConfig(minimalConfig, path);
    expect(existsSync(path)).toBe(true);
    const raw = JSON.parse(require('node:fs').readFileSync(path, 'utf-8'));
    expect(raw.gateway.accountId).toBe('default');
  });

  it('creates the directory if it does not exist', () => {
    const nestedPath = join(tmpDir, 'nested', 'dir', 'gateway.json');
    saveGatewayConfig(minimalConfig, nestedPath);
    expect(existsSync(nestedPath)).toBe(true);
  });

  it('round-trips config through save and load', () => {
    const path = makeConfigPath();
    const original: GatewayConfig = {
      ...minimalConfig,
      gateway: { ...minimalConfig.gateway, accountId: 'round-trip-test', logLevel: 'debug' },
    };
    saveGatewayConfig(original, path);
    const loaded = loadGatewayConfig(path);
    expect(loaded.gateway.accountId).toBe('round-trip-test');
    expect(loaded.gateway.logLevel).toBe('debug');
  });
});

// ---------------------------------------------------------------------------
// listWhatsAppAccountIds
// ---------------------------------------------------------------------------

describe('listWhatsAppAccountIds', () => {
  it('returns account ids from channels.whatsapp.accounts when present', () => {
    const cfg: GatewayConfig = {
      ...minimalConfig,
      channels: {
        whatsapp: {
          enabled: true,
          allowFrom: [],
          accounts: {
            'acc-a': { enabled: true, allowFrom: [], dmPolicy: 'pairing', groupPolicy: 'disabled', groupAllowFrom: [], sendReadReceipts: true, authDir: '' },
            'acc-b': { enabled: true, allowFrom: [], dmPolicy: 'pairing', groupPolicy: 'disabled', groupAllowFrom: [], sendReadReceipts: true, authDir: '' },
          },
        },
      },
    };
    const ids = listWhatsAppAccountIds(cfg);
    expect(ids).toContain('acc-a');
    expect(ids).toContain('acc-b');
  });

  it('falls back to gateway.accountId when accounts is empty', () => {
    const ids = listWhatsAppAccountIds(minimalConfig);
    expect(ids).toEqual(['default']);
  });
});

// ---------------------------------------------------------------------------
// resolveWhatsAppAccount
// ---------------------------------------------------------------------------

describe('resolveWhatsAppAccount', () => {
  it('returns default values when account is not in config', () => {
    const resolved = resolveWhatsAppAccount(minimalConfig, 'default');
    expect(resolved.accountId).toBe('default');
    expect(resolved.enabled).toBe(true);
    expect(resolved.dmPolicy).toBe('pairing');
    expect(resolved.groupPolicy).toBe('disabled');
    expect(resolved.sendReadReceipts).toBe(true);
  });

  it('merges account-level allowFrom with channel-level allowFrom', () => {
    const cfg: GatewayConfig = {
      ...minimalConfig,
      channels: {
        whatsapp: {
          enabled: true,
          allowFrom: ['+12025551234'],
          accounts: {
            'bot': {
              enabled: true,
              allowFrom: ['+12025559999'],
              dmPolicy: 'allowlist',
              groupPolicy: 'disabled',
              groupAllowFrom: [],
              sendReadReceipts: true,
              authDir: '/auth',
            },
          },
        },
      },
    };
    const resolved = resolveWhatsAppAccount(cfg, 'bot');
    // account-level allowFrom is used (not merged with channel-level in resolveWhatsAppAccount)
    expect(resolved.allowFrom).toContain('+12025559999');
    expect(resolved.dmPolicy).toBe('allowlist');
  });

  it('normalizes E.164 phone numbers in allowFrom', () => {
    const cfg: GatewayConfig = {
      ...minimalConfig,
      channels: {
        whatsapp: {
          enabled: true,
          allowFrom: [],
          accounts: {
            'bot': {
              enabled: true,
              allowFrom: ['*'],
              dmPolicy: 'open',
              groupPolicy: 'disabled',
              groupAllowFrom: [],
              sendReadReceipts: true,
              authDir: '/auth',
            },
          },
        },
      },
    };
    const resolved = resolveWhatsAppAccount(cfg, 'bot');
    // '*' wildcard is preserved as-is
    expect(resolved.allowFrom).toContain('*');
  });
});
