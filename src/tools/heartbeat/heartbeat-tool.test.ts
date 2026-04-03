import { describe, expect, it, beforeEach, afterEach } from 'bun:test';
import { heartbeatTool, HEARTBEAT_TOOL_DESCRIPTION } from './heartbeat-tool.js';
import { mkdirSync, rmSync, existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

// ---------------------------------------------------------------------------
// We test the heartbeat tool by redirecting DEXTER_DIR to a temp folder so we
// don't touch the real .cramer-short/HEARTBEAT.md during the test run.
// The heartbeat-tool uses cramerShortPath() which joins getCramerShortDir() + filename.
// We override getCramerShortDir's underlying folder by writing to the real path but
// in a controlled tmp directory via process.chdir, then restore.
// ---------------------------------------------------------------------------

let tmpDir: string;
let originalCwd: string;

beforeEach(() => {
  originalCwd = process.cwd();
  tmpDir = join(tmpdir(), `dexter-heartbeat-test-${Date.now()}`);
  mkdirSync(tmpDir, { recursive: true });
  process.chdir(tmpDir);
});

afterEach(() => {
  process.chdir(originalCwd);
  rmSync(tmpDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// HEARTBEAT_TOOL_DESCRIPTION
// ---------------------------------------------------------------------------

describe('HEARTBEAT_TOOL_DESCRIPTION', () => {
  it('is a non-empty string', () => {
    expect(typeof HEARTBEAT_TOOL_DESCRIPTION).toBe('string');
    expect(HEARTBEAT_TOOL_DESCRIPTION.length).toBeGreaterThan(50);
  });

  it('describes both actions', () => {
    expect(HEARTBEAT_TOOL_DESCRIPTION).toContain('view');
    expect(HEARTBEAT_TOOL_DESCRIPTION).toContain('update');
  });
});

// ---------------------------------------------------------------------------
// view action
// ---------------------------------------------------------------------------

describe('heartbeatTool — view', () => {
  it('returns default message when no checklist file exists', async () => {
    const result = await heartbeatTool.invoke({ action: 'view' });
    expect(result).toContain('No heartbeat checklist');
    expect(result).toContain('update action');
  });

  it('returns the file content when checklist exists', async () => {
    const dexterDir = join(tmpDir, '.cramer-short');
    mkdirSync(dexterDir, { recursive: true });
    writeFileSync(join(dexterDir, 'HEARTBEAT.md'), '- Check AAPL\n- Check markets\n');

    const result = await heartbeatTool.invoke({ action: 'view' });
    expect(result).toContain('Check AAPL');
    expect(result).toContain('Check markets');
  });
});

// ---------------------------------------------------------------------------
// update action
// ---------------------------------------------------------------------------

describe('heartbeatTool — update', () => {
  it('returns error when content is missing', async () => {
    const result = await heartbeatTool.invoke({ action: 'update' });
    expect(result).toContain('content is required');
  });

  it('writes content to HEARTBEAT.md and returns summary with item count', async () => {
    const content = '- Watch AAPL\n- Watch NVDA\n- Check market open\n';
    const result = await heartbeatTool.invoke({ action: 'update', content });
    expect(result).toContain('3 items');
    // Verify the file was actually written
    const file = join(tmpDir, '.cramer-short', 'HEARTBEAT.md');
    expect(existsSync(file)).toBe(true);
  });

  it('creates the .cramer-short directory if it does not exist', async () => {
    const dexterDir = join(tmpDir, '.cramer-short');
    expect(existsSync(dexterDir)).toBe(false);

    await heartbeatTool.invoke({ action: 'update', content: '- Check news\n' });
    expect(existsSync(dexterDir)).toBe(true);
  });

  it('uses singular "item" when there is exactly 1 item', async () => {
    const result = await heartbeatTool.invoke({ action: 'update', content: '- Just one item\n' });
    expect(result).toContain('1 item');
    expect(result).not.toContain('1 items');
  });

  it('uses generic summary when no list items are in the content', async () => {
    const result = await heartbeatTool.invoke({ action: 'update', content: 'Just plain text, no dashes.' });
    expect(result).toContain('Updated heartbeat checklist');
    expect(result).not.toContain('items');
  });

  it('creates a gateway.json enabling heartbeat when items are present', async () => {
    await heartbeatTool.invoke({ action: 'update', content: '- Watch TSLA\n' });
    const gatewayPath = join(tmpDir, '.cramer-short', 'gateway.json');
    expect(existsSync(gatewayPath)).toBe(true);
    const cfg = JSON.parse(require('node:fs').readFileSync(gatewayPath, 'utf-8'));
    expect(cfg.gateway.heartbeat?.enabled).toBe(true);
  });

  it('does NOT create/modify gateway.json when content has no list items', async () => {
    await heartbeatTool.invoke({ action: 'update', content: 'No list items here.' });
    const gatewayPath = join(tmpDir, '.cramer-short', 'gateway.json');
    // gateway.json should not be created when no items are present
    expect(existsSync(gatewayPath)).toBe(false);
  });
});
