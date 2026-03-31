import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { writeFile, mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { parseSessionTranscripts } from './session-files.js';

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await mkdtemp(join(tmpdir(), 'session-files-test-'));
});

afterEach(async () => {
  await rm(tmpDir, { recursive: true, force: true });
});

function chatHistoryPath(): string {
  return join(tmpDir, 'chat_history.json');
}

function makeChatHistory(messages: Array<{
  id?: string; timestamp?: string; userMessage: string; agentResponse: string | null;
}>) {
  return JSON.stringify({
    messages: messages.map((m, i) => ({
      id: m.id ?? `msg-${i}`,
      timestamp: m.timestamp ?? '2024-01-15T12:00:00Z',
      userMessage: m.userMessage,
      agentResponse: m.agentResponse,
    })),
  });
}

describe('parseSessionTranscripts', () => {
  it('returns empty array when file does not exist', async () => {
    const result = await parseSessionTranscripts('/nonexistent/path/chat_history.json');
    expect(result).toHaveLength(0);
  });

  it('returns empty array when file content is not valid JSON', async () => {
    await writeFile(chatHistoryPath(), 'not-json', 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result).toHaveLength(0);
  });

  it('returns empty array when messages field is missing', async () => {
    await writeFile(chatHistoryPath(), JSON.stringify({ other: 'data' }), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result).toHaveLength(0);
  });

  it('returns empty array when messages array is empty', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result).toHaveLength(0);
  });

  it('skips messages with null agentResponse', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([
      { userMessage: 'Hello', agentResponse: null },
      { userMessage: 'How are you?', agentResponse: 'I am fine.' },
    ]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result).toHaveLength(1);
    expect(result[0]!.content).toContain('How are you?');
  });

  it('parses a valid conversation turn into SessionEntry', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([
      { userMessage: 'What is AAPL?', agentResponse: 'Apple Inc. stock.', timestamp: '2024-03-01T10:00:00Z' },
    ]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result).toHaveLength(1);
    expect(result[0]!.content).toContain('User: What is AAPL?');
    expect(result[0]!.content).toContain('Apple Inc. stock.');
    expect(result[0]!.timestamp).toBe('2024-03-01T10:00:00Z');
  });

  it('generates a sha256 contentHash for each entry', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([
      { userMessage: 'Test query', agentResponse: 'Test answer.' },
    ]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result[0]!.contentHash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('generates different hashes for different content', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([
      { userMessage: 'Q1', agentResponse: 'A1' },
      { userMessage: 'Q2', agentResponse: 'A2' },
    ]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result).toHaveLength(2);
    expect(result[0]!.contentHash).not.toBe(result[1]!.contentHash);
  });

  it('generates stable hashes (same content → same hash)', async () => {
    const history = makeChatHistory([{ userMessage: 'Hello', agentResponse: 'Hi there.' }]);
    const path1 = join(tmpDir, 'h1.json');
    const path2 = join(tmpDir, 'h2.json');
    await writeFile(path1, history, 'utf-8');
    await writeFile(path2, history, 'utf-8');
    const r1 = await parseSessionTranscripts(path1);
    const r2 = await parseSessionTranscripts(path2);
    expect(r1[0]!.contentHash).toBe(r2[0]!.contentHash);
  });

  it('formats content as "User: ...\n\nAssistant:\n\n..."', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([
      { userMessage: 'What is 2+2?', agentResponse: '4' },
    ]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result[0]!.content).toMatch(/^User: What is 2\+2\?\n\nAssistant:\n\n4$/);
  });

  it('handles multiple turns and returns all valid entries', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([
      { userMessage: 'Q1', agentResponse: 'A1' },
      { userMessage: 'Q2', agentResponse: 'A2' },
      { userMessage: 'Q3', agentResponse: null },
      { userMessage: 'Q4', agentResponse: 'A4' },
    ]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result).toHaveLength(3);
  });

  it('normalizes whitespace in userMessage but preserves newlines in agentResponse', async () => {
    await writeFile(chatHistoryPath(), makeChatHistory([
      { userMessage: '  What   is   AAPL?  ', agentResponse: 'Line one.\n\nLine two.' },
    ]), 'utf-8');
    const result = await parseSessionTranscripts(chatHistoryPath());
    expect(result[0]!.content).toContain('User: What is AAPL?');
    expect(result[0]!.content).toContain('Line one.\n\nLine two.');
  });
});
