import { afterAll, beforeEach, describe, expect, it, mock } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';

type Handler = (payload: any) => unknown;

let eventHandlers: Record<string, Handler> = {};

const sendMessageMock = mock(async () => {});
const sendPresenceUpdateMock = mock(async () => {});
const readMessagesMock = mock(async () => {});
const groupMetadataMock = mock(async () => ({
  subject: 'Research group',
  participants: [{ id: '12025550101@s.whatsapp.net' }, { id: '12025550102@s.whatsapp.net' }],
}));
const closeSocketMock = mock(() => {});
const createWaSocketMock = mock(async () => ({
  user: { id: '12025550000@s.whatsapp.net', lid: 'bot@lid' },
  signalRepository: undefined,
  ev: {
    on: (event: string, handler: Handler) => { eventHandlers[event] = handler; },
    off: (event: string) => { delete eventHandlers[event]; },
  },
  sendMessage: sendMessageMock,
  sendPresenceUpdate: sendPresenceUpdateMock,
  readMessages: readMessagesMock,
  groupMetadata: groupMetadataMock,
  ws: { close: closeSocketMock },
}));
const waitForWaConnectionMock = mock(async () => {});
const checkInboundAccessControlMock = mock(async (params: { accountId: string }) => ({
  allowed: true,
  resolvedAccountId: params.accountId,
  shouldMarkRead: true,
  isSelfChat: false,
}));
const resolveJidToPhoneJidMock = mock(async (jid: string) => (
  jid === 'alice@lid' ? '12025550100@s.whatsapp.net' : null
));
const isRecentInboundMessageMock = mock(() => false);
const setActiveWebListenerMock = mock(() => {});

mock.module('@whiskeysockets/baileys', () => ({
  isJidGroup: (jid: string) => jid.endsWith('@g.us'),
  normalizeMessageContent: (message: unknown) => message,
  extractMessageContent: (message: unknown) => message,
}));
mock.module('../../../utils/paths.js', () => ({
  getCramerShortDir: () => '.cramer-short-test/whatsapp-inbound',
  cramerShortPath: (...segments: string[]) => ['.cramer-short-test', 'whatsapp-inbound', ...segments].join('/'),
  arbiterReplayCachePath: (...segments: string[]) =>
    ['.cramer-short-test', 'whatsapp-inbound', 'cache', 'arbiter-replay', ...segments].join('/'),
  experimentsPath: (...segments: string[]) =>
    ['.cramer-short-test', 'whatsapp-inbound', 'experiments', ...segments].join('/'),
  getExperimentsDir: () => '.cramer-short-test/whatsapp-inbound/experiments',
  getExperimentLedgerPath: () => '.cramer-short-test/whatsapp-inbound/experiments/forecast-results.tsv',
  getExperimentRunsDir: () => '.cramer-short-test/whatsapp-inbound/experiments/runs',
  getExperimentRunDir: (runId: string) =>
    ['.cramer-short-test', 'whatsapp-inbound', 'experiments', 'runs', runId].join('/'),
  getExperimentRunArtifactsDir: (runId: string) =>
    ['.cramer-short-test', 'whatsapp-inbound', 'experiments', 'runs', runId, 'artifacts'].join('/'),
  getExperimentRunManifestPath: (runId: string) =>
    ['.cramer-short-test', 'whatsapp-inbound', 'experiments', 'runs', runId, 'manifest.json'].join('/'),
}));
mock.module('./session.js', () => ({
  createWaSocket: createWaSocketMock,
  getStatusCode: () => 401,
  isLoggedOutReason: () => true,
  waitForWaConnection: waitForWaConnectionMock,
}));
mock.module('./outbound.js', () => ({ setActiveWebListener: setActiveWebListenerMock }));
mock.module('./dedupe.js', () => ({ isRecentInboundMessage: isRecentInboundMessageMock }));
mock.module('./auth-store.js', () => ({ readSelfId: () => ({ e164: '+12025550000' }) }));
mock.module('../../access-control.js', () => ({ checkInboundAccessControl: checkInboundAccessControlMock }));
mock.module('./lid.js', () => ({ resolveJidToPhoneJid: resolveJidToPhoneJidMock }));

const {
  extractMentionedJids,
  extractText,
  jidToE164,
  monitorWebInbox,
  toPhoneFromJid,
} = await import(`./inbound.js?t=${Date.now()}`) as typeof import('./inbound.js');

beforeEach(() => {
  rmSync('.cramer-short-test/whatsapp-inbound', { recursive: true, force: true });
  mkdirSync('.cramer-short-test/whatsapp-inbound', { recursive: true });
  eventHandlers = {};
  for (const fn of [
    sendMessageMock,
    sendPresenceUpdateMock,
    readMessagesMock,
    groupMetadataMock,
    closeSocketMock,
    createWaSocketMock,
    waitForWaConnectionMock,
    checkInboundAccessControlMock,
    resolveJidToPhoneJidMock,
    isRecentInboundMessageMock,
    setActiveWebListenerMock,
  ]) {
    fn.mockClear();
  }
});

afterAll(() => {
  rmSync('.cramer-short-test/whatsapp-inbound', { recursive: true, force: true });
});

describe('WhatsApp inbound parsing helpers', () => {
  it('extracts text from conversation, extended text, and media captions', () => {
    expect(extractText({ message: { conversation: '  hello  ' } } as any)).toBe('hello');
    expect(extractText({ message: { extendedTextMessage: { text: ' extended ' } } } as any)).toBe('extended');
    expect(extractText({ message: { imageMessage: { caption: ' chart caption ' } } } as any)).toBe('chart caption');
  });

  it('extracts mentioned JIDs and normalizes phone JIDs to E.164', () => {
    const message = {
      message: {
        extendedTextMessage: {
          text: '@bot',
          contextInfo: { mentionedJid: ['bot@s.whatsapp.net', 123, 'bot@lid'] },
        },
      },
    };

    expect(extractMentionedJids(message as any)).toEqual(['bot@s.whatsapp.net', 'bot@lid']);
    expect(toPhoneFromJid('12025550100:7@s.whatsapp.net')).toBe('+12025550100');
    expect(jidToE164('12025550100@s.whatsapp.net')).toBe('+12025550100');
    expect(jidToE164(null)).toBeNull();
  });
});

describe('monitorWebInbox inbound conversion', () => {
  it('converts direct messages into WhatsAppInboundMessage without connecting to WhatsApp', async () => {
    const received: any[] = [];
    await monitorWebInbox({
      accountId: 'default',
      authDir: '.cramer-short-test/auth',
      verbose: false,
      allowFrom: [],
      dmPolicy: 'open',
      groupPolicy: 'disabled',
      groupAllowFrom: [],
      onMessage: async (msg) => { received.push(msg); },
    });

    await eventHandlers['messages.upsert']!({
      type: 'notify',
      messages: [{
        key: { remoteJid: 'alice@lid', id: 'direct-1', fromMe: false },
        message: { conversation: '  Analyze AAPL  ' },
        messageTimestamp: 123,
        pushName: 'Alice',
      }],
    });

    expect(received).toHaveLength(1);
    expect(received[0]).toMatchObject({
      id: 'direct-1',
      accountId: 'default',
      chatId: 'alice@lid',
      replyToJid: '12025550100@s.whatsapp.net',
      chatType: 'direct',
      from: '+12025550100',
      senderId: '+12025550100',
      senderName: 'Alice',
      body: 'Analyze AAPL',
      timestamp: 123000,
    });
    expect(readMessagesMock).toHaveBeenCalledWith([
      { remoteJid: 'alice@lid', id: 'direct-1', participant: undefined, fromMe: false },
    ]);

    await received[0].sendComposing();
    await received[0].reply('hello');
    expect(sendPresenceUpdateMock).toHaveBeenCalledWith('composing', '12025550100@s.whatsapp.net');
    expect(sendMessageMock).toHaveBeenCalledWith('12025550100@s.whatsapp.net', { text: 'hello' });
  });

  it('converts group messages with mentions and metadata', async () => {
    const received: any[] = [];
    await monitorWebInbox({
      accountId: 'default',
      authDir: '.cramer-short-test/auth',
      verbose: false,
      allowFrom: [],
      dmPolicy: 'disabled',
      groupPolicy: 'open',
      groupAllowFrom: [],
      onMessage: async (msg) => { received.push(msg); },
    });

    await eventHandlers['messages.upsert']!({
      type: 'notify',
      messages: [{
        key: {
          remoteJid: 'research@g.us',
          participant: '12025550101@s.whatsapp.net',
          id: 'group-1',
          fromMe: false,
        },
        message: {
          extendedTextMessage: {
            text: ' @bot thoughts? ',
            contextInfo: { mentionedJid: ['bot@s.whatsapp.net'] },
          },
        },
        messageTimestamp: 456,
        pushName: 'Bob',
      }],
    });

    expect(received).toHaveLength(1);
    expect(received[0]).toMatchObject({
      chatId: 'research@g.us',
      replyToJid: 'research@g.us',
      chatType: 'group',
      from: '+12025550101',
      senderId: '+12025550101',
      groupSubject: 'Research group',
      groupParticipants: ['+12025550101', '+12025550102'],
      mentionedJids: ['bot@s.whatsapp.net'],
      body: '@bot thoughts?',
    });
  });

  it('marks append messages as read but does not auto-reply to offline history', async () => {
    const onMessageMock = mock(async () => {});
    await monitorWebInbox({
      accountId: 'default',
      authDir: '.cramer-short-test/auth',
      verbose: false,
      allowFrom: [],
      dmPolicy: 'open',
      groupPolicy: 'disabled',
      groupAllowFrom: [],
      onMessage: onMessageMock,
    });

    await eventHandlers['messages.upsert']!({
      type: 'append',
      messages: [{
        key: { remoteJid: 'alice@lid', id: 'history-1', fromMe: false },
        message: { conversation: 'historical message' },
        messageTimestamp: 100,
      }],
    });

    expect(readMessagesMock).toHaveBeenCalled();
    expect(onMessageMock).not.toHaveBeenCalled();
  });
});
