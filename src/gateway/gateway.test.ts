import { afterAll, beforeEach, describe, expect, it, mock } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import type { GatewayConfig } from './config.js';
import type { GatewayRuntime } from './gateway.js';
import type { WhatsAppInboundMessage } from './channels/whatsapp/types.js';

const cfg: GatewayConfig = {
  gateway: { accountId: 'default', logLevel: 'info' },
  channels: { whatsapp: { enabled: true, accounts: {}, allowFrom: [] } },
  bindings: [],
};

let capturedOnMessage: ((msg: WhatsAppInboundMessage) => Promise<void>) | undefined;
let groupMentioned = false;

const createWhatsAppPluginMock = mock((params: { onMessage: (msg: WhatsAppInboundMessage) => Promise<void> }) => {
  capturedOnMessage = params.onMessage;
  return { name: 'whatsapp-test-plugin' };
});
const startAllMock = mock(async () => {});
const stopAllMock = mock(async () => {});
const getSnapshotMock = mock(() => ({ default: { accountId: 'default', running: true, connected: true } }));
const createChannelManagerMock = mock(() => ({
  startAll: startAllMock,
  stopAll: stopAllMock,
  getSnapshot: getSnapshotMock,
}));
const heartbeatStopMock = mock(() => {});
const loadGatewayConfigMock = mock(() => cfg);
const assertOutboundAllowedMock = mock(() => {});
const sendComposingMock = mock(async () => {});
const sendMessageWhatsAppMock = mock(async () => {});
const resolveRouteMock = mock(() => ({
  agentId: 'agent-default',
  accountId: 'default',
  sessionKey: 'whatsapp:default:+12025550100',
}));
const resolveSessionStorePathMock = mock(() => '.cramer-short-test/gateway/sessions.json');
const upsertSessionMetaMock = mock(() => {});
const runAgentForMessageMock = mock(async () => '**agent answer**');
const cleanMarkdownForWhatsAppMock = mock((text: string) => `clean:${text}`);
const isBotMentionedMock = mock(() => groupMentioned);
const recordGroupMessageMock = mock(() => {});
const getAndClearGroupHistoryMock = mock(() => [
  { senderName: 'Alice', senderId: '+12025550101', body: 'earlier context', timestamp: 1 },
]);
const formatGroupHistoryContextMock = mock(() => 'formatted group query');
const noteGroupMemberMock = mock(() => {});
const formatGroupMembersListMock = mock(() => 'Alice (+12025550101)');
const getSettingMock = mock(<T>(_key: string, fallback: T): T => fallback);

const { startGateway } = await import('./gateway.js');

function runtime(): Partial<GatewayRuntime> {
  return {
    createWhatsAppPlugin: createWhatsAppPluginMock as unknown as GatewayRuntime['createWhatsAppPlugin'],
    createChannelManager: createChannelManagerMock as unknown as GatewayRuntime['createChannelManager'],
    loadGatewayConfig: loadGatewayConfigMock,
    assertOutboundAllowed: assertOutboundAllowedMock as unknown as GatewayRuntime['assertOutboundAllowed'],
    sendComposing: sendComposingMock,
    sendMessageWhatsApp: sendMessageWhatsAppMock as unknown as GatewayRuntime['sendMessageWhatsApp'],
    resolveRoute: resolveRouteMock as unknown as GatewayRuntime['resolveRoute'],
    resolveSessionStorePath: resolveSessionStorePathMock,
    upsertSessionMeta: upsertSessionMetaMock as unknown as GatewayRuntime['upsertSessionMeta'],
    runAgentForMessage: runAgentForMessageMock,
    cleanMarkdownForWhatsApp: cleanMarkdownForWhatsAppMock,
    startHeartbeatRunner: (() => ({ stop: heartbeatStopMock })) as GatewayRuntime['startHeartbeatRunner'],
    isBotMentioned: isBotMentionedMock,
    recordGroupMessage: recordGroupMessageMock,
    getAndClearGroupHistory: getAndClearGroupHistoryMock,
    formatGroupHistoryContext: formatGroupHistoryContextMock,
    noteGroupMember: noteGroupMemberMock,
    formatGroupMembersList: formatGroupMembersListMock,
    getSetting: getSettingMock,
    debugLog: () => {},
  };
}

function inbound(overrides: Partial<WhatsAppInboundMessage> = {}): WhatsAppInboundMessage {
  return {
    id: 'msg-1',
    accountId: 'default',
    chatId: '12025550100@s.whatsapp.net',
    replyToJid: '12025550100@s.whatsapp.net',
    chatType: 'direct',
    from: '+12025550100',
    senderId: '+12025550100',
    senderName: 'Alice',
    body: 'Analyze AAPL',
    sendComposing: mock(async () => {}),
    reply: mock(async () => {}),
    sendMedia: mock(async () => {}),
    ...overrides,
  };
}

beforeEach(() => {
  rmSync('.cramer-short-test/gateway-unit', { recursive: true, force: true });
  mkdirSync('.cramer-short-test/gateway-unit', { recursive: true });
  capturedOnMessage = undefined;
  groupMentioned = false;
  for (const fn of [
    createWhatsAppPluginMock,
    startAllMock,
    stopAllMock,
    getSnapshotMock,
    heartbeatStopMock,
    loadGatewayConfigMock,
    assertOutboundAllowedMock,
    sendComposingMock,
    sendMessageWhatsAppMock,
    resolveRouteMock,
    resolveSessionStorePathMock,
    upsertSessionMetaMock,
    runAgentForMessageMock,
    cleanMarkdownForWhatsAppMock,
    isBotMentionedMock,
    recordGroupMessageMock,
    getAndClearGroupHistoryMock,
    formatGroupHistoryContextMock,
    noteGroupMemberMock,
    formatGroupMembersListMock,
    getSettingMock,
  ]) {
    fn.mockClear();
  }
});

afterAll(() => {
  rmSync('.cramer-short-test/gateway-unit', { recursive: true, force: true });
});

describe('startGateway orchestration', () => {
  it('starts the channel manager and stops heartbeat before channels', async () => {
    const service = await startGateway({ configPath: 'gateway.json', runtime: runtime() });

    expect(createWhatsAppPluginMock).toHaveBeenCalled();
    expect(createChannelManagerMock).toHaveBeenCalled();
    expect(startAllMock).toHaveBeenCalledTimes(1);
    expect(service.snapshot()).toEqual({ default: { accountId: 'default', running: true, connected: true } });

    await service.stop();
    expect(heartbeatStopMock).toHaveBeenCalledTimes(1);
    expect(stopAllMock).toHaveBeenCalledTimes(1);
  });

  it('routes direct WhatsApp inbound messages through the agent and outbound sender', async () => {
    await startGateway({ configPath: 'gateway.json', runtime: runtime() });
    await capturedOnMessage!(inbound());

    expect(assertOutboundAllowedMock).toHaveBeenCalledWith({
      to: '12025550100@s.whatsapp.net',
      accountId: 'default',
    });
    expect(upsertSessionMetaMock).toHaveBeenCalledWith(expect.objectContaining({
      sessionKey: 'whatsapp:default:+12025550100',
      to: '+12025550100',
    }));
    expect(sendComposingMock).toHaveBeenCalledWith({
      to: '12025550100@s.whatsapp.net',
      accountId: 'default',
    });
    expect(runAgentForMessageMock).toHaveBeenCalledWith(expect.objectContaining({
      sessionKey: 'whatsapp:default:+12025550100',
      query: 'Analyze AAPL',
      channel: 'whatsapp',
      model: 'gpt-5.4',
      modelProvider: 'openai',
    }));
    expect(sendMessageWhatsAppMock).toHaveBeenCalledWith({
      to: '12025550100@s.whatsapp.net',
      body: 'clean:**agent answer**',
      accountId: 'default',
    });
  });

  it('buffers unmentioned group messages without running the agent', async () => {
    await startGateway({ configPath: 'gateway.json', runtime: runtime() });
    await capturedOnMessage!(inbound({
      chatId: 'group-1@g.us',
      replyToJid: 'group-1@g.us',
      chatType: 'group',
      senderId: '+12025550101',
      body: 'background chatter',
    }));

    expect(noteGroupMemberMock).toHaveBeenCalledWith('group-1@g.us', '+12025550101', 'Alice');
    expect(recordGroupMessageMock).toHaveBeenCalledWith('group-1@g.us', expect.objectContaining({
      body: 'background chatter',
      senderId: '+12025550101',
    }));
    expect(runAgentForMessageMock).not.toHaveBeenCalled();
    expect(sendMessageWhatsAppMock).not.toHaveBeenCalled();
  });

  it('replies in mentioned groups using group context and inbound.reply', async () => {
    groupMentioned = true;
    const replyMock = mock(async () => {});
    await startGateway({ configPath: 'gateway.json', runtime: runtime() });
    await capturedOnMessage!(inbound({
      chatId: 'group-1@g.us',
      replyToJid: 'group-1@g.us',
      chatType: 'group',
      senderId: '+12025550101',
      body: '@bot analyze NVDA',
      groupSubject: 'Research chat',
      groupParticipants: ['+12025550101'],
      reply: replyMock,
    }));

    expect(formatGroupHistoryContextMock).toHaveBeenCalled();
    expect(runAgentForMessageMock).toHaveBeenCalledWith(expect.objectContaining({
      query: 'formatted group query',
      groupContext: {
        groupName: 'Research chat',
        membersList: 'Alice (+12025550101)',
        activationMode: 'mention',
      },
    }));
    expect(replyMock).toHaveBeenCalledWith('clean:**agent answer**');
    expect(sendMessageWhatsAppMock).not.toHaveBeenCalled();
  });
});
