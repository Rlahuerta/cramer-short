import { createChannelManager } from './channels/manager.js';
import { createWhatsAppPlugin } from './channels/whatsapp/plugin.js';
import {
  assertOutboundAllowed,
  sendComposing,
  sendMessageWhatsApp,
  type WhatsAppInboundMessage,
} from './channels/whatsapp/index.js';
import { resolveRoute } from './routing/resolve-route.js';
import { resolveSessionStorePath, upsertSessionMeta } from './sessions/store.js';
import { loadGatewayConfig, type GatewayConfig } from './config.js';
import { runAgentForMessage } from './agent-runner.js';
import { cleanMarkdownForWhatsApp } from './utils.js';
import { startHeartbeatRunner } from './heartbeat/index.js';
import {
  isBotMentioned,
  recordGroupMessage,
  getAndClearGroupHistory,
  formatGroupHistoryContext,
  noteGroupMember,
  formatGroupMembersList,
} from './group/index.js';
import type { GroupContext } from '../agent/prompts.js';
import { appendFileSync } from 'node:fs';
import { cramerShortPath } from '../utils/paths.js';
import { getSetting } from '../utils/config.js';

const LOG_PATH = cramerShortPath('gateway-debug.log');
function debugLog(msg: string) {
  appendFileSync(LOG_PATH, `${new Date().toISOString()} ${msg}\n`);
}

export type GatewayService = {
  stop: () => Promise<void>;
  snapshot: () => Record<string, { accountId: string; running: boolean; connected?: boolean }>;
};

export type GatewayRuntime = {
  createWhatsAppPlugin: typeof createWhatsAppPlugin;
  createChannelManager: typeof createChannelManager;
  loadGatewayConfig: typeof loadGatewayConfig;
  assertOutboundAllowed: typeof assertOutboundAllowed;
  sendComposing: typeof sendComposing;
  sendMessageWhatsApp: typeof sendMessageWhatsApp;
  resolveRoute: typeof resolveRoute;
  resolveSessionStorePath: typeof resolveSessionStorePath;
  upsertSessionMeta: typeof upsertSessionMeta;
  runAgentForMessage: typeof runAgentForMessage;
  cleanMarkdownForWhatsApp: typeof cleanMarkdownForWhatsApp;
  startHeartbeatRunner: typeof startHeartbeatRunner;
  isBotMentioned: typeof isBotMentioned;
  recordGroupMessage: typeof recordGroupMessage;
  getAndClearGroupHistory: typeof getAndClearGroupHistory;
  formatGroupHistoryContext: typeof formatGroupHistoryContext;
  noteGroupMember: typeof noteGroupMember;
  formatGroupMembersList: typeof formatGroupMembersList;
  getSetting: typeof getSetting;
  debugLog: (msg: string) => void;
};

const defaultRuntime: GatewayRuntime = {
  createWhatsAppPlugin,
  createChannelManager,
  loadGatewayConfig,
  assertOutboundAllowed,
  sendComposing,
  sendMessageWhatsApp,
  resolveRoute,
  resolveSessionStorePath,
  upsertSessionMeta,
  runAgentForMessage,
  cleanMarkdownForWhatsApp,
  startHeartbeatRunner,
  isBotMentioned,
  recordGroupMessage,
  getAndClearGroupHistory,
  formatGroupHistoryContext,
  noteGroupMember,
  formatGroupMembersList,
  getSetting,
  debugLog,
};

function elide(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + '...';
}

async function handleInbound(
  cfg: GatewayConfig,
  inbound: WhatsAppInboundMessage,
  runtime: GatewayRuntime,
): Promise<void> {
  const bodyPreview = elide(inbound.body.replace(/\n/g, ' '), 50);
  const isGroup = inbound.chatType === 'group';
  console.log(`Inbound message ${inbound.from} (${inbound.chatType}, ${inbound.body.length} chars): "${bodyPreview}"`);
  runtime.debugLog(`[gateway] handleInbound from=${inbound.from} isGroup=${isGroup} body="${inbound.body.slice(0, 30)}..."`);

  // --- Group-specific: track member, check mention gating ---
  if (isGroup) {
    runtime.noteGroupMember(inbound.chatId, inbound.senderId, inbound.senderName);

    const mentioned = runtime.isBotMentioned({
      mentionedJids: inbound.mentionedJids,
      selfJid: inbound.selfJid,
      selfLid: inbound.selfLid,
      selfE164: inbound.selfE164,
      body: inbound.body,
    });
    runtime.debugLog(`[gateway] group mention check: mentioned=${mentioned}`);

    if (!mentioned) {
      // Buffer the message for future context but don't reply
      runtime.recordGroupMessage(inbound.chatId, {
        senderName: inbound.senderName ?? inbound.senderId,
        senderId: inbound.senderId,
        body: inbound.body,
        timestamp: inbound.timestamp ?? Date.now(),
      });
      runtime.debugLog(`[gateway] group message buffered (no mention), skipping reply`);
      return;
    }
  }

  // --- Routing: use chatId for groups (group JID), senderId for DMs ---
  const peerId = isGroup ? inbound.chatId : inbound.senderId;
  const route = runtime.resolveRoute({
    cfg,
    channel: 'whatsapp',
    accountId: inbound.accountId,
    peer: { kind: inbound.chatType, id: peerId },
  });

  const storePath = runtime.resolveSessionStorePath(route.agentId);
  runtime.upsertSessionMeta({
    storePath,
    sessionKey: route.sessionKey,
    channel: 'whatsapp',
    to: inbound.from,
    accountId: route.accountId,
    agentId: route.agentId,
  });

  // Start typing indicator loop to keep it alive during long agent runs
  const TYPING_INTERVAL_MS = 5000; // Refresh every 5 seconds
  let typingTimer: ReturnType<typeof setInterval> | undefined;

  const startTypingLoop = async () => {
    // For groups, use inbound.sendComposing directly (bypasses outbound strict checks)
    if (isGroup) {
      await inbound.sendComposing();
      typingTimer = setInterval(() => { void inbound.sendComposing(); }, TYPING_INTERVAL_MS);
    } else {
      await runtime.sendComposing({ to: inbound.replyToJid, accountId: inbound.accountId });
      typingTimer = setInterval(() => {
        void runtime.sendComposing({ to: inbound.replyToJid, accountId: inbound.accountId });
      }, TYPING_INTERVAL_MS);
    }
  };

  const stopTypingLoop = () => {
    if (typingTimer) {
      clearInterval(typingTimer);
      typingTimer = undefined;
    }
  };

  try {
    // Defense-in-depth: verify outbound destination is allowed before any messaging
    // For groups, use chatId (the group JID); for DMs, use replyToJid
    const outboundTarget = isGroup ? inbound.chatId : inbound.replyToJid;
    try {
      runtime.assertOutboundAllowed({ to: outboundTarget, accountId: inbound.accountId });
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      runtime.debugLog(`[gateway] outbound BLOCKED: ${msg}`);
      console.log(msg);
      return;
    }

    await startTypingLoop();

    // --- Build query: for groups, include buffered history context ---
    let query = inbound.body;
    let groupContext: GroupContext | undefined;

    if (isGroup) {
      const history = runtime.getAndClearGroupHistory(inbound.chatId);
      query = runtime.formatGroupHistoryContext({
        history,
        currentSenderName: inbound.senderName ?? inbound.senderId,
        currentSenderId: inbound.senderId,
        currentBody: inbound.body,
      });
      runtime.debugLog(`[gateway] group query with ${history.length} history entries`);

      const membersList = runtime.formatGroupMembersList({
        groupId: inbound.chatId,
        participants: inbound.groupParticipants,
      });
      groupContext = {
        groupName: inbound.groupSubject,
        membersList: membersList || undefined,
        activationMode: 'mention',
      };
    }

    console.log(`Processing message with agent...`);
    runtime.debugLog(`[gateway] running agent for session=${route.sessionKey}`);
    const startedAt = Date.now();
    const model = runtime.getSetting('modelId', 'gpt-5.4') as string;
    const modelProvider = runtime.getSetting('provider', 'openai') as string;
    const answer = await runtime.runAgentForMessage({
      sessionKey: route.sessionKey,
      query,
      model,
      modelProvider,
      channel: 'whatsapp',
      groupContext,
    });
    const durationMs = Date.now() - startedAt;
    runtime.debugLog(`[gateway] agent answer length=${answer.length}`);

    // Stop typing loop before sending reply
    stopTypingLoop();

    if (answer.trim()) {
      const cleanedAnswer = runtime.cleanMarkdownForWhatsApp(answer);

      if (isGroup) {
        // For groups, use inbound.reply() directly (bypasses outbound strict E.164 checks)
        runtime.debugLog(`[gateway] sending group reply to ${inbound.chatId}`);
        await inbound.reply(cleanedAnswer);
      } else {
        runtime.debugLog(`[gateway] sending reply to ${inbound.replyToJid}`);
        await runtime.sendMessageWhatsApp({
          to: inbound.replyToJid,
          body: cleanedAnswer,
          accountId: inbound.accountId,
        });
      }
      console.log(`Sent reply (${answer.length} chars, ${durationMs}ms)`);
      runtime.debugLog(`[gateway] reply sent`);
    } else {
      console.log(`Agent returned empty response (${durationMs}ms)`);
      runtime.debugLog(`[gateway] empty answer, not sending`);
    }
  } catch (err) {
    stopTypingLoop();
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`Error: ${msg}`);
    runtime.debugLog(`[gateway] ERROR: ${msg}`);
  }
}

export async function startGateway(
  params: { configPath?: string; runtime?: Partial<GatewayRuntime> } = {},
): Promise<GatewayService> {
  const runtime: GatewayRuntime = { ...defaultRuntime, ...params.runtime };
  const cfg = runtime.loadGatewayConfig(params.configPath);
  const plugin = runtime.createWhatsAppPlugin({
    loadConfig: () => runtime.loadGatewayConfig(params.configPath),
    onMessage: async (inbound) => {
      const current = runtime.loadGatewayConfig(params.configPath);
      await handleInbound(current, inbound, runtime);
    },
  });
  const manager = runtime.createChannelManager({
    plugin,
    loadConfig: () => runtime.loadGatewayConfig(params.configPath),
  });
  await manager.startAll();

  const heartbeat = runtime.startHeartbeatRunner({ configPath: params.configPath });

  return {
    stop: async () => {
      heartbeat.stop();
      await manager.stopAll();
    },
    snapshot: () => manager.getSnapshot(),
  };
}
