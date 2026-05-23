import { resolveLlmCallTimeoutMs } from '../../model/llm.js';
import { getSetting, setSetting, validateConfigValue } from '../../utils/config.js';

interface ConfigChatLog {
  clearAll(): void;
  addQuery(query: string): void;
  resetToolGrouping(): void;
  finalizeAnswer(answer: string): void;
}

interface HandleConfigSlashCommandOptions {
  readonly chatLog: ConfigChatLog;
  readonly currentModel: () => string;
  readonly refreshError: () => void;
  readonly requestRender: () => void;
  readonly setError: (message: string | null) => void;
  readonly setStatus: (message: string) => void;
  readonly setTimeoutFn?: typeof setTimeout;
}

export function buildConfigSummary(): string {
  const configKeys: Array<{ key: string; default: number }> = [
    { key: 'maxIterations',    default: 25 },
    { key: 'contextThreshold', default: 100000 },
    { key: 'keepToolUses',     default: 5 },
    { key: 'cacheTtlMs',       default: 900000 },
    { key: 'parallelToolLimit',default: 0 },
  ];
  const lines: string[] = ['Current Configuration:'];
  for (const { key, default: def } of configKeys) {
    const raw = getSetting<number | undefined>(key, undefined);
    const isDefault = raw === undefined;
    const display = raw ?? def;
    const extra = isDefault ? ' (default)' : '';
    const suffix = key === 'parallelToolLimit' && display === 0 ? ' (unlimited)' : '';
    lines.push(`  ${key.padEnd(18)} ${display}${suffix}${extra}`);
  }

  const timeout = resolveLlmCallTimeoutMs();
  const timeoutSource = timeout.source === 'config' ? '' :
                         timeout.source === 'env' ? ' (from env)' : ' (default)';
  lines.push(`  ${'llmCallTimeoutMs'.padEnd(18)} ${timeout.value}${timeoutSource}`);

  const provider = getSetting<string | undefined>('provider', undefined);
  const modelId  = getSetting<string | undefined>('modelId', undefined);
  if (provider) lines.push(`  ${'provider'.padEnd(18)} ${provider}`);
  if (modelId)  lines.push(`  ${'modelId'.padEnd(18)} ${modelId}`);

  return lines.join('\n');
}

export function handleConfigSlashCommand(
  query: string,
  options: HandleConfigSlashCommandOptions,
): boolean {
  if (query === '/config' || query === '/config show') {
    options.chatLog.clearAll();
    options.chatLog.addQuery(query);
    options.chatLog.resetToolGrouping();
    options.chatLog.finalizeAnswer(buildConfigSummary());
    options.requestRender();
    return true;
  }

  if (!query.startsWith('/config set ')) {
    return false;
  }

  const parts = query.slice('/config set '.length).trim().split(/\s+/);
  if (parts.length < 2) {
    options.setError('Usage: /config set <key> <value>');
    options.refreshError();
    options.requestRender();
    return true;
  }

  const [cfgKey, rawVal] = parts;
  const numVal = Number(rawVal);
  const value: unknown = Number.isFinite(numVal) ? numVal : rawVal;
  const validation = validateConfigValue(cfgKey, value);
  if (!validation.valid) {
    options.setError(`Config error: ${validation.error}`);
    options.refreshError();
    options.requestRender();
    return true;
  }

  const saved = setSetting(cfgKey, value);
  if (!saved) {
    options.setError('Failed to save config to disk');
    options.refreshError();
    options.requestRender();
    return true;
  }

  options.setError(null);
  options.setStatus(`✓ Config: ${cfgKey} = ${String(value)}`);
  options.requestRender();
  const timeout = options.setTimeoutFn ?? setTimeout;
  timeout(() => {
    options.setStatus(options.currentModel());
    options.requestRender();
  }, 3000);
  return true;
}
