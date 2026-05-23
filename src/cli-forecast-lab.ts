import { getForecastLabProfile, listForecastLabProfiles } from './experiments/forecast-lab/profiles.js';
import type { ForecastLabCommand } from './experiments/forecast-lab/profiles.js';
import { runForecastLab } from './experiments/forecast-lab/runner.js';
import type { ForecastLabRunOptions, ForecastLabRunResult } from './experiments/forecast-lab/runner.js';

type ForecastLabCliOptions = {
  readonly log?: (message: string) => void;
  readonly error?: (message: string) => void;
  readonly write?: (chunk: string) => void;
  readonly exit?: (code: number) => void;
  readonly runLab?: (options: ForecastLabRunOptions) => Promise<ForecastLabRunResult>;
};

function printUsage(log: (message: string) => void): void {
  log(
    [
      'Usage:',
      '  cramer-short lab list',
      '  cramer-short lab run <profileId> --dry-run',
      '  cramer-short lab run <profileId> --skip-mutation',
      '  cramer-short lab run <profileId> --mutation structured [--mutator <id>] [--keep-worktree]',
      '',
      'Real mutation requires an explicit --mutation structured flag. Use --dry-run or --skip-mutation for the no-mutation paths.',
    ].join('\n'),
  );
}

const FORECAST_LAB_MUTATION_MODES = new Set(['structured']);

function parseRunArgs(argv: string[]): ForecastLabRunOptions {
  const profileId = argv[1];
  const args = argv.slice(2);

  if (!profileId) {
    throw new Error('Missing forecast-lab profile id.');
  }

  let dryRun = false;
  let skipMutation = false;
  let mutationMode: ForecastLabRunOptions['mutationMode'];
  let keepWorktree = false;
  let mutator: string | undefined;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === '--dry-run') {
      dryRun = true;
      continue;
    }

    if (arg === '--skip-mutation') {
      skipMutation = true;
      continue;
    }

    if (arg === '--keep-worktree') {
      keepWorktree = true;
      continue;
    }

    if (arg === '--mutation') {
      const value = args[index + 1];
      if (!value || value.startsWith('--')) {
        throw new Error('Missing value for --mutation. Expected: structured');
      }
      if (!FORECAST_LAB_MUTATION_MODES.has(value)) {
        throw new Error(`Unknown forecast-lab mutation mode: "${value}"`);
      }
      mutationMode = value as ForecastLabRunOptions['mutationMode'];
      index += 1;
      continue;
    }

    if (arg === '--mutator') {
      const value = args[index + 1];
      if (!value || value.startsWith('--')) {
        throw new Error('Missing value for --mutator.');
      }
      mutator = value;
      index += 1;
      continue;
    }

    throw new Error(`Unknown forecast-lab flag: "${arg}"`);
  }

  if (dryRun && skipMutation) {
    throw new Error('Conflicting forecast-lab flags: --dry-run and --skip-mutation cannot be used together.');
  }

  if (dryRun || skipMutation) {
    if (mutationMode !== undefined) {
      throw new Error('Conflicting forecast-lab flags: --mutation cannot be combined with --dry-run or --skip-mutation.');
    }
    if (keepWorktree) {
      throw new Error('Conflicting forecast-lab flags: --keep-worktree requires --mutation structured.');
    }
    if (mutator !== undefined) {
      throw new Error('Conflicting forecast-lab flags: --mutator requires --mutation structured.');
    }
  } else if (keepWorktree || mutator !== undefined) {
    if (mutationMode !== 'structured') {
      throw new Error('Conflicting forecast-lab flags: --keep-worktree and --mutator require --mutation structured.');
    }
  } else if (mutationMode === undefined) {
    throw new Error('Real forecast-lab mutation requires an explicit flag: --mutation structured.');
  }

  return {
    profileId,
    dryRun,
    skipMutation,
    mutationMode,
    keepWorktree,
    mutator,
  };
}

function formatCommandParameters(commands: readonly ForecastLabCommand[]): string[] {
  return commands.map((command) => {
    const parts = [`command=${command.command}`];

    if (command.timeoutMs !== undefined) {
      parts.push(`timeoutMs=${command.timeoutMs}`);
    }

    if (command.env && Object.keys(command.env).length > 0) {
      parts.push(`env=${JSON.stringify(command.env)}`);
    }

    return `  - ${command.id}: ${parts.join(', ')}`;
  });
}

function formatMutationParameterLabel(filePath: string, parameterId: string): string {
  const fileName = filePath.split('/').at(-1) ?? filePath;
  return `${fileName}: ${parameterId}`;
}

function formatMutationParameterSections(result: ForecastLabRunResult): {
  readonly previous: string[];
  readonly next: string[];
} | undefined {
  const replayPayload = result.manifest.mutationReplayPayload;
  if (!replayPayload) {
    return undefined;
  }

  return {
    previous: replayPayload.edits.map((edit) => `  - ${formatMutationParameterLabel(edit.filePath, edit.parameterId)} = ${edit.beforeValue}`),
    next: replayPayload.edits.map((edit) => `  - ${formatMutationParameterLabel(edit.filePath, edit.parameterId)} = ${edit.afterValue}`),
  };
}

function readExitCode(summary: unknown): number | undefined {
  if (!summary || typeof summary !== 'object') {
    return undefined;
  }

  const exitCode = (summary as Record<string, unknown>).exitCode;
  return typeof exitCode === 'number' ? exitCode : undefined;
}

function printRunSummary(log: (message: string) => void, result: ForecastLabRunResult): void {
  const profile = getForecastLabProfile(result.manifest.profileId);
  const baselineExitCode = readExitCode(result.baseline);
  const candidateExitCode = readExitCode(result.candidate);
  const mutationParameters = formatMutationParameterSections(result);

  log('');
  log('Evolution summary:');
  if (baselineExitCode !== undefined) {
    log(`  baseline exitCode: ${baselineExitCode}`);
  }
  if (candidateExitCode !== undefined) {
    log(`  candidate exitCode: ${candidateExitCode}`);
  }

  if (result.decision.metrics.length > 0) {
    log('  compared metrics:');
    for (const metric of result.decision.metrics) {
      log(`    - ${metric.name}: baseline=${metric.baseline}, candidate=${metric.candidate}, delta=${metric.delta}`);
    }
  }

  log('');
  if (mutationParameters) {
    log(`Mutation summary: ${result.manifest.mutationSummary ?? result.manifest.mutationReplayPayload?.specSummary.summary ?? 'n/a'}`);
    if (result.manifest.mutationId) {
      log(`  mutation id: ${result.manifest.mutationId}`);
    }
    log('');
    log('Previous parameters (baseline defaults):');
    for (const line of mutationParameters.previous) {
      log(line);
    }
    log('');
    log('New parameters (candidate mutation):');
    for (const line of mutationParameters.next) {
      log(line);
    }
  } else {
    log('Previous parameters (baseline gate):');
    for (const line of formatCommandParameters(profile.baselineCommands)) {
      log(line);
    }

    log('');
    log('New parameters (candidate gate):');
    for (const line of formatCommandParameters(profile.candidateCommands)) {
      log(line);
    }

    if (result.manifest.candidateWorkspace === undefined && profile.baselineCommands === profile.candidateCommands) {
      log('');
      log('Note: with no candidate workspace, baseline and candidate parameters are typically identical.');
    }
  }
}

export async function runForecastLabCommand(argv: string[], options: ForecastLabCliOptions = {}): Promise<void> {
  const log = options.log ?? console.log;
  const error = options.error ?? console.error;
  const write = options.write ?? ((chunk: string) => process.stdout.write(chunk));
  const exit = options.exit ?? ((code: number) => process.exit(code));
  const [subCmd] = argv;

  if (!subCmd || subCmd === 'help' || subCmd === '--help') {
    printUsage(log);
    return;
  }

  if (subCmd === 'list') {
    log('Forecast-lab profiles:');
    for (const profile of listForecastLabProfiles()) {
      log(`  ${profile.id.padEnd(30)} ${profile.targetSubsystem}`);
    }
    return;
  }

  if (subCmd === 'run') {
    if (!argv[1]) {
      error('Missing forecast-lab profile id.');
      printUsage(log);
      exit(1);
      return;
    }

    try {
      const runLab = options.runLab ?? runForecastLab;
      const runOptions = parseRunArgs(argv);
      log(
        `Running forecast-lab profile "${runOptions.profileId}"` +
          (runOptions.mutationMode ? ` with ${runOptions.mutationMode} mutation...` : '...'),
      );
      const result = await runLab({
        ...runOptions,
        progress: log,
        output: write,
      });
      printRunSummary(log, result);
      log(`forecast-lab ${result.decision.decision}: ${result.decision.reason}`);
      log(`artifacts: ${result.manifest.artifactsPath}`);
    } catch (caught) {
      error(caught instanceof Error ? caught.message : String(caught));
      exit(1);
    }
    return;
  }

  error(`Unknown forecast-lab subcommand: "${subCmd}"`);
  printUsage(log);
  exit(1);
}
