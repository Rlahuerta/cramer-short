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
      '',
      'Forecast-lab V1 requires --dry-run or --skip-mutation; real mutation is not supported yet.',
    ].join('\n'),
  );
}

function parseRunArgs(argv: string[]): ForecastLabRunOptions {
  const profileId = argv[1];
  const flags = new Set(argv.slice(2));

  if (!profileId) {
    throw new Error('Missing forecast-lab profile id.');
  }

  return {
    profileId,
    dryRun: flags.has('--dry-run'),
    skipMutation: flags.has('--skip-mutation'),
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
  log('Previous parameters (baseline gate):');
  for (const line of formatCommandParameters(profile.baselineCommands)) {
    log(line);
  }

  log('');
  log('New parameters (candidate gate):');
  for (const line of formatCommandParameters(profile.candidateCommands)) {
    log(line);
  }

  if (profile.baselineCommands === profile.candidateCommands) {
    log('');
    log('Note: in V1 dry-run mode there is no source mutation, so baseline and candidate parameters are typically identical.');
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
      log(`Running forecast-lab profile "${runOptions.profileId}"...`);
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
