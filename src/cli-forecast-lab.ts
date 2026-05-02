import { listForecastLabProfiles } from './experiments/forecast-lab/profiles.js';
import { runForecastLab } from './experiments/forecast-lab/runner.js';
import type { ForecastLabRunOptions, ForecastLabRunResult } from './experiments/forecast-lab/runner.js';

type ForecastLabCliOptions = {
  readonly log?: (message: string) => void;
  readonly error?: (message: string) => void;
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

export async function runForecastLabCommand(argv: string[], options: ForecastLabCliOptions = {}): Promise<void> {
  const log = options.log ?? console.log;
  const error = options.error ?? console.error;
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
      const result = await runLab(parseRunArgs(argv));
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
