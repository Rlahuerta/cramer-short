/**
 * Pure routing dispatch for src/index.tsx.
 * Extracted to make argv-based subcommand routing unit-testable.
 */

export type IndexHandlers = {
  schedule: (args: string[]) => Promise<void>;
  lab: (args: string[]) => Promise<void>;
  replayLabel: (args: string[]) => Promise<void>;
  cli: () => Promise<void>;
};

export async function routeCommand(argv: string[], handlers: IndexHandlers): Promise<void> {
  const sub = argv[2];
  if (sub === 'schedule') {
    await handlers.schedule(argv.slice(3));
  } else if (sub === 'lab') {
    await handlers.lab(argv.slice(3));
  } else if (sub === 'replay-label') {
    await handlers.replayLabel(argv.slice(3));
  } else {
    await handlers.cli();
  }
}
