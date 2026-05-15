#!/usr/bin/env bun
import { config } from 'dotenv';
import { runCli } from './cli.js';
import { runForecastLabCommand } from './cli-forecast-lab.js';
import { runReplayLabelCommand } from './cli-replay-label.js';
import { runScheduleCommand } from './cli-schedule.js';
import { closeBrowser } from './tools/browser/index.js';
import { routeCommand } from './index-routing.js';
import { logError } from './utils/error-logger.js';

// Load environment variables
config({ quiet: true });

// Ensure the headless Chromium spawned by the browser tool is released on
// shutdown. Without this, repeated runs leak chromium processes and FDs
// because the tool only closes the browser when the LLM calls `close`.
let shuttingDown = false;
async function shutdown(code = 0): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  try {
    await closeBrowser();
  } catch (err) {
    // best-effort cleanup; log but don't block shutdown
    logError({ type: 'system_error', message: `Browser cleanup failed: ${err}`, context: 'shutdown' });
  }
  process.exit(code);
}
process.on('beforeExit', () => {
  void closeBrowser().catch((err) => {
    logError({ type: 'system_error', message: `Browser cleanup failed: ${err}`, context: 'beforeExit' });
  });
});
process.on('SIGINT', () => void shutdown(130));
process.on('SIGTERM', () => void shutdown(143));

// Detect headless subcommands before launching the TUI
await routeCommand(process.argv, {
  schedule: async (args) => {
    await runScheduleCommand(args);
    await closeBrowser().catch((err) => {
      logError({ type: 'system_error', message: `Browser cleanup failed: ${err}`, context: 'schedule' });
    });
  },
  lab: async (args) => {
    await runForecastLabCommand(args);
    await closeBrowser().catch((err) => {
      logError({ type: 'system_error', message: `Browser cleanup failed: ${err}`, context: 'lab' });
    });
  },
  replayLabel: async (args) => {
    await runReplayLabelCommand(args);
    await closeBrowser().catch((err) => {
      logError({ type: 'system_error', message: `Browser cleanup failed: ${err}`, context: 'replayLabel' });
    });
  },
  cli: async () => { await runCli(); },
});
