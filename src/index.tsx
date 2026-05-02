#!/usr/bin/env bun
import { config } from 'dotenv';
import { runCli } from './cli.js';
import { runForecastLabCommand } from './cli-forecast-lab.js';
import { runScheduleCommand } from './cli-schedule.js';
import { closeBrowser } from './tools/browser/index.js';

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
  } catch {
    // best-effort cleanup
  }
  process.exit(code);
}
process.on('beforeExit', () => {
  void closeBrowser().catch(() => {});
});
process.on('SIGINT', () => void shutdown(130));
process.on('SIGTERM', () => void shutdown(143));

// Detect headless subcommands before launching the TUI
const subCommand = process.argv[2];
if (subCommand === 'schedule') {
  await runScheduleCommand(process.argv.slice(3));
  await closeBrowser().catch(() => {});
} else if (subCommand === 'lab') {
  await runForecastLabCommand(process.argv.slice(3));
  await closeBrowser().catch(() => {});
} else {
  await runCli();
}
