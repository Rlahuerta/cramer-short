import { DynamicStructuredTool } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import { chromium } from 'playwright';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { logger } from '../../utils/logger.js';

type BrowserRef = { role: string; name?: string; nth?: number };
type BrowserLaunchOptions = { headless: boolean };
type LocatorLike = {
  click(options: { timeout: number }): Promise<void>;
  fill(text: string, options: { timeout: number }): Promise<void>;
  hover(options: { timeout: number }): Promise<void>;
  nth(nth: number): LocatorLike;
  ariaSnapshot(): Promise<string>;
};
type BrowserContextLike = {
  newPage(): Promise<PageLike>;
};
type PageLike = {
  goto(url: string, options: { timeout: number; waitUntil: 'networkidle' }): Promise<unknown>;
  url(): string;
  title(): Promise<string>;
  context(): BrowserContextLike;
  close(): Promise<void>;
  waitForLoadState(state: 'networkidle', options: { timeout: number }): Promise<void>;
  locator(selector: string): LocatorLike;
  getByRole(role: string, options: { name?: string | RegExp; exact?: boolean }): LocatorLike;
  keyboard: {
    press(key: string): Promise<void>;
  };
  mouse: {
    wheel(x: number, y: number): Promise<void>;
  };
  waitForTimeout(ms: number): Promise<void>;
  evaluate(pageFunction: () => unknown): Promise<unknown>;
  _snapshotForAI?: (opts: { timeout: number; track: string }) => Promise<SnapshotForAIResult>;
};
type BrowserLike = {
  newContext(): Promise<BrowserContextLike>;
  close(): Promise<void>;
};
type BrowserLauncher = (options: BrowserLaunchOptions) => Promise<BrowserLike>;
type BrowserRuntimeOptions = { launchBrowser?: BrowserLauncher };

const actRequestSchema = z.object({
  kind: z.enum(['click', 'type', 'press', 'hover', 'scroll', 'wait']).describe('The type of interaction'),
  ref: z.string().max(128).optional().describe('Element ref from snapshot (e.g., e12)'),
  text: z.string().max(10_000).optional().describe('Text for type action'),
  key: z.string().max(128).optional().describe('Key for press action (e.g., Enter, Tab)'),
  direction: z.enum(['up', 'down']).optional().describe('Scroll direction'),
  timeMs: z.number().optional().describe('Wait time in milliseconds'),
});
type BrowserActRequest = z.infer<typeof actRequestSchema>;

interface BrowserNavigateResult {
  ok: true;
  url: string;
  title: string;
  hint: string;
}

type BrowserOpenResult = BrowserNavigateResult;

interface BrowserSnapshotResult {
  url: string;
  title: string;
  snapshot: string;
  truncated: boolean;
  refCount: number;
  refs: Record<string, BrowserRef>;
  partial?: true;
  loadWarning?: string;
  hint: string;
}

type BrowserActResult =
  | { ok: true; clicked: string; loadWarning?: string; hint: string }
  | { ok: true; ref: string; typed: string }
  | { ok: true; pressed: string; loadWarning?: string }
  | { ok: true; hovered: string }
  | { ok: true; scrolled: 'up' | 'down' }
  | { ok: true; waited: number };

interface BrowserReadResult {
  url: string;
  title: string;
  content: string;
  loadWarning?: string;
}

export interface BrowserRuntime {
  navigate(url: string, signal?: AbortSignal): Promise<BrowserNavigateResult>;
  open(url: string, signal?: AbortSignal): Promise<BrowserOpenResult>;
  snapshot(maxChars?: number, signal?: AbortSignal): Promise<BrowserSnapshotResult>;
  act(request: BrowserActRequest, signal?: AbortSignal): Promise<BrowserActResult>;
  read(signal?: AbortSignal): Promise<BrowserReadResult>;
  close(): Promise<void>;
}

function defaultBrowserLauncher(): BrowserLauncher {
  return (options) => chromium.launch(options);
}

function createAbortError(message: string = 'Browser action aborted'): Error {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw createAbortError();
  }
}

function isPlaywrightTimeoutError(error: unknown): error is Error {
  return error instanceof Error && error.name === 'TimeoutError';
}

async function withAbort<T>(operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  throwIfAborted(signal);
  if (!signal) {
    return operation();
  }

  return await new Promise<T>((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      reject(createAbortError());
    };

    signal.addEventListener('abort', onAbort, { once: true });
    operation().then(
      (value) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (error) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', onAbort);
        reject(error);
      },
    );
  });
}

async function waitForNetworkIdleOrMarkPartial(
  page: PageLike,
  timeout: number,
  signal: AbortSignal | undefined,
  markPartial: () => void,
): Promise<void> {
  try {
    await withAbort(() => page.waitForLoadState('networkidle', { timeout }), signal);
  } catch (error) {
    if (isPlaywrightTimeoutError(error)) {
      markPartial();
      return;
    }
    throw error;
  }
}

// Type for Playwright's _snapshotForAI result
interface SnapshotForAIResult {
  full?: string;
}

/** Owns browser/page/ref state for a browser tool instance. */
class PlaywrightBrowserRuntime implements BrowserRuntime {
  private browser: BrowserLike | null = null;
  private page: PageLike | null = null;
  private launchBrowser: BrowserLauncher;
  private currentRefs: Map<string, BrowserRef> = new Map();

  constructor(options: BrowserRuntimeOptions = {}) {
    this.launchBrowser = options.launchBrowser ?? defaultBrowserLauncher();
  }

  setLauncher(launcher?: BrowserLauncher): void {
    this.launchBrowser = launcher ?? defaultBrowserLauncher();
  }

  async navigate(url: string, signal?: AbortSignal): Promise<BrowserNavigateResult> {
    const p = await this.ensurePage(signal);
    await withAbort(() => p.goto(url, { timeout: 30000, waitUntil: 'networkidle' }), signal);
    return {
      ok: true,
      url: p.url(),
      title: await p.title(),
      hint: 'Page loaded. Call snapshot to see page structure and find elements to interact with.',
    };
  }

  async open(url: string, signal?: AbortSignal): Promise<BrowserOpenResult> {
    const currentPage = await this.ensurePage(signal);
    const context = currentPage.context();
    const newPage = await withAbort(() => context.newPage(), signal);
    await withAbort(() => newPage.goto(url, { timeout: 30000, waitUntil: 'networkidle' }), signal);
    await this.replaceActivePage(newPage);
    return {
      ok: true,
      url: newPage.url(),
      title: await newPage.title(),
      hint: 'New tab opened. Call snapshot to see page structure and find elements to interact with.',
    };
  }

  async snapshot(maxChars?: number, signal?: AbortSignal): Promise<BrowserSnapshotResult> {
    const p = await this.ensurePage(signal);
    let partialLoad = false;
    await waitForNetworkIdleOrMarkPartial(p, 5000, signal, () => { partialLoad = true; });

    const { snapshot, truncated } = await this.takeSnapshot(p, maxChars, signal);

    return {
      url: p.url(),
      title: await p.title(),
      snapshot,
      truncated,
      refCount: this.refCount,
      refs: this.refsAsRecord(),
      ...(partialLoad ? { partial: true, loadWarning: 'Page may not be fully loaded (networkidle timeout). Content could be incomplete — consider retrying or using the read action.' } : {}),
      hint: 'Use act with kind="click" and ref="eN" to click elements. Or navigate directly to a /url visible in the snapshot.',
    };
  }

  async act(request: BrowserActRequest, signal?: AbortSignal): Promise<BrowserActResult> {
    const p = await this.ensurePage(signal);
    const { kind, ref, text, key, direction, timeMs } = request;

    switch (kind) {
      case 'click': {
        if (!ref) {
          throw new Error('ref is required for click');
        }
        const locator = this.resolveRefToLocator(p, ref);
        await withAbort(() => locator.click({ timeout: 8000 }), signal);
        let clickLoadPartial = false;
        await waitForNetworkIdleOrMarkPartial(p, 10000, signal, () => { clickLoadPartial = true; });
        return {
          ok: true,
          clicked: ref,
          ...(clickLoadPartial ? { loadWarning: 'Post-click navigation may not be complete (networkidle timeout). Call snapshot to see current state.' } : {}),
          hint: 'Click successful. Call snapshot to see the updated page.',
        };
      }

      case 'type': {
        if (!ref) {
          throw new Error('ref is required for type');
        }
        if (!text) {
          throw new Error('text is required for type');
        }
        const locator = this.resolveRefToLocator(p, ref);
        await withAbort(() => locator.fill(text, { timeout: 8000 }), signal);
        return { ok: true, ref, typed: text };
      }

      case 'press': {
        if (!key) {
          throw new Error('key is required for press');
        }
        await withAbort(() => p.keyboard.press(key), signal);
        let pressLoadPartial = false;
        await waitForNetworkIdleOrMarkPartial(p, 5000, signal, () => { pressLoadPartial = true; });
        return { ok: true, pressed: key, ...(pressLoadPartial ? { loadWarning: 'Post-keypress navigation may not be complete. Call snapshot to see current state.' } : {}) };
      }

      case 'hover': {
        if (!ref) {
          throw new Error('ref is required for hover');
        }
        const locator = this.resolveRefToLocator(p, ref);
        await withAbort(() => locator.hover({ timeout: 8000 }), signal);
        return { ok: true, hovered: ref };
      }

      case 'scroll': {
        const scrollDirection = direction ?? 'down';
        const amount = scrollDirection === 'down' ? 500 : -500;
        await withAbort(() => p.mouse.wheel(0, amount), signal);
        await withAbort(() => p.waitForTimeout(500), signal);
        return { ok: true, scrolled: scrollDirection };
      }

      case 'wait': {
        const waitTime = Math.min(timeMs ?? 2000, 10000);
        await withAbort(() => p.waitForTimeout(waitTime), signal);
        return { ok: true, waited: waitTime };
      }
    }
  }

  async read(signal?: AbortSignal): Promise<BrowserReadResult> {
    const p = await this.ensurePage(signal);
    let readPartial = false;
    await waitForNetworkIdleOrMarkPartial(p, 5000, signal, () => { readPartial = true; });

    const content = await withAbort(
      () => p.evaluate(() => {
        const main = document.querySelector('main, article, [role="main"], .content, #content') as HTMLElement | null;
        return (main || document.body).innerText;
      }),
      signal,
    );
    return {
      url: p.url(),
      title: await p.title(),
      content: String(content),
      ...(readPartial ? { loadWarning: 'Page may not be fully loaded (networkidle timeout). Content could be incomplete.' } : {}),
    };
  }

  private async ensurePage(signal?: AbortSignal): Promise<PageLike> {
    throwIfAborted(signal);
    if (!this.browser) {
      this.browser = await withAbort(() => this.launchBrowser({ headless: false }), signal);
    }
    const activeBrowser = this.browser;
    if (!this.page) {
      const context = await withAbort(() => activeBrowser.newContext(), signal);
      this.page = await withAbort(() => context.newPage(), signal);
    }
    return this.page;
  }

  async close(): Promise<void> {
    this.currentRefs.clear();
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
    this.page = null;
  }

  private async replaceActivePage(nextPage: PageLike): Promise<void> {
    const previousPage = this.page;
    this.page = nextPage;
    this.currentRefs.clear();

    if (!previousPage || previousPage === nextPage) {
      return;
    }

    try {
      await previousPage.close();
    } catch (error) {
      logger.warn('[Browser (Playwright)] failed to close replaced page', error);
    }
  }

  private resolveRefToLocator(p: PageLike, ref: string): LocatorLike {
    const refData = this.currentRefs.get(ref);

    if (!refData) {
      return p.locator(`aria-ref=${ref}`);
    }

    const options: { name?: string | RegExp; exact?: boolean } = {};
    if (refData.name) {
      options.name = refData.name;
      options.exact = true;
    }

    let locator = p.getByRole(refData.role, options);

    if (typeof refData.nth === 'number' && refData.nth > 0) {
      locator = locator.nth(refData.nth);
    }

    return locator;
  }

  private async takeSnapshot(
    p: PageLike,
    maxChars?: number,
    signal?: AbortSignal,
  ): Promise<{ snapshot: string; truncated: boolean }> {
    let snapshot: string;

    if (p._snapshotForAI) {
      const snapshotForAI = p._snapshotForAI.bind(p);
      const result = await withAbort(
        () => snapshotForAI({ timeout: 10000, track: 'response' }),
        signal,
      );
      snapshot = String(result?.full ?? '');
    } else {
      snapshot = await withAbort(() => p.locator(':root').ariaSnapshot(), signal);
    }

    this.currentRefs = parseRefsFromSnapshot(snapshot);

    let truncated = false;
    const limit = maxChars ?? 50000;
    if (snapshot.length > limit) {
      snapshot = `${snapshot.slice(0, limit)}\n\n[...TRUNCATED - page too large, use read action for full text]`;
      truncated = true;
    }

    return { snapshot, truncated };
  }

  private get refCount(): number {
    return this.currentRefs.size;
  }

  private refsAsRecord(): Record<string, BrowserRef> {
    return Object.fromEntries(this.currentRefs);
  }
}

export function createBrowserRuntime(options: BrowserRuntimeOptions = {}): BrowserRuntime {
  return new PlaywrightBrowserRuntime(options);
}

const productionBrowserRuntime = new PlaywrightBrowserRuntime();

/**
 * Rich description for the browser tool.
 * Used in the system prompt to guide the LLM on when and how to use this tool.
 */
export const BROWSER_DESCRIPTION = `
Control a web browser to navigate websites and extract information.

**NOTE: For simply reading a web page's content, prefer web_fetch which returns content directly in a single call. Use browser only for interactive tasks requiring JavaScript rendering, clicking, or form filling.**

## When to Use

- Accessing dynamic/JavaScript-rendered content that requires a real browser
- Multi-step web navigation (click links, fill search boxes)
- Interacting with SPAs or pages that require JavaScript to load content
- When web_fetch fails or returns incomplete content due to JS-dependent rendering

## When NOT to Use

- Reading static web pages or articles (use **web_fetch** instead - it is faster and returns content in a single call)
- Simple queries that web_search can already answer
- Structured financial data (use get_financials instead)
- SEC filings content (use read_filings instead)
- General knowledge questions

## CRITICAL: Navigate Returns NO Content

The \`navigate\` action only loads the page - it does NOT return page content.
You MUST call \`snapshot\` after navigate to see what's on the page.

## CRITICAL: Use Visible URLs - Do NOT Guess

When the snapshot shows a link with a URL (e.g., \`/url: https://...\`):
1. **Option A**: Click the link using its ref (e.g., act with kind="click", ref="e22")
2. **Option B**: Navigate directly to the URL shown in the snapshot

**NEVER make up or guess URLs based on common patterns**. If you need to reach a page:
1. Take a snapshot
2. Find the link in the snapshot
3. Either click it OR navigate to its visible /url value

Bad: Guessing https://company.com/news-events/press-releases
Good: Using the /url value you SEE in the snapshot

## Available Actions

- **navigate** - Navigate to a URL in the current tab (returns only url/title, no content)
- **open** - Open a URL in a NEW tab (use when starting a fresh browsing session)
- **snapshot** - See page structure with clickable refs (e.g., e1, e2, e3)
- **act** - Interact with elements using refs (click, type, press, scroll)
- **read** - Extract full text content from the page
- **close** - Free browser resources when done

## Workflow (MUST FOLLOW)

1. **navigate** or **open** - Load a URL (returns only url/title, no content)
2. **snapshot** - See page structure with clickable refs (e.g., e1, e2, e3)
3. **act** - Interact with elements using refs:
   - kind="click", ref="e5" - Click a link/button
   - kind="type", ref="e3", text="search query" - Type in an input
   - kind="press", key="Enter" - Press a key
   - kind="scroll", direction="down" - Scroll the page
4. **snapshot** again - See updated page after interaction
5. **Repeat steps 3-4** until you find the content you need
6. **read** - Extract full text content from the page
7. **close** - Free browser resources when done

## Snapshot Format

The snapshot returns an AI-optimized accessibility tree with refs:
- navigation [ref=e1]:
  - link "Home" [ref=e2]
  - link "Investors" [ref=e3]
  - link "Press Releases" [ref=e4]
- main:
  - heading "Welcome to Acme Corp" [ref=e5]
  - paragraph: Latest news and updates
  - link "Q4 2024 Earnings" [ref=e6]
  - link "View All Press Releases" [ref=e7]

## Act Action Examples

To click a link with ref=e4:
  action="act", request with kind="click" and ref="e4"

To type in a search box with ref=e10:
  action="act", request with kind="type", ref="e10", text="earnings"

To press Enter:
  action="act", request with kind="press" and key="Enter"

## Example: Finding a Press Release

1. navigate to https://investors.company.com
2. snapshot - see links like "Press Releases" [ref=e4]
3. act with kind="click", ref="e4" - click Press Releases link
4. snapshot - see list of press releases
5. act with kind="click", ref="e12" - click specific press release
6. read - extract the full press release text

## Usage Notes

- Always call snapshot after navigate/open - they return only url/title, no content
- Use **open** to start a fresh tab; use **navigate** to go to a URL within the current tab
- After clicking, always call snapshot again to see the new page
- The browser persists across calls - no need to re-navigate to the same URL
- Use read for bulk text extraction once you've navigated to the right page
- Close the browser when done to free system resources
`.trim();

/**
 * Close the browser and reset production runtime state.
 * Exported so process-exit hooks (registered in src/index.tsx) can release
 * the Chromium subprocess and its file descriptors when the agent shuts down.
 */
export async function closeBrowser(): Promise<void> {
  await productionBrowserRuntime.close();
}

/** @internal Test-only: inject a Playwright launcher for the production browser runtime. */
export function _setBrowserLauncherForTest(launcher?: BrowserLauncher): void {
  productionBrowserRuntime.setLauncher(launcher);
}

/**
 * Parse refs from the AI snapshot format.
 * Extracts [ref=eN] patterns and builds a ref map.
 */
function parseRefsFromSnapshot(snapshot: string): Map<string, BrowserRef> {
  const refs = new Map<string, BrowserRef>();
  const lines = snapshot.split('\n');
  
  for (const line of lines) {
    // Match patterns like: - button "Click me" [ref=e12]
    const refMatch = line.match(/\[ref=(e\d+)\]/);
    if (!refMatch) continue;
    
    const ref = refMatch[1];
    
    // Extract role (first word after "- ")
    const roleMatch = line.match(/^\s*-\s*(\w+)/);
    const role = roleMatch ? roleMatch[1] : 'generic';
    
    // Extract name (text in quotes)
    const nameMatch = line.match(/"([^"]+)"/);
    const name = nameMatch ? nameMatch[1] : undefined;
    
    // Extract nth if present
    const nthMatch = line.match(/\[nth=(\d+)\]/);
    const nth = nthMatch ? parseInt(nthMatch[1], 10) : undefined;
    
    refs.set(ref, { role, name, nth });
  }
  
  return refs;
}

function getActRequestValidationError(request: BrowserActRequest): string | undefined {
  switch (request.kind) {
    case 'click':
      return request.ref ? undefined : 'ref is required for click';
    case 'type':
      if (!request.ref) {
        return 'ref is required for type';
      }
      return request.text ? undefined : 'text is required for type';
    case 'press':
      return request.key ? undefined : 'key is required for press';
    case 'hover':
      return request.ref ? undefined : 'ref is required for hover';
    case 'scroll':
    case 'wait':
      return undefined;
  }
}

/** Exposes browser automation for page navigation and content extraction. */
export function createBrowserTool(options: { runtime?: BrowserRuntime } = {}): DynamicStructuredTool {
  const runtime = options.runtime ?? productionBrowserRuntime;

  return new DynamicStructuredTool({
    name: 'browser',
    description: 'Navigate websites, read content, and interact with pages. Use for accessing company websites, earnings reports, and dynamic content.',
    schema: z.object({
      action: z.enum(['navigate', 'open', 'snapshot', 'act', 'read', 'close']).describe('The browser action to perform'),
      url: z.string().max(4096).optional().describe('URL for navigate action'),
      maxChars: z.number().optional().describe('Max characters for snapshot (default 50000)'),
      request: actRequestSchema.optional().describe('Request object for act action'),
    }),
    func: async ({ action, url, maxChars, request }, _runManager, config?: RunnableConfig) => {
      const signal = config?.signal;
      let shouldCloseBrowser = action === 'close';

      try {
        throwIfAborted(signal);
        switch (action) {
          case 'navigate':
            if (!url) {
              return formatToolResult({ error: 'url is required for navigate action' });
            }
            return formatToolResult(await runtime.navigate(url, signal));

          case 'open':
            if (!url) {
              return formatToolResult({ error: 'url is required for open action' });
            }
            return formatToolResult(await runtime.open(url, signal));

          case 'snapshot':
            return formatToolResult(await runtime.snapshot(maxChars, signal));

          case 'act': {
            if (!request) {
              return formatToolResult({ error: 'request is required for act action' });
            }
            const validationError = getActRequestValidationError(request);
            if (validationError) {
              return formatToolResult({ error: validationError });
            }
            return formatToolResult(await runtime.act(request, signal));
          }

          case 'read':
            return formatToolResult(await runtime.read(signal));

          case 'close':
            return formatToolResult({ ok: true, message: 'Browser closed' });
        }
      } catch (error) {
        shouldCloseBrowser = true;
        if (!(error instanceof Error)) {
          throw error;
        }
        const message = error.message;
        logger.error(`[Browser (Playwright)] error: ${message}`);
        return formatToolResult({ error: `[Browser (Playwright)] ${message}` });
      } finally {
        if (shouldCloseBrowser) {
          await runtime.close().catch((error) => {
            logger.error('[Browser (Playwright)] failed to close browser', error);
          });
        }
      }
    },
  });
}

export const browserTool = createBrowserTool();
