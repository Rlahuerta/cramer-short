import { afterEach, beforeEach, describe, expect, it, mock } from 'bun:test';

type LocatorCall = {
  selector?: string;
  role?: string;
  options?: unknown;
  action?: string;
  value?: unknown;
};

const locatorCalls: LocatorCall[] = [];
const pageCalls: Array<{ action: string; value?: unknown }> = [];
const browserCloseMock = mock(async () => {});
const launchMock = mock(async () => fakeBrowser);

let fakePage: any;
let fakeNewPage: any;
let fakeBrowser: any;

function makeLocator(seed: LocatorCall = {}) {
  return {
    click: mock(async () => { locatorCalls.push({ ...seed, action: 'click' }); }),
    fill: mock(async (text: string) => { locatorCalls.push({ ...seed, action: 'fill', value: text }); }),
    hover: mock(async () => { locatorCalls.push({ ...seed, action: 'hover' }); }),
    nth: mock((nth: number) => makeLocator({ ...seed, value: nth })),
    ariaSnapshot: mock(async () => fakePage.snapshot),
  };
}

function makePage(title: string) {
  return {
    currentUrl: 'about:blank',
    currentTitle: title,
    snapshot: [
      '- navigation [ref=e1]:',
      '  - link "Investors" [ref=e2]',
      '  - button "Download" [ref=e3] [nth=1]',
      '  - paragraph: This is deliberately long snapshot content for truncation checks.',
    ].join('\n'),
    goto: mock(async function (this: any, url: string) {
      this.currentUrl = url;
      pageCalls.push({ action: 'goto', value: url });
    }),
    url: mock(function (this: any) { return this.currentUrl; }),
    title: mock(async function (this: any) { return this.currentTitle; }),
    context: mock(() => ({ newPage: mock(async () => fakeNewPage) })),
    waitForLoadState: mock(async () => {}),
    _snapshotForAI: mock(async function (this: any) { return { full: this.snapshot }; }),
    locator: mock((selector: string) => makeLocator({ selector })),
    getByRole: mock((role: string, options: unknown) => makeLocator({ role, options })),
    keyboard: {
      press: mock(async (key: string) => { pageCalls.push({ action: 'press', value: key }); }),
    },
    mouse: {
      wheel: mock(async (_x: number, y: number) => { pageCalls.push({ action: 'wheel', value: y }); }),
    },
    waitForTimeout: mock(async (ms: number) => { pageCalls.push({ action: 'wait', value: ms }); }),
    evaluate: mock(async () => 'Main visible text'),
  };
}

const { _setBrowserLauncherForTest, browserTool } =
  await import(`./browser.js?t=${Date.now()}`) as typeof import('./browser.js');

function parseToolResult(result: string): Record<string, unknown> {
  return (JSON.parse(result) as { data: Record<string, unknown> }).data;
}

beforeEach(() => {
  locatorCalls.length = 0;
  pageCalls.length = 0;
  browserCloseMock.mockClear();
  launchMock.mockClear();
  fakePage = makePage('First page');
  fakeNewPage = makePage('Second page');
  fakeBrowser = {
    newContext: mock(async () => ({ newPage: mock(async () => fakePage) })),
    close: browserCloseMock,
  };
  _setBrowserLauncherForTest(launchMock as any);
});

afterEach(async () => {
  await browserTool.invoke({ action: 'close' });
  _setBrowserLauncherForTest();
});

describe('browser tool with mocked Playwright', () => {
  it('validates required URLs without launching a browser', async () => {
    const result = parseToolResult(await browserTool.invoke({ action: 'navigate' }) as string);

    expect(result.error).toBe('url is required for navigate action');
    expect(launchMock).not.toHaveBeenCalled();
  });

  it('navigates and returns page metadata without page content', async () => {
    const result = parseToolResult(await browserTool.invoke({
      action: 'navigate',
      url: 'https://example.com/investors',
    }) as string);

    expect(launchMock).toHaveBeenCalledWith({ headless: false });
    expect(result).toMatchObject({
      ok: true,
      url: 'https://example.com/investors',
      title: 'First page',
    });
    expect(result).not.toHaveProperty('snapshot');
  });

  it('snapshots, parses refs, and uses stored refs for actions', async () => {
    await browserTool.invoke({ action: 'navigate', url: 'https://example.com' });

    const snapshot = parseToolResult(await browserTool.invoke({ action: 'snapshot', maxChars: 80 }) as string);
    expect(snapshot.truncated).toBe(true);
    expect(snapshot.refCount).toBe(3);
    expect(snapshot.refs).toMatchObject({
      e2: { role: 'link', name: 'Investors' },
      e3: { role: 'button', name: 'Download', nth: 1 },
    });

    const click = parseToolResult(await browserTool.invoke({
      action: 'act',
      request: { kind: 'click', ref: 'e2' },
    }) as string);
    expect(click).toMatchObject({ ok: true, clicked: 'e2' });
    expect(locatorCalls).toContainEqual({
      role: 'link',
      options: { name: 'Investors', exact: true },
      action: 'click',
    });
  });

  it('supports read, open, and close actions with mocked pages', async () => {
    const open = parseToolResult(await browserTool.invoke({
      action: 'open',
      url: 'https://example.com/new-tab',
    }) as string);
    expect(open).toMatchObject({
      ok: true,
      url: 'https://example.com/new-tab',
      title: 'Second page',
    });

    const read = parseToolResult(await browserTool.invoke({ action: 'read' }) as string);
    expect(read).toMatchObject({
      url: 'https://example.com/new-tab',
      title: 'Second page',
      content: 'Main visible text',
    });

    const close = parseToolResult(await browserTool.invoke({ action: 'close' }) as string);
    expect(close).toEqual({ ok: true, message: 'Browser closed' });
    expect(browserCloseMock).toHaveBeenCalledTimes(1);
  });

  it('falls back to aria-ref selectors when acting on refs absent from the latest snapshot', async () => {
    await browserTool.invoke({ action: 'navigate', url: 'https://example.com' });
    await browserTool.invoke({ action: 'snapshot' });

    const typed = parseToolResult(await browserTool.invoke({
      action: 'act',
      request: { kind: 'type', ref: 'e99', text: 'query' },
    }) as string);

    expect(typed).toMatchObject({ ok: true, ref: 'e99', typed: 'query' });
    expect(locatorCalls).toContainEqual({
      selector: 'aria-ref=e99',
      action: 'fill',
      value: 'query',
    });
  });
});
