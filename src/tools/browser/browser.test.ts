import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { afterEach, beforeEach, describe, expect, it, mock, setSystemTime } from 'bun:test';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

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
const contextRouteMock = mock(async (_url: string, _handler: unknown) => {});

type FakePageThis = {
  currentUrl: string;
  currentTitle: string;
  snapshot: string;
};

let fakePage: ReturnType<typeof makePage>;
let fakeNewPage: ReturnType<typeof makePage>;
let fakeBrowser: ReturnType<typeof makeBrowser>;
const launchMock = mock(async () => fakeBrowser);

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
    goto: mock(async function (this: FakePageThis, url: string) {
      this.currentUrl = url;
      pageCalls.push({ action: 'goto', value: url });
    }),
    url: mock(function (this: FakePageThis) { return this.currentUrl; }),
    title: mock(async function (this: FakePageThis) { return this.currentTitle; }),
    context: mock(() => ({ newPage: mock(async () => fakeNewPage), route: mock(async () => {}) })),
    close: mock(async () => {}),
    waitForLoadState: mock(async () => {}),
    _snapshotForAI: mock(async function (this: FakePageThis) { return { full: this.snapshot }; }),
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

function makeBrowser(page: ReturnType<typeof makePage>, close = browserCloseMock) {
  return {
    newContext: mock(async (_options: { serviceWorkers: 'block' }) => ({
      newPage: mock(async () => page),
      route: contextRouteMock,
    })),
    close,
  };
}

const { _setBrowserLauncherForTest, browserTool, createBrowserRuntime, createBrowserTool, guardRoute } =
  await import(`./browser.js?t=${nextTestId('module')}`) as typeof import('./browser.js');
const { _setDnsLookupForTest } = await import('../fetch/url-guard.js');

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
  fakeBrowser = makeBrowser(fakePage);
  _setBrowserLauncherForTest(launchMock);
  contextRouteMock.mockClear();
  _setDnsLookupForTest(async () => [{ address: '93.184.216.34' }]);
});

afterEach(async () => {
  await browserTool.invoke({ action: 'close' });
  _setBrowserLauncherForTest();
  _setDnsLookupForTest();
});

describe('browser tool with mocked Playwright', () => {
  it('validates required URLs without launching a browser', async () => {
    const result = parseToolResult(await browserTool.invoke({ action: 'navigate' }) as string);

    expect(result.error).toBe('url is required for navigate action');
    expect(launchMock).not.toHaveBeenCalled();
  });

  it('uses an injected runtime launcher without touching the production singleton launcher', async () => {
    const isolatedPage = makePage('Isolated page');
    const isolatedCloseMock = mock(async () => {});
    const isolatedBrowser = makeBrowser(isolatedPage, isolatedCloseMock);
    const isolatedLaunchMock = mock(async () => isolatedBrowser);
    const isolatedTool = createBrowserTool({
      runtime: createBrowserRuntime({ launchBrowser: isolatedLaunchMock }),
    });

    const result = parseToolResult(await isolatedTool.invoke({
      action: 'navigate',
      url: 'https://example.com/isolated',
    }) as string);

    expect(result).toMatchObject({
      ok: true,
      url: 'https://example.com/isolated',
      title: 'Isolated page',
    });
    expect(isolatedLaunchMock).toHaveBeenCalledWith({
      headless: false,
      // All browser traffic (incl. WebSockets) goes through the local SSRF guard proxy.
      proxy: { server: expect.stringMatching(/^http:\/\/127\.0\.0\.1:\d+$/) },
    });
    expect(launchMock).not.toHaveBeenCalled();

    await isolatedTool.invoke({ action: 'close' });
    expect(isolatedCloseMock).toHaveBeenCalledTimes(1);
    expect(launchMock).not.toHaveBeenCalled();
  });

  it('navigates and returns page metadata without page content', async () => {
    const result = parseToolResult(await browserTool.invoke({
      action: 'navigate',
      url: 'https://example.com/investors',
    }) as string);

    expect(launchMock).toHaveBeenCalledWith({
      headless: false,
      // All browser traffic (incl. WebSockets) goes through the local SSRF guard proxy.
      proxy: { server: expect.stringMatching(/^http:\/\/127\.0\.0\.1:\d+$/) },
    });
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
    expect(fakePage.close).toHaveBeenCalledTimes(1);

    const read = parseToolResult(await browserTool.invoke({ action: 'read' }) as string);
    expect(read).toMatchObject({
      url: 'https://example.com/new-tab',
      title: 'Second page',
    });
    expect(read.content).toContain('Main visible text');

    const close = parseToolResult(await browserTool.invoke({ action: 'close' }) as string);
    expect(close).toEqual({ ok: true, message: 'Browser closed' });
    expect(browserCloseMock).toHaveBeenCalledTimes(1);
  });

  it('wraps page text from read and snapshot in untrusted-content markers (PI-44)', async () => {
    await browserTool.invoke({ action: 'navigate', url: 'https://example.com' });

    const read = parseToolResult(await browserTool.invoke({ action: 'read' }) as string);
    const snapshot = parseToolResult(await browserTool.invoke({ action: 'snapshot' }) as string);

    for (const text of [read.content, snapshot.snapshot] as string[]) {
      expect(text).toContain('<<<EXTERNAL_UNTRUSTED_CONTENT>>>');
      expect(text).toContain('<<<END_EXTERNAL_UNTRUSTED_CONTENT>>>');
    }
    expect(read.content).toContain('Main visible text');
    expect(snapshot.snapshot).toContain('This is deliberately long snapshot content');
  });

  it('closes the browser when a Playwright action throws', async () => {
    fakePage.goto = mock(async () => {
      throw new Error('navigation failed');
    });

    const result = parseToolResult(await browserTool.invoke({
      action: 'navigate',
      url: 'https://example.com/bad',
    }) as string);

    expect(result.error).toBe('[Browser (Playwright)] navigation failed');
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

describe('browser tool SSRF guard', () => {
  it('rejects file:// URL on navigate before launching a browser', async () => {
    const result = parseToolResult(
      await browserTool.invoke({ action: 'navigate', url: 'file:///etc/passwd' }) as string,
    );
    expect(String(result.error)).toContain('must be http or https');
    expect(launchMock).not.toHaveBeenCalled();
  });

  it('rejects 169.254.169.254 on navigate', async () => {
    const result = parseToolResult(
      await browserTool.invoke({ action: 'navigate', url: 'http://169.254.169.254/' }) as string,
    );
    expect(String(result.error)).toContain('internal or reserved');
    expect(launchMock).not.toHaveBeenCalled();
  });

  it('rejects file:// URL on open', async () => {
    const result = parseToolResult(
      await browserTool.invoke({ action: 'open', url: 'file:///etc/passwd' }) as string,
    );
    expect(String(result.error)).toContain('must be http or https');
    expect(launchMock).not.toHaveBeenCalled();
  });

  it('accepts a normal https URL on navigate', async () => {
    const result = parseToolResult(
      await browserTool.invoke({ action: 'navigate', url: 'https://example.com/' }) as string,
    );
    expect(result).toMatchObject({ ok: true, url: 'https://example.com/' });
  });
});

describe('browser SSRF guard (DNS + route interception)', () => {
  it('refuses to navigate to a hostname that resolves to a private address', async () => {
    _setDnsLookupForTest(async () => [{ address: '10.0.0.5' }]);
    const result = parseToolResult(await browserTool.invoke({
      action: 'navigate',
      url: 'https://intranet.attacker.example/',
    }) as string);

    expect(String(result.error)).toMatch(/internal or reserved/);
    expect(launchMock).not.toHaveBeenCalled();
  });

  it('installs a catch-all route guard and blocks service workers on the browser context', async () => {
    await browserTool.invoke({ action: 'navigate', url: 'https://example.com' });

    expect(fakeBrowser.newContext).toHaveBeenCalledWith({ serviceWorkers: 'block' });
    expect(contextRouteMock).toHaveBeenCalledWith('**/*', guardRoute);
  });

  function makeRoute(url: string) {
    return {
      request: () => ({ url: () => url }),
      continue: mock(async () => {}),
      abort: mock(async (_code?: string) => {}),
    };
  }

  it('guardRoute aborts subresource requests to internal hosts', async () => {
    for (const url of ['http://127.0.0.1:8080/admin', 'http://[fd00:ec2::254]/', 'file:///etc/passwd']) {
      const route = makeRoute(url);
      await guardRoute(route);
      expect(route.abort).toHaveBeenCalledWith('blockedbyclient');
      expect(route.continue).not.toHaveBeenCalled();
    }
  });

  it('guardRoute continues public requests', async () => {
    const route = makeRoute('https://example.com/app.js');
    await guardRoute(route);
    expect(route.continue).toHaveBeenCalledTimes(1);
    expect(route.abort).not.toHaveBeenCalled();
  });
});
