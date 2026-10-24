import { afterEach, beforeEach, describe, expect, it, mock } from 'bun:test';
import type { TUI } from '@mariozechner/pi-tui';
import type { DisplayEvent } from '../agent/types.js';
import { ChatLogComponent } from '../components/chat-log.js';
import { renderCurrentQuery } from './cli-rendering.js';
import type { HistoryItem } from './types.js';

function stripAnsi(value: string): string {
  // eslint-disable-next-line no-control-regex
  return value.replace(/\x1b\[[0-9;]*m/g, '');
}

function pinTerminalSize(rows: number, columns: number): void {
  Object.defineProperty(process.stdout, 'rows', { value: rows, configurable: true });
  Object.defineProperty(process.stdout, 'columns', { value: columns, configurable: true });
}

const originalRows = process.stdout.rows;
const originalColumns = process.stdout.columns;

beforeEach(() => {
  pinTerminalSize(16, 80);
});

afterEach(() => {
  Object.defineProperty(process.stdout, 'rows', { value: originalRows, configurable: true });
  Object.defineProperty(process.stdout, 'columns', { value: originalColumns, configurable: true });
});

function snapshot(chatLog: ChatLogComponent): string {
  return chatLog.render(80).map(stripAnsi).join('\n');
}

function toolEvents(): { start: DisplayEvent; done: DisplayEvent } {
  const start: DisplayEvent = {
    id: 'tool-1',
    event: { type: 'tool_start', tool: 'get_stock_price', args: { ticker: 'AAPL' } },
    completed: false,
  };
  const done: DisplayEvent = {
    ...start,
    completed: true,
    endEvent: {
      type: 'tool_end',
      tool: 'get_stock_price',
      args: { ticker: 'AAPL' },
      result: '{"data":{"price":180.5}}',
      duration: 120,
    },
  };
  return { start, done };
}

export function buildShortTrace(): string {
  const chatLog = new ChatLogComponent({} as TUI);
  const steps: string[] = [];
  let item: HistoryItem = {
    id: 'golden-short',
    query: 'What is AAPL?',
    events: [],
    answer: '',
    status: 'processing',
  };
  const render = () => {
    renderCurrentQuery(chatLog, [item]);
    steps.push(snapshot(chatLog));
  };

  render();
  const { start, done } = toolEvents();
  item = { ...item, events: [...item.events, start] };
  render();
  item = { ...item, events: [done] };
  render();
  for (const chunk of ['AAPL is ', 'trading at ', '$180.50, ', 'up 1.2% ', 'today.']) {
    item = { ...item, answer: item.answer + chunk };
    render();
  }
  item = {
    ...item,
    status: 'complete',
    duration: 1500,
    tokenUsage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
  };
  render();
  return steps.join('\n=====\n');
}

export function buildLongTrace(): string {
  const chatLog = new ChatLogComponent({} as TUI);
  const steps: string[] = [];
  let item: HistoryItem = {
    id: 'golden-long',
    query: 'Give me a long answer',
    events: [],
    answer: '',
    status: 'processing',
  };

  for (let i = 1; i <= 12; i += 1) {
    item = { ...item, answer: `${item.answer}line ${i}\n` };
    renderCurrentQuery(chatLog, [item]);
    steps.push(snapshot(chatLog));
  }
  return steps.join('\n=====\n');
}

const GOLDEN_SHORT = `
❯ What is AAPL?                                                                 
=====

❯ What is AAPL?                                                                 

⚡ Stock Price(ticker=AAPL)                                                     
⎿  Searching...                                                                 
=====

❯ What is AAPL?                                                                 

✓ Stock Price(ticker=AAPL)                                                      
⎿  Received 1 fields in 120ms                                                   
=====

❯ What is AAPL?                                                                 

✓ Stock Price(ticker=AAPL)                                                      
⎿  Received 1 fields in 120ms                                                   

AAPL is                                                                         
=====

❯ What is AAPL?                                                                 

✓ Stock Price(ticker=AAPL)                                                      
⎿  Received 1 fields in 120ms                                                   

AAPL is trading at                                                              
=====

❯ What is AAPL?                                                                 

✓ Stock Price(ticker=AAPL)                                                      
⎿  Received 1 fields in 120ms                                                   

AAPL is trading at $180.50,                                                     
=====

❯ What is AAPL?                                                                 

✓ Stock Price(ticker=AAPL)                                                      
⎿  Received 1 fields in 120ms                                                   

AAPL is trading at $180.50, up 1.2%                                             
=====

❯ What is AAPL?                                                                 

✓ Stock Price(ticker=AAPL)                                                      
⎿  Received 1 fields in 120ms                                                   

AAPL is trading at $180.50, up 1.2% today.                                      
=====

❯ What is AAPL?                                                                 

✓ Stock Price(ticker=AAPL)                                                      
⎿  Received 1 fields in 120ms                                                   

AAPL is trading at $180.50, up 1.2% today.                                      

✻ 2s                                                                            `;
const GOLDEN_LONG = `
❯ Give me a long answer                                                         

line 1                                                                          
=====

❯ Give me a long answer                                                         

line 1                                                                          
line 2                                                                          
=====

❯ Give me a long answer                                                         

line 1                                                                          
line 2                                                                          
line 3                                                                          
=====

❯ Give me a long answer                                                         

line 1                                                                          
line 2                                                                          
line 3                                                                          
line 4                                                                          
=====

❯ Give me a long answer                                                         

line 1                                                                          
line 2                                                                          
line 3                                                                          
line 4                                                                          
line 5                                                                          
=====

❯ Give me a long answer                                                         

line 1                                                                          
line 2                                                                          
line 3                                                                          
line 4                                                                          
line 5                                                                          
line 6                                                                          
=====

❯ Give me a long answer                                                         

line 1                                                                          
line 2                                                                          
line 3                                                                          
line 4                                                                          
line 5                                                                          
line 6                                                                          
line 7                                                                          
=====

❯ Give me a long answer                                                         

line 1                                                                          
line 2                                                                          
line 3                                                                          
line 4                                                                          
line 5                                                                          
line 6                                                                          
line 7                                                                          
line 8                                                                          
=====

❯ Give me a long answer                                                         

…                                                                               
line 3                                                                          
line 4                                                                          
line 5                                                                          
line 6                                                                          
line 7                                                                          
line 8                                                                          
line 9                                                                          
=====

❯ Give me a long answer                                                         

…                                                                               
line 4                                                                          
line 5                                                                          
line 6                                                                          
line 7                                                                          
line 8                                                                          
line 9                                                                          
line 10                                                                         
=====

❯ Give me a long answer                                                         

…                                                                               
line 5                                                                          
line 6                                                                          
line 7                                                                          
line 8                                                                          
line 9                                                                          
line 10                                                                         
line 11                                                                         
=====

❯ Give me a long answer                                                         

…                                                                               
line 6                                                                          
line 7                                                                          
line 8                                                                          
line 9                                                                          
line 10                                                                         
line 11                                                                         
line 12                                                                         `;

describe('renderCurrentQuery — incremental streamed rendering', () => {
  it('does not rebuild the chat log for every streamed answer chunk', () => {
    const counters = { clearAll: 0, answerBoxConstructions: 0, appendText: 0 };
    const fakeAnswerComponent = {
      appendText: mock((_chunk: string) => {
        counters.appendText += 1;
      }),
      setText: mock((_text: string) => {}),
    };
    const fakeChatLog = {
      clearAll: mock(() => {
        counters.clearAll += 1;
      }),
      addQuery: mock((_query: string) => {}),
      resetToolGrouping: mock(() => {}),
      addInterrupted: mock(() => {}),
      addChild: mock((_component: unknown) => {}),
      startTool: mock(() => ({
        setComplete: () => {},
        setActive: () => {},
        setError: () => {},
        setApproval: () => {},
        setDenied: () => {},
      })),
      addContextCleared: mock(() => {}),
      finalizeAnswer: mock((_text: string) => {
        counters.answerBoxConstructions += 1;
        return fakeAnswerComponent;
      }),
      addPerformanceStats: mock(() => {}),
    } as unknown as ChatLogComponent;

    const events: DisplayEvent[] = [];
    let item: HistoryItem = {
      id: 'perf-1',
      query: 'stream a long answer',
      events,
      answer: '',
      status: 'processing',
    };

    renderCurrentQuery(fakeChatLog, [item]);
    for (let i = 0; i < 200; i += 1) {
      item = { ...item, answer: `${item.answer}x` };
      renderCurrentQuery(fakeChatLog, [item]);
    }

    expect(counters.clearAll).toBeLessThanOrEqual(3);
    expect(counters.answerBoxConstructions).toBeLessThanOrEqual(3);
    expect(counters.appendText).toBeGreaterThanOrEqual(197);
  });

  it('keeps the golden rendering of a fixed short event sequence', () => {
    expect(buildShortTrace()).toBe(GOLDEN_SHORT);
  });

  it('keeps the golden rendering of a truncated long answer', () => {
    expect(buildLongTrace()).toBe(GOLDEN_LONG);
  });
});
