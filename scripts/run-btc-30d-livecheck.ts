import { Agent } from '../src/agent/agent.js';
import type { AgentEvent, DoneEvent, ToolEndEvent, ToolStartEvent } from '../src/agent/types.js';

const QUERY = '--deep Provide a Bitcoin forecast based on markov chain and polymarket for the next 30 days';
const MODEL = process.env.E2E_MODEL ?? 'ollama:minimax-m2.7:cloud';
const MAX_ITERATIONS = 40;

function isToolStartEvent(event: AgentEvent): event is ToolStartEvent {
  return event.type === 'tool_start';
}

function isToolEndEvent(event: AgentEvent): event is ToolEndEvent {
  return event.type === 'tool_end';
}

function isDoneEvent(event: AgentEvent): event is DoneEvent {
  return event.type === 'done';
}

const agent = await Agent.create({
  model: MODEL,
  maxIterations: MAX_ITERATIONS,
  memoryEnabled: false,
});

const events: AgentEvent[] = [];

for await (const event of agent.run(QUERY)) {
  events.push(event);
  if (isToolStartEvent(event)) {
    console.log(`TOOL_START ${event.tool} ${JSON.stringify(event.args)}`);
  } else if (isToolEndEvent(event)) {
    console.log(`TOOL_END ${event.tool}`);
    console.log(event.result);
    console.log('TOOL_END_DONE');
  } else if (isDoneEvent(event)) {
    console.log('DONE_EVENT');
    console.log(JSON.stringify({
      iterations: event.iterations,
      totalTime: event.totalTime,
      answer: event.answer,
      toolCalls: event.toolCalls.map((toolCall) => ({ tool: toolCall.tool, args: toolCall.args })),
    }));
  }
}
