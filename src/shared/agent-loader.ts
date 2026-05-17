export async function loadAgentCtor(): Promise<typeof import('../agent/agent.js').Agent> {
  const { Agent } = await import('../agent/agent.js');
  return Agent;
}
