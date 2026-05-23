import { describe, expect, it } from 'bun:test';
import { isE2EExternalModelError, isRetryableE2EExternalModelError } from './e2e-helpers.js';

describe('isRetryableE2EExternalModelError', () => {
  it('does not retry E2E timeout failures', () => {
    const error = new Error('E2E agent timed out after 600000ms.');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(false);
  });

  it('classifies bare timeout transport failures as external but not retryable', () => {
    for (const message of ['request timeout', 'timeout error', 'timed out']) {
      expect(isE2EExternalModelError(message)).toBe(true);
      expect(isRetryableE2EExternalModelError(message)).toBe(false);
    }
  });

  it('retries overload failures', () => {
    const error = new Error('provider overloaded_error: high demand');

    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('retries rate limit failures', () => {
    const error = new Error('429 Too Many Requests');

    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('retries 503 service temporarily unavailable failures', () => {
    const error = new Error('503 Service Temporarily Unavailable');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('retries 503 service is unavailable failures', () => {
    const error = new Error('503 service is unavailable');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('retries provider-first service unavailable failures', () => {
    const error = new Error('OpenAI service unavailable');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('retries provider API temporarily unavailable failures', () => {
    const error = new Error('Anthropic API is temporarily unavailable');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('does not retry auth failures', () => {
    const error = new Error('invalid api key');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(false);
  });

  it('does not retry billing failures', () => {
    const error = new Error('payment required');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(false);
  });

  it('does not classify valid BTC forecast abstention/no-trade text as external model error', () => {
    const answer = `
## BTC 24h Polymarket + Markov forecast

### Markov 9-Bucket Terminal Distribution
Status: 🚫 ABSTAINED
No 9-part density distribution is available because the Markov model is unavailable after the structural-break downgrade.

### Structural Break Diagnostic
BTC spot and Polymarket-implied odds diverged in a high demand tape, CI widening was applied, and leverage should be reduced.

### Final Arbitrator Verdict
NO_TRADE — wait for a cleaner entry/stop setup.
`;

    expect(isE2EExternalModelError(answer)).toBe(false);
    expect(isRetryableE2EExternalModelError(answer)).toBe(false);
  });

  it('does not classify domain prose about trading models as external model error', () => {
    const answer = 'Trading model shows service unavailable patterns';

    expect(isE2EExternalModelError(answer)).toBe(false);
    expect(isRetryableE2EExternalModelError(answer)).toBe(false);
  });

  it('still classifies provider model-unavailability failures as external model errors', () => {
    const error = new Error('Ollama provider model is unavailable');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(false);
  });

  it('classifies error-prefixed model-unavailability failures as external model errors', () => {
    const error = new Error('Error: Model is unavailable');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(false);
  });

  it('classifies bare model-unavailability failures as external model errors', () => {
    expect(isE2EExternalModelError('Model unavailable')).toBe(true);
    expect(isRetryableE2EExternalModelError('Model unavailable')).toBe(false);
  });
});
