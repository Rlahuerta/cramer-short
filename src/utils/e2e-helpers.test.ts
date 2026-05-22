import { describe, expect, it } from 'bun:test';
import { isE2EExternalModelError, isRetryableE2EExternalModelError } from './e2e-helpers.js';

describe('isRetryableE2EExternalModelError', () => {
  it('does not retry E2E timeout failures', () => {
    const error = new Error('E2E agent timed out after 600000ms.');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(false);
  });

  it('retries overload failures', () => {
    const error = new Error('provider overloaded_error: high demand');

    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('retries rate limit failures', () => {
    const error = new Error('429 Too Many Requests');

    expect(isRetryableE2EExternalModelError(error)).toBe(true);
  });

  it('does not retry auth failures', () => {
    const error = new Error('invalid api key');

    expect(isE2EExternalModelError(error)).toBe(true);
    expect(isRetryableE2EExternalModelError(error)).toBe(false);
  });
});
