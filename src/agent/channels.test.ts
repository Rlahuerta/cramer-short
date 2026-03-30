import { describe, it, expect } from 'bun:test';
import { getChannelProfile } from './channels.js';

describe('getChannelProfile', () => {
  it('returns CLI profile when no channel argument given', () => {
    const profile = getChannelProfile();
    expect(profile.label).toBe('CLI');
  });

  it('returns CLI profile for "cli"', () => {
    const profile = getChannelProfile('cli');
    expect(profile.label).toBe('CLI');
  });

  it('returns WhatsApp profile for "whatsapp"', () => {
    const profile = getChannelProfile('whatsapp');
    expect(profile.label).toBe('WhatsApp');
  });

  it('falls back to CLI profile for unknown channel', () => {
    const profile = getChannelProfile('telegram');
    expect(profile.label).toBe('CLI');
  });

  it('falls back to CLI for undefined channel', () => {
    const profile = getChannelProfile(undefined);
    expect(profile.label).toBe('CLI');
  });

  it('CLI profile has non-null tables section', () => {
    const profile = getChannelProfile('cli');
    expect(profile.tables).not.toBeNull();
    expect(typeof profile.tables).toBe('string');
  });

  it('WhatsApp profile has tables: null', () => {
    const profile = getChannelProfile('whatsapp');
    expect(profile.tables).toBeNull();
  });

  it('CLI profile has behavior and responseFormat arrays', () => {
    const profile = getChannelProfile('cli');
    expect(Array.isArray(profile.behavior)).toBe(true);
    expect(profile.behavior.length).toBeGreaterThan(0);
    expect(Array.isArray(profile.responseFormat)).toBe(true);
    expect(profile.responseFormat.length).toBeGreaterThan(0);
  });

  it('WhatsApp profile has a preamble mentioning WhatsApp', () => {
    const profile = getChannelProfile('whatsapp');
    expect(profile.preamble.toLowerCase()).toContain('whatsapp');
  });

  it('each profile has required fields', () => {
    for (const channel of ['cli', 'whatsapp']) {
      const profile = getChannelProfile(channel);
      expect(typeof profile.label).toBe('string');
      expect(typeof profile.preamble).toBe('string');
      expect(Array.isArray(profile.behavior)).toBe(true);
      expect(Array.isArray(profile.responseFormat)).toBe(true);
    }
  });
});
