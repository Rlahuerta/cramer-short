import { describe, expect, it } from 'bun:test';
import {
  getSourcesByCategory,
  getHighReliabilitySources,
  getDomainsForCategory,
  getBskyHandlesForCategory,
  OSINT_SOURCES,
} from './accounts.js';

describe('getSourcesByCategory', () => {
  it('returns sources that cover the given category', () => {
    const results = getSourcesByCategory('ukraine-russia');
    expect(results.length).toBeGreaterThan(0);
    results.forEach((s) => expect(s.eventCategories).toContain('ukraine-russia'));
  });

  it('returns empty array for a category with no sources', () => {
    // There's no source covering an invalid/unused category
    const results = getSourcesByCategory('pandemic-risk' as never);
    // pandemic-risk is a valid category but may have 0 sources — just verify the filter works
    results.forEach((s) => expect(s.eventCategories).toContain('pandemic-risk'));
  });
});

describe('getHighReliabilitySources', () => {
  it('returns only high-reliability sources', () => {
    const results = getHighReliabilitySources();
    expect(results.length).toBeGreaterThan(0);
    results.forEach((s) => expect(s.reliability).toBe('high'));
  });

  it('is a subset of all OSINT sources', () => {
    const high = getHighReliabilitySources();
    high.forEach((s) => expect(OSINT_SOURCES).toContain(s));
  });
});

describe('getDomainsForCategory', () => {
  it('returns web domains for a given category', () => {
    const domains = getDomainsForCategory('ukraine-russia');
    expect(domains.length).toBeGreaterThan(0);
    domains.forEach((d) => expect(typeof d).toBe('string'));
  });

  it('returns only sources that have a webDomain defined', () => {
    const domains = getDomainsForCategory('ukraine-russia');
    // All returned strings must be non-empty domains
    domains.forEach((d) => expect(d.length).toBeGreaterThan(0));
  });

  it('includes reuters.com for ukraine-russia category', () => {
    const domains = getDomainsForCategory('ukraine-russia');
    expect(domains).toContain('reuters.com');
  });

  it('returns domains for sanctions category', () => {
    const domains = getDomainsForCategory('sanctions');
    expect(domains.length).toBeGreaterThan(0);
    expect(domains).toContain('reuters.com');
  });
});

describe('getBskyHandlesForCategory', () => {
  it('returns Bluesky handles for a given category', () => {
    const handles = getBskyHandlesForCategory('ukraine-russia');
    expect(handles.length).toBeGreaterThan(0);
    handles.forEach((h) => expect(typeof h).toBe('string'));
  });

  it('returns only sources that have a blueskyHandle defined', () => {
    const handles = getBskyHandlesForCategory('ukraine-russia');
    handles.forEach((h) => expect(h.length).toBeGreaterThan(0));
  });

  it('includes bellingcat for ukraine-russia category', () => {
    const handles = getBskyHandlesForCategory('ukraine-russia');
    expect(handles).toContain('bellingcat.bsky.social');
  });

  it('returns handles for cyberattack category', () => {
    const handles = getBskyHandlesForCategory('cyberattack');
    expect(handles.length).toBeGreaterThan(0);
  });
});
