import { describe, test, expect } from 'bun:test';
import { assertPublicHttpUrl } from './url-guard.js';

describe('assertPublicHttpUrl', () => {
  test('accepts a normal https URL', () => {
    expect(() => assertPublicHttpUrl('https://example.com/page')).not.toThrow();
  });

  test('accepts a normal http URL', () => {
    expect(() => assertPublicHttpUrl('http://example.com/page')).not.toThrow();
  });

  test('rejects file:// scheme', () => {
    expect(() => assertPublicHttpUrl('file:///etc/passwd')).toThrow(/must be http or https/);
  });

  test('rejects ftp:// scheme', () => {
    expect(() => assertPublicHttpUrl('ftp://evil.com/file')).toThrow(/must be http or https/);
  });

  test('rejects data: scheme', () => {
    expect(() => assertPublicHttpUrl('data:text/html,<h1>hi</h1>')).toThrow(/must be http or https/);
  });

  test('rejects chrome: scheme', () => {
    expect(() => assertPublicHttpUrl('chrome://settings')).toThrow(/must be http or https/);
  });

  test('rejects 127.0.0.1 (loopback)', () => {
    expect(() => assertPublicHttpUrl('http://127.0.0.1:11434/api/tags')).toThrow(/internal or reserved/);
  });

  test('rejects localhost', () => {
    expect(() => assertPublicHttpUrl('http://localhost:3000/')).toThrow(/internal or reserved/);
  });

  test('rejects 169.254.169.254 (link-local / cloud metadata)', () => {
    expect(() => assertPublicHttpUrl('http://169.254.169.254/latest/meta-data/')).toThrow(/internal or reserved/);
  });

  test('rejects 10.0.0.1 (private class A)', () => {
    expect(() => assertPublicHttpUrl('http://10.0.0.1/')).toThrow(/internal or reserved/);
  });

  test('rejects 172.16.0.1 (private class B)', () => {
    expect(() => assertPublicHttpUrl('http://172.16.0.1/')).toThrow(/internal or reserved/);
  });

  test('rejects 192.168.1.1 (private class C)', () => {
    expect(() => assertPublicHttpUrl('http://192.168.1.1/')).toThrow(/internal or reserved/);
  });

  test('rejects [::1] (IPv6 loopback)', () => {
    expect(() => assertPublicHttpUrl('http://[::1]/')).toThrow(/internal or reserved/);
  });

  test('rejects metadata.google.internal', () => {
    expect(() => assertPublicHttpUrl('http://metadata.google.internal/')).toThrow(/internal or reserved/);
  });

  test('accepts a public IP', () => {
    expect(() => assertPublicHttpUrl('https://1.1.1.1/')).not.toThrow();
  });

  test('rejects an unparseable URL', () => {
    expect(() => assertPublicHttpUrl('not-a-url')).toThrow(/Invalid URL/);
  });

  // --- Gap-closers beyond the source plan (E5) ---

  test('rejects 0.0.0.0 (unspecified / routes to localhost)', () => {
    expect(() => assertPublicHttpUrl('http://0.0.0.0:8080/')).toThrow(/internal or reserved/);
  });

  test('rejects [::] (IPv6 unspecified)', () => {
    expect(() => assertPublicHttpUrl('http://[::]/')).toThrow(/internal or reserved/);
  });

  test('rejects [::ffff:127.0.0.1] (IPv4-mapped loopback)', () => {
    expect(() => assertPublicHttpUrl('http://[::ffff:127.0.0.1]/')).toThrow(/internal or reserved/);
  });

  test('rejects [::ffff:169.254.169.254] (IPv4-mapped link-local)', () => {
    expect(() => assertPublicHttpUrl('http://[::ffff:169.254.169.254]/')).toThrow(/internal or reserved/);
  });
});
