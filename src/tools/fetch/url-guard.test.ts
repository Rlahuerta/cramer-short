import { afterEach, describe, expect, it, test } from 'bun:test';
import { _setDnsLookupForTest, assertPublicHttpUrl, assertPublicUrl } from './url-guard.js';

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

  test('rejects trailing-dot and *.localhost hosts', () => {
    expect(() => assertPublicHttpUrl('http://localhost./')).toThrow(/internal or reserved/);
    expect(() => assertPublicHttpUrl('http://app.localhost/')).toThrow(/internal or reserved/);
  });

  test('rejects CGNAT, ULA, link-local and NAT64 literals', () => {
    for (const url of ['http://100.64.0.1/', 'http://[fd00:ec2::254]/', 'http://[fe80::1]/', 'http://[64:ff9b::7f00:1]/']) {
      expect(() => assertPublicHttpUrl(url)).toThrow(/internal or reserved/);
    }
  });
});

const PUBLIC = [{ address: '93.184.216.34' }];

afterEach(() => _setDnsLookupForTest());

describe('assertPublicUrl', () => {
  it('allows public http(s) hosts', async () => {
    _setDnsLookupForTest(async () => PUBLIC);
    await expect(assertPublicUrl('https://example.com/page')).resolves.toBeUndefined();
    await expect(assertPublicUrl('http://[2606:4700::1111]/')).resolves.toBeUndefined();
  });

  for (const url of ['file:///etc/passwd', 'ftp://example.com/x', 'data:text/html,hi', 'not a url']) {
    it(`rejects non-http URL ${url}`, async () => {
      await expect(assertPublicUrl(url)).rejects.toThrow(/must be http or https/);
    });
  }

  for (const url of [
    'http://localhost/',
    'http://localhost./',
    'http://foo.localhost/',
    'http://FOO.LOCALHOST../',
    'http://metadata.google.internal./',
    'http://127.0.0.1/',
    'http://2130706433/', // decimal 127.0.0.1
    'http://0x7f.1/', // hex 127.0.0.1
    'http://0.0.0.0/',
    'http://10.1.2.3/',
    'http://100.64.0.1/', // CGNAT
    'http://169.254.169.254/latest/meta-data/',
    'http://172.16.0.1/',
    'http://192.168.1.1/',
    'http://[::1]/',
    'http://[::]/',
    'http://[fc00::1]/',
    'http://[fd00:ec2::254]/', // AWS IMDS v6
    'http://[fe80::1]/',
    'http://[::ffff:127.0.0.1]/', // IPv4-mapped
    'http://[::ffff:169.254.169.254]/',
    'http://[64:ff9b::7f00:1]/', // NAT64
  ]) {
    it(`rejects internal host ${url}`, async () => {
      _setDnsLookupForTest(async () => PUBLIC);
      await expect(assertPublicUrl(url)).rejects.toThrow(/internal or reserved/);
    });
  }

  it('rejects hostnames when ANY resolved address is private', async () => {
    _setDnsLookupForTest(async () => [{ address: '93.184.216.34' }, { address: '10.0.0.5' }]);
    await expect(assertPublicUrl('https://rebind.example/')).rejects.toThrow(/internal or reserved/);
  });

  it('rejects hostnames resolving to private IPv6', async () => {
    _setDnsLookupForTest(async () => [{ address: 'fd12::1' }]);
    await expect(assertPublicUrl('https://v6.example/')).rejects.toThrow(/internal or reserved/);
  });

  it('fails closed when DNS resolution fails', async () => {
    _setDnsLookupForTest(async () => {
      throw new Error('ENOTFOUND');
    });
    await expect(assertPublicUrl('https://nx.example/')).rejects.toThrow(/Could not resolve/);
  });
});
