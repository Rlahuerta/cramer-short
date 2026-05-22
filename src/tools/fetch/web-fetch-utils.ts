export type ExtractMode = "markdown" | "text";
export type HtmlExtractor = "htmlToMarkdown" | "readability";

export interface ReadableContentResult {
  text: string;
  title?: string;
  extractor: HtmlExtractor;
}

interface ReadabilityDocument {
  baseURI?: string;
}

type HtmlExtractionDeps = {
  Readability: new (
    document: ReadabilityDocument,
    options?: { charThreshold?: number },
  ) => {
    parse(): {
      content?: string | null;
      title?: string | null;
      textContent?: string | null;
    } | null | undefined;
  };
  parseHTML: (html: string) => { document: ReadabilityDocument };
};

function findPropertyDescriptor(target: object, property: string): PropertyDescriptor | undefined {
  let current: object | null = target;
  while (current) {
    const descriptor = Object.getOwnPropertyDescriptor(current, property);
    if (descriptor) {
      return descriptor;
    }
    current = Object.getPrototypeOf(current);
  }
  return undefined;
}

function hasWritableBaseUri(document: ReadabilityDocument): boolean {
  const descriptor = findPropertyDescriptor(document, "baseURI");
  if (!descriptor) {
    return Object.isExtensible(document);
  }
  return descriptor.writable === true || typeof descriptor.set === "function";
}

function isObjectRecord(value: unknown): value is Record<PropertyKey, unknown> {
  return typeof value === "object" && value !== null;
}

function isDomDocument(value: unknown): value is Document {
  return (
    isObjectRecord(value) &&
    typeof value.querySelector === "function" &&
    typeof value.createElement === "function"
  );
}

function decodeEntities(value: string): string {
  return value
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCharCode(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/gi, (_, dec) => String.fromCharCode(Number.parseInt(dec, 10)));
}

function stripTags(value: string): string {
  return decodeEntities(value.replace(/<[^>]+>/g, ""));
}

function normalizeWhitespace(value: string): string {
  return value
    .replace(/\r/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

export function htmlToMarkdown(html: string): { text: string; title?: string } {
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = titleMatch ? normalizeWhitespace(stripTags(titleMatch[1])) : undefined;
  let text = html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, "");
  text = text.replace(/<a\s+[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi, (_, href, body) => {
    const label = normalizeWhitespace(stripTags(body));
    if (!label) {
      return href;
    }
    return `[${label}](${href})`;
  });
  text = text.replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi, (_, level, body) => {
    const prefix = "#".repeat(Math.max(1, Math.min(6, Number.parseInt(level, 10))));
    const label = normalizeWhitespace(stripTags(body));
    return `\n${prefix} ${label}\n`;
  });
  text = text.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, (_, body) => {
    const label = normalizeWhitespace(stripTags(body));
    return label ? `\n- ${label}` : "";
  });
  text = text
    .replace(/<(br|hr)\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|section|article|header|footer|table|tr|ul|ol)>/gi, "\n");
  text = stripTags(text);
  text = normalizeWhitespace(text);
  return { text, title };
}

export function markdownToText(markdown: string): string {
  let text = markdown;
  text = text.replace(/!\[[^\]]*]\([^)]+\)/g, "");
  text = text.replace(/\[([^\]]+)]\([^)]+\)/g, "$1");
  text = text.replace(/```[\s\S]*?```/g, (block) =>
    block.replace(/```[^\n]*\n?/g, "").replace(/```/g, ""),
  );
  text = text.replace(/`([^`]+)`/g, "$1");
  text = text.replace(/^#{1,6}\s+/gm, "");
  text = text.replace(/^\s*[-*+]\s+/gm, "");
  text = text.replace(/^\s*\d+\.\s+/gm, "");
  return normalizeWhitespace(text);
}

export function truncateText(
  value: string,
  maxChars: number,
): { text: string; truncated: boolean } {
  if (value.length <= maxChars) {
    return { text: value, truncated: false };
  }
  return { text: value.slice(0, maxChars), truncated: true };
}

const ARTICLE_LIKE_HTML_PATTERN =
  /<(?:!doctype|html|head|body|main|article|section|header|footer|title|h1|h2|h3|h4|h5|h6|time|figure|figcaption|blockquote)\b/i;
const HTML_OPENING_TAG_PATTERN = /<(?!\/|!)([a-z][\w:-]*)\b/gi;
const PARAGRAPH_LIKE_TAG_PATTERN = /<(?:p|li)\b/gi;

function shouldFastPathHtmlFallback(html: string): boolean {
  const trimmed = html.trim();
  if (trimmed.length === 0) {
    return true;
  }
  if (trimmed.length > 4_000 || ARTICLE_LIKE_HTML_PATTERN.test(trimmed)) {
    return false;
  }
  const openingTagCount = trimmed.match(HTML_OPENING_TAG_PATTERN)?.length ?? 0;
  if (openingTagCount > 6) {
    return false;
  }
  const paragraphLikeTagCount = trimmed.match(PARAGRAPH_LIKE_TAG_PATTERN)?.length ?? 0;
  return paragraphLikeTagCount <= 1;
}

function warnReadabilityFallback(error: Error): void {
  console.warn(
    `[web_fetch] readability extraction failed; falling back to htmlToMarkdown: ${error.message}`,
  );
}

export async function extractReadableContent(params: {
  html: string;
  url: string;
  extractMode: ExtractMode;
  loadDeps?: () => Promise<HtmlExtractionDeps>;
}): Promise<ReadableContentResult> {
  const fallback = (): ReadableContentResult => {
    const rendered = htmlToMarkdown(params.html);
    if (params.extractMode === "text") {
      const text = markdownToText(rendered.text) || normalizeWhitespace(stripTags(params.html));
      return { text, title: rendered.title, extractor: "htmlToMarkdown" };
    }
    return { ...rendered, extractor: "htmlToMarkdown" };
  };
  if (shouldFastPathHtmlFallback(params.html)) {
    return fallback();
  }
  try {
    const { Readability, parseHTML } = params.loadDeps
      ? await params.loadDeps()
      : await (async (): Promise<HtmlExtractionDeps> => {
          const [{ Readability }, { parseHTML }] = await Promise.all([
            import("@mozilla/readability"),
            import("linkedom"),
          ]);
          return {
            Readability: class {
              private readonly reader: InstanceType<typeof Readability>;

              constructor(document: ReadabilityDocument, options?: { charThreshold?: number }) {
                if (!isDomDocument(document)) {
                  throw new Error("parseHTML did not return a DOM document");
                }
                this.reader = new Readability(document, options);
              }

              parse(): {
                content?: string | null;
                title?: string | null;
                textContent?: string | null;
              } | null {
                const parsed = this.reader.parse();
                if (!parsed) {
                  return null;
                }
                return {
                  content: typeof parsed.content === "string" ? parsed.content : null,
                  title: parsed.title,
                  textContent: parsed.textContent,
                };
              }
            },
            parseHTML,
          };
        })();
    const { document } = parseHTML(params.html);
    if (hasWritableBaseUri(document)) {
      document.baseURI = params.url;
    }
    const reader = new Readability(document, { charThreshold: 0 });
    const parsed = reader.parse();
    if (!parsed?.content) {
      return fallback();
    }
    const title = parsed.title || undefined;
    if (params.extractMode === "text") {
      const text = normalizeWhitespace(parsed.textContent ?? "");
      return text ? { text, title, extractor: "readability" } : fallback();
    }
    const rendered = htmlToMarkdown(parsed.content);
    return { text: rendered.text, title: title ?? rendered.title, extractor: "readability" };
  } catch (error) {
    if (error instanceof Error) {
      warnReadabilityFallback(error);
      return fallback();
    }
    throw error;
  }
}
