import { Container, Markdown, Spacer } from '@mariozechner/pi-tui';
import { formatResponseTui } from '../utils/ui/markdown-table.js';
import { markdownTheme } from '../theme.js';

export class AnswerBoxComponent extends Container {
  private readonly body: Markdown;
  private value = '';

  constructor(initialText = '') {
    super();
    this.addChild(new Spacer(1));
    this.body = new Markdown('', 0, 0, markdownTheme, { color: (line) => line });
    this.addChild(this.body);
    this.setText(initialText);
  }

  setText(text: string) {
    this.value = text;
    const rendered = formatResponseTui(text);
    const normalized = rendered.replace(/^\n+/, '');
    this.body.setText(normalized);
  }

  appendText(chunk: string) {
    this.setText(this.value + chunk);
  }
}
