import { Container, Markdown, Spacer } from '@mariozechner/pi-tui';
import { formatResponseTui } from '../utils/ui/markdown-table.js';
import { markdownTheme } from '../theme.js';

export class AnswerBoxComponent extends Container {
  private readonly body: Markdown;

  constructor(initialText = '') {
    super();
    this.addChild(new Spacer(1));
    this.body = new Markdown('', 0, 0, markdownTheme, { color: (line) => line });
    this.addChild(this.body);
    this.setText(initialText);
  }

  setText(text: string) {
    const rendered = formatResponseTui(text);
    const normalized = rendered.replace(/^\n+/, '');
    this.body.setText(normalized);
  }
}
