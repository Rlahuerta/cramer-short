import { Container, Text } from '@mariozechner/pi-tui';
import type { ApprovalDecision } from '../agent/types.js';
import { createApprovalSelector } from './select-list.js';
import { theme } from '../theme.js';
import { formatResponseTui } from '../utils/ui/markdown-table.js';

function formatToolLabel(tool: string): string {
  return tool
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export class ApprovalPromptComponent extends Container {
  readonly selector: ReturnType<typeof createApprovalSelector>;
  onSelect?: (decision: ApprovalDecision) => void;

  constructor(tool: string, args: Record<string, unknown>) {
    super();
    this.selector = createApprovalSelector((decision) => this.onSelect?.(decision));
    const width = Math.max(20, process.stdout.columns ?? 80);
    const border = theme.warning('─'.repeat(width));
    // The path is model-controlled: strip terminal control sequences and keep it on one line
    // so a newline cannot forge an extra option line above the selector.
    const path = formatResponseTui(
      String((args.path as string) || (args.action as string) || '<unknown>'),
    ).replace(/\n/g, ' ');

    // Show a short content preview so users can make an informed approval decision
    const rawContent = formatResponseTui(
      (args.content as string) ||
        (args.text as string) ||
        (args.new_string as string) ||
        '',
    );
    const previewText = rawContent.trim().replace(/\s+/g, ' ').slice(0, 100);
    const preview = previewText
      ? (previewText.length < rawContent.trim().replace(/\s+/g, ' ').length
          ? `"${previewText}…"`
          : `"${previewText}"`)
      : null;

    this.addChild(new Text(border, 0, 0));
    this.addChild(new Text(theme.warning(theme.bold('Permission required')), 0, 0));
    this.addChild(new Text(`${formatToolLabel(tool)} ${path}`, 0, 0));
    if (preview) {
      this.addChild(new Text(theme.muted(preview), 0, 0));
    }
    this.addChild(new Text(theme.muted('Do you want to allow this?'), 0, 0));
    this.addChild(new Text('', 0, 0));
    this.addChild(this.selector);
    this.addChild(new Text('', 0, 0));
    this.addChild(new Text(theme.muted('Enter to confirm · esc to deny'), 0, 0));
    this.addChild(new Text(border, 0, 0));
  }
}
