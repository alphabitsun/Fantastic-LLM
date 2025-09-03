#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'docs'

inline_re = re.compile(r"\$(?!\$)([^\$\n]+?)\$(?!\$)")
singleline_block_re = re.compile(r"\$\$([^\$\n]+?)\$\$")

def convert_line_inline(line: str) -> str:
    def repl(m):
        content = m.group(1)
        # Heuristic: ensure it's likely math (letters or backslashes), not money like $100
        if re.search(r"[A-Za-z\\]", content):
            return f"\\({content}\\)"
        return m.group(0)
    return inline_re.sub(repl, line)

def process_file(path: Path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    out = []
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # toggle fenced code blocks
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue
        # Handle block math spanning multiple lines: $$ ... $$ on separate lines
        if stripped == '$$':
            # collect until next $$ line
            block = []
            i += 1
            while i < len(lines):
                if lines[i].lstrip() == '$$':
                    break
                block.append(lines[i])
                i += 1
            # now lines[i] is '$$' or EOF
            out.append('\\[')
            out.extend(block)
            out.append('\\]')
            # skip closing $$ if present
            if i < len(lines) and lines[i].lstrip() == '$$':
                i += 1
            continue
        # Convert single-line $$...$$ to \[...\]
        line_conv = singleline_block_re.sub(r"\\[\1\\]", line)
        # Convert inline $...$ to \(...\)
        line_conv = convert_line_inline(line_conv)
        out.append(line_conv)
        i += 1
    new_text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    if new_text != text:
        path.write_text(new_text, encoding='utf-8')
        return True
    return False

def main():
    changed = 0
    for md in ROOT.rglob('*.md'):
        if process_file(md):
            changed += 1
            print(f"updated: {md}")
    print(f"done. files changed: {changed}")

if __name__ == '__main__':
    main()

