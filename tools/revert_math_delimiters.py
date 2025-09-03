#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'docs'

# Inline: \( ... \)  -> $...$
inline_re = re.compile(r"\\\(([^\n]+?)\\\)")

# Single-line display: \[ ... \] -> $$...$$
singleline_block_re = re.compile(r"\\\[([^\n]+?)\\\]")

def process_file(path: Path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    out = []
    in_code = False
    for line in lines:
        stripped = line.lstrip()
        # toggle fenced code blocks
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        # Revert single-line display first
        line_conv = singleline_block_re.sub(r"$$\1$$", line)
        # Revert inline
        line_conv = inline_re.sub(r"$\1$", line_conv)
        out.append(line_conv)
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

