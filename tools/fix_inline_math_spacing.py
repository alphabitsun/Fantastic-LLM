#!/usr/bin/env python3
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'docs'


def fix_segment(seg: str) -> str:
    # Walk through segment and ensure a space before inline math $...$
    # Only for single $ delimiters (not $$)
    out = []
    i = 0
    n = len(seg)
    changed = False
    while i < n:
        ch = seg[i]
        if ch == '$':
            # skip if part of $$
            prev = seg[i-1] if i > 0 else ''
            nxt = seg[i+1] if i + 1 < n else ''
            if prev == '$' or nxt == '$':
                out.append(ch)
                i += 1
                continue
            # find closing single $
            j = i + 1
            while j < n and seg[j] != '$':
                # do not cross newline inside segment (segments are single-line pieces)
                j += 1
            if j < n and seg[j] == '$':
                # ensure a space before opening $
                if len(out) > 0 and not out[-1].isspace():
                    out.append(' ')
                    changed = True
                # copy the inline math as-is
                out.append(seg[i:j+1])
                i = j + 1
                continue
            else:
                # unmatched $, just copy
                out.append(ch)
                i += 1
                continue
        out.append(ch)
        i += 1
    return ''.join(out), changed


def process_line(line: str) -> (str, bool):
    # Split by inline code spans marked with backticks and only transform outside
    parts = re.split(r"(`[^`]*`)", line)
    changed_any = False
    for idx in range(0, len(parts)):
        # odd indices are code spans kept as-is
        if idx % 2 == 0:
            fixed, changed = fix_segment(parts[idx])
            parts[idx] = fixed
            changed_any = changed_any or changed
    return ''.join(parts), changed_any


def process_file(path: Path) -> bool:
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    out_lines = []
    in_code_fence = False
    changed_any = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_fence = not in_code_fence
            out_lines.append(line)
            continue
        if in_code_fence:
            out_lines.append(line)
            continue
        fixed, changed = process_line(line)
        out_lines.append(fixed)
        changed_any = changed_any or changed
    if changed_any:
        path.write_text('\n'.join(out_lines) + ('\n' if text.endswith('\n') else ''), encoding='utf-8')
    return changed_any


def main():
    changed = 0
    for md in ROOT.rglob('*.md'):
        if process_file(md):
            changed += 1
            print(f"fixed: {md}")
    print(f"done. files changed: {changed}")


if __name__ == '__main__':
    main()

