"""Structural and theme sanity checks for the report template.

Catches two things that stay invisible until the page is published: unbalanced
tags, and a CSS custom property defined only inside a dark-mode block, which
renders one theme's text on the other theme's ground.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "source", "track", "wbr"}


class TagBalance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append("stray closing tag: " + tag)
        elif self.stack[-1] != tag:
            self.errors.append(f"closing {tag} but innermost open is {self.stack[-1]}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True)
    args = ap.parse_args()

    html = args.template.read_text(encoding="utf-8")
    problems: list[str] = []

    checker = TagBalance()
    checker.feed(html)
    if checker.stack:
        problems.append(f"unclosed tags at EOF: {checker.stack}")
    problems.extend(checker.errors)

    # Every var() reference must resolve on bare :root, not only inside a
    # theme override.
    root_match = re.search(r":root \{(.*?)\n  \}", html, re.S)
    if not root_match:
        problems.append("could not locate the bare :root token block")
    else:
        defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", root_match.group(1)))
        used = set(re.findall(r"var\((--[a-z0-9-]+)", html))
        undefined = sorted(used - defined)
        if undefined:
            problems.append(f"tokens used but not defined on bare :root: {undefined}")
        unused = sorted(defined - used)
        if unused:
            print(f"note: tokens defined but never used: {unused}")

    leftover = re.findall(r"\{\{[A-Z]+:[a-z0-9_]+\}\}", html)
    if leftover:
        print(f"note: {len(leftover)} figure placeholders awaiting build")

    if not re.search(r"body\s*\{[^}]*background:\s*var\(", html):
        problems.append("body does not set an explicit background from a token")

    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)

    print("report template OK: tags balanced, tokens complete, body ground explicit")


if __name__ == "__main__":
    main()
